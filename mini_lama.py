#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback



os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'



import numpy as np
import torch

import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
import PIL.Image as Image
import cv2 as cv

from saicinpainting.training.trainers import load_checkpoint

LOGGER = logging.getLogger(__name__)


def mini_load_image(img):
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255

    return out_img
            
def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img
def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')
def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod
        
def move_to_device(obj, device):
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f'Unexpected type {type(obj)}')


def smart_erase(image, mask):
    try:
        image = mini_load_image(image)
        mask = mini_load_image(mask)
        
        with open('configs\\prediction\\mini.yaml', 'r') as f:
            predict_config = OmegaConf.create(yaml.safe_load(f))
        
        predict_config.model.path = 'big-lama'
        predict_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location = predict_config.device)
        model.freeze()
        
        model.to(device)
        
        dataset = dict(image=image, mask=mask[None, ...])


        if predict_config.dataset.pad_out_to_modulo is not None and predict_config.dataset.pad_out_to_modulo > 1:
            dataset['unpad_to_size'] = dataset['image'].shape[1:]
            dataset['image'] = pad_img_to_modulo(dataset['image'], predict_config.dataset.pad_out_to_modulo)
            dataset['mask'] = pad_img_to_modulo(dataset['mask'], predict_config.dataset.pad_out_to_modulo)
        
        batch = default_collate([dataset])

        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = model(batch)                    
            cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad_to_size = batch.get('unpad_to_size', None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')

        return cur_res
        

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)

def main():
    img_filename = 'lama_sample_images\\000069.png'
    mask_filename = 'lama_sample_images\\000069_mask.png'
    image = cv.imread(img_filename)
    mask = cv.imread(mask_filename)[:,:,0]
    
    cur_res = smart_erase(image, mask)
    cv.imwrite("out.png", cur_res)
    
if __name__ == '__main__':
    main()
