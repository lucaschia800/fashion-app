import json
import os 
import random
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2  
import torchvision.datasets as datasets
import os
from tqdm import tqdm 
from PIL import Image
import numpy as np
import json
import copy
import typing
import classes.multiheaded_Fefficient as mhf
from torchvision.models import EfficientNet_V2_M_Weights
import requests


def get_model(ckpt_path):
    model = mhf.MultiHead_FEfficientNet(ckpt_path = ckpt_path)
    model.eval()
    transforms = EfficientNet_V2_M_Weights.DEFAULT.transforms()
    
    return model, transforms

def detect_attributes(model, dataset, letter, batch_size = 16, device = 'cuda', transforms = None):
    for designer in dataset:
        if not designer['Designer_Name'].startswith(letter):
            continue

        batch_imgs = []
        for show in designer['Shows']:
            for look in show['Looks']:
                look_url = look['Look_Url']
                try:
                    img = requests.get(look_url)
                    img = Image.open(img)
                except:
                    print(f"Error: {look_url}")
                    continue
                for box in look['Garments']['boxes']:
                    x1, y1, x2, y2 = box
                    image = image.crop((x1, y1, x2, y2))
                    image = transforms(image)

                    batch_imgs.append(image)

        if len(batch_imgs) == batch_size:
            _infer_batch(model, batch_imgs, device)
            batch_imgs.clear()


def _infer_batch(model, img_list, device):
    batch = torch.stack(img_list).to(device)          # [B,3,H,W]
    with torch.no_grad():
        logits = model(batch)                         # dict[str, Tensor], each [B, C]

    return logits

def main():
    model, transforms = get_model('weights/Mefficientnet_v2_freeze_ckpt2.pth')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = json.load(open('letter_A.json'))
    detect_attributes(model, dataset, 'A', device = device, transforms = transforms)



