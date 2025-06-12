import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2  
import torchvision.datasets as datasets
from torchvision.models import efficientnet_v2_m
from torchvision.models import EfficientNet_V2_M_Weights
import os
from tqdm import tqdm 
from PIL import Image
import numpy as np
import mlflow
import mlflow.pytorch
import json
import copy

class CustomDataset(Dataset):
    def __init__(self, data_path, split, transforms=None, categories = {'gender' : 3, 'material' : 23, 'pattern' : 18, 'style' : 10, 'sleeve' : 4, 'category': 48, 'color' : 19}): #categories is 1 indexed ie len()
        data = json.load(open(data_path, 'r'))
        self.split = split
        self.transforms = transforms
        self.data = data['annotations']
        self.categories = categories

    def __getitem__(self, idx):

        image_path = "imat_data/img/" + self.data[idx]['imageId'] + ".jpg" if self.split == "Train" \
            else "imat_data/img_val/" + self.data[idx]['imageId'] + ".jpg"
        image = Image.open(image_path).convert("RGB")

        label_map = {}

        for category, num_classes in self.categories.items():
            category_values = self.data[idx]['labelId'].get(category, [])
            category_values = torch.tensor(category_values, dtype=torch.long)
            labels = F.one_hot(category_values, num_classes=num_classes).sum(dim=0).float()
            label_map[category] = labels
            

        if self.transforms is not None:
            image = self.transforms(image)


        return image, label_map

    def __len__(self):
        return len(self.data)


def get_transform():
    weights = EfficientNet_V2_M_Weights.DEFAULT
    transforms = weights.transforms()
    return transforms