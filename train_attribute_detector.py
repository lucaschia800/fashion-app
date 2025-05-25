import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import json
import io
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import Subset
import copy
from torch.cuda.amp import autocast, GradScaler


class CustomDataset(Dataset):
    def __init__(self, hdf5_file, transforms=None):
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.transforms = transforms
        self.images = self.hdf5_file['images']
        self.bounding_boxes = self.hdf5_file['bounding_boxes']
        self.labels = self.hdf5_file['labels']

    def __getitem__(self, idx):

        image = self.images[idx]
        image = Image.open(io.BytesIO(image)).convert("RGB")
        image = np.array(image)

        boxes = np.array(self.bounding_boxes[idx], dtype=np.float32).reshape(-1, 4)
        labels = [int(label) for label in self.labels[idx].decode('utf-8').split(",")]
        if self.transforms is not None:
            transformed = self.transforms(image = image, bboxes = boxes, labels = labels)

        image = torch.as_tensor(transformed['image'], dtype = torch.float32) / 255.0
        boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
        labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

  

        return image, target

    def __len__(self):
        return len(self.images)


def get_transform(train):
    if train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        transform = A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    return transform



def get_model(num_classes):
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace the classifier with a new one for transfer learning
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Unfreeze last few layers for fine-tuning
    for name, param in model.named_parameters():
        if "layer 3" in name or "layer4" in name or "layer3" in name:  
            param.requires_grad = True
        else:
            param.requires_grad = False

    

    return model

def train_model(model, data_loader, val_loader, optimizer, device, batch_size, num_epochs=75):
    model.to(device)

    best_loss = float('inf')
    best_model_weights = None
    patience = 10
    early_stop = False

    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(data_loader):

            if i % 500 == 0:
                print('Photos Processed:' + str(i * batch_size))

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values()) #Investigate whether I should be using the sum of all losses

            
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

        if epoch > 9:
            val_loss = validate(model, val_loader, device)
            compute_map(model, val_loader, device)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here
                patience = 10  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    print('Early stop at epochL:' + str(epoch))
                    early_stop = True
                    break

    if early_stop:
        return best_model_weights
    else:
        return model.state_dict()

def validate(model, data_loader, device):
    model.train()
    val_loss = 0
    print('Validating')
    val_loss_class = 0
    val_loss_bbox = 0
    val_loss_obj = 0
    val_loss_rpn = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            val_loss += losses.item()
            val_loss_class += loss_dict['loss_classifier'].item()
            val_loss_bbox += loss_dict['loss_box_reg'].item()
            val_loss_obj += loss_dict['loss_objectness'].item()
            val_loss_rpn += loss_dict['loss_rpn_box_reg'].item()

    print(f"Loss classifier: {val_loss_class}, Loss box reg: {val_loss_bbox}, Loss objectnes {val_loss_obj}, Loss RPN {val_loss_rpn}")

    avg_val_loss = val_loss / len(data_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def compute_map(model, data_loader, device):
    print('Calculating MAP')
    metric = MeanAveragePrecision(iou_type = 'bbox')
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)
            predictions = [{k: v.to(device) for k, v in prediction.items()} for prediction in predictions]

            metric.update(predictions, targets)
        print(metric.compute())


# Main execution
if __name__ == "__main__":
    # Setup
    
    # print("CUDA Available: ", torch.cuda.is_available())
    # print("Number of GPUs: ", torch.cuda.device_count())
    # print("GPU Name: ", torch.cuda.get_device_name(0))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # Create datasets and instantiate dataloader
    batch_size = 16
    training_data = CustomDataset(hdf5_file = 'h5 files/train.h5', transforms=get_transform(train=True))
    subset_data = torch.utils.data.Subset(training_data, random.sample(range(170000), 2000))
    train_loader = DataLoader(subset_data, batch_size= batch_size, shuffle=True, num_workers= 6 )

    validation_data = CustomDataset(hdf5_file = 'h5 files/validation.h5' , transforms=get_transform(train=False))
    subset_val = torch.utils.data.Subset(validation_data, random.sample(range(4000), 200))
    val_loader = DataLoader(subset_val, batch_size = 16, collate_fn=lambda x: tuple(zip(*x)), num_workers = 6)

    num_classes = 14
    # Get model
    model = get_model(num_classes)


    
    # Define different learning rates
    lr_backbone = 0.0000714  # For fine-tuning backbone layers
    lr_rpn = 0.000714    # For RPN
    lr_roi_heads = 0.0714 # For RoI heads

    # Create parameter groups
    params = [
        {'params': [p for n, p in model.backbone.body.named_parameters() if p.requires_grad], 'lr': lr_backbone},
        {'params': model.backbone.fpn.parameters(), 'lr': lr_backbone},  # FPN parameters
        {'params': model.rpn.parameters(), 'lr': lr_rpn},
        {'params': model.roi_heads.parameters(), 'lr': lr_roi_heads},
    ]

    optimizer = torch.optim.AdamW(params, betas=(0.9, 0.999), weight_decay=1e-2  )



    torch.save(train_model(model, train_loader, val_loader, optimizer, device, batch_size),'rcnn_resnet_model.pth')
