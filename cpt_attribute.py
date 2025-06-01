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
    def __init__(self, data_path, split, transforms=None):
        data = json.load(open(data_path, 'r'))
        self.split = split
        self.transforms = transforms
        self.data = data['annotations']

    def __getitem__(self, idx):

        image_path = "imat_data/img/" + self.data[idx]['imageId'] + ".jpg" if self.split == "Train" \
            else "imat_data/img_val/" + self.data[idx]['imageId'] + ".jpg"
        image = Image.open(image_path).convert("RGB")

        labels = [int(label) for label in self.data[idx]['labelId']]
        labels = torch.tensor(labels, dtype=torch.long)
        labels = F.one_hot(labels, num_classes=131).sum(dim=0).float()
        if self.transforms is not None:
            image = self.transforms(image)


        return image, labels

    def __len__(self):
        return len(self.data)


def get_transform():
    weights = EfficientNet_V2_M_Weights.DEFAULT
    transforms = weights.transforms()
    return transforms



def get_model(num_classes = 131, path = None):
    # Load pre-trained model
    weights = EfficientNet_V2_M_Weights.DEFAULT  
    model = efficientnet_v2_m(weights=weights)  

    # Replace the classifier with a new one for transfer learning
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    if path:
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint) #keep it strict for now

    # Unfreeze last few layers for fine-tuning
    for name, param in model.named_parameters():
        if any(x in name for x in ["features.7", "features.8", "classifier"]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

def train_model(model, data_loader, val_loader, optimizer, device, criterion, num_epochs=75):
    model.to(device)

    best_loss = float('inf')
    best_model_weights = None
    patience = 10
    early_stop = False

    with mlflow.start_run():

        mlflow.log_params({
            "model_type": "efficientnet_v2_m",
            "batch_size": data_loader.batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "num_epochs": num_epochs,
            "optimizer": type(optimizer).__name__
        })

        for epoch in tqdm(range(num_epochs), desc = "Training"):
            model.train()
            epoch_loss = 0.0
            val_loss = None  
            for images, labels in tqdm(data_loader, leave = False):



                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = model(images)
                    loss = criterion(output, labels.float())
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss = epoch_loss / len(data_loader)

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")

            if epoch >= 10:
                val_loss = validate(model, val_loader, criterion, device)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here
                    patience = 10  # Reset patience counter
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stop at epoch:' + str(epoch))
                        early_stop = True
                        break
            metrics = {
                "epoch": epoch,
                "train_loss": loss,  
            }
            if val_loss is not None:
                metrics["val_loss"] = val_loss
            
            mlflow.log_metrics(metrics, step=epoch)
    if early_stop:
        return best_model_weights
    else:
        return model.state_dict()
    


def validate(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    print('Validating...')  

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating", leave=False):
            
            images = images.to(device)
            targets =  targets.to(device)
            
           
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            val_loss += loss.item()

    avg_val_loss = val_loss / len(data_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss


# Main execution
if __name__ == "__main__":

    mlflow.set_experiment("Fefficientnet_M")


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # Create datasets and instantiate dataloader
    batch_size = 128
    lr = 5e-4
    training_data = CustomDataset("imat_data/train_annos_relabel.json", "Train", transforms=get_transform())
    train_loader = DataLoader(training_data, batch_size= batch_size, shuffle=True, num_workers= 6)

    validation_data = CustomDataset("imat_data/val_annos_relabel.json" , "Val", transforms=get_transform())
    val_loader = DataLoader(validation_data, batch_size = batch_size,  num_workers = 6)

    num_classes = 131
    # Get model
    model = get_model(num_classes, path = 'weights/Fefficientnet.pth')



    # Create parameter groups
    backbone_params = [p for n, p in model.named_parameters() 
                      if p.requires_grad and "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters() 
                        if p.requires_grad and "classifier" in n]
    
    param_groups = [
        {'params': backbone_params, 'lr': lr * 0.1},  # Backbone with lower learning rate
        {'params': classifier_params, 'lr': lr}  # Classifier with higher learning rate
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=1e-2, lr = lr)
    criterion = nn.BCEWithLogitsLoss()



    torch.save(train_model(model, train_loader, val_loader, optimizer, device, criterion),'Fefficientnet_pt2.pth')
