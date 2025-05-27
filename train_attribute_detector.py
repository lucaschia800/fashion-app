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
    def __init__(self, data_path, transforms=None):
        data = json.load(open(data_path, 'r'))
        self.transforms = transforms
        self.image_paths = data['images']
        self.labels = data['annotations']

    def __getitem__(self, idx):

        image = self.image_paths[idx]['imageId']
        image = Image.open(image).convert("RGB")

        labels = [int(label) for label in self.labels[idx]['labelId']]
        labels = torch.tensor(labels, dtype=torch.long)
        labels = F.one_hot(labels, num_classes=131) 
        if self.transforms is not None:
            image = self.transforms(image)


        return image, labels

    def __len__(self):
        return len(self.image_paths)


def get_transform():
    weights = EfficientNet_V2_M_Weights.DEFAULT
    transforms = weights.transforms()
    return transforms



def get_model(num_classes = 131):
    # Load pre-trained model
    weights = EfficientNet_V2_M_Weights.DEFAULT  
    model = efficientnet_v2_m(weights=weights)  

    # Replace the classifier with a new one for transfer learning
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    # Unfreeze last few layers for fine-tuning
    for name, param in model.named_parameters():
        if "clasifier" in name:
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

    with mlflow.start_run():

        mlflow.log_params({
            "model_type": "efficientnet_v2_m",
            "batch_size": train_loader.batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "num_epochs": num_epochs,
            "optimizer": type(optimizer).__name__
        })

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for i, (images, labels) in enumerate(data_loader):



                images = list(image.to(device) for image in images)
                labels = [[label.to(device) for label in labels] for labels in labels]

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = model(images)
                    loss = criterion(output, labels.float())#Investigate whether I should be using the sum of all losses

                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            loss = epoch_loss / len(data_loader)

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")

            mlflow.log_metrics({
                "epoch": epoch,
                "train_loss": loss / len(train_loader),
                "val_loss": val_loss,
            }, step=epoch)

            if epoch >= 10:
                val_loss = validate(model, val_loader, device)

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

    if early_stop:
        return best_model_weights
    else:
        return model.state_dict()

def validate(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    print('Validating...')

    with torch.no_grad():
        for images, targets in data_loader:
            
            images = list(image.to(device) for image in images)
            targets = [[target.to(device) for target in targets] for targets in targets]
            
           
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
    training_data = CustomDataset("imat_data/", transforms=get_transform())
    train_loader = DataLoader(training_data, batch_size= batch_size, shuffle=True, num_workers= 6)

    validation_data = CustomDataset("imat_data/" , transforms=get_transform())
    val_loader = DataLoader(validation_data, batch_size = batch_size,  num_workers = 6)

    num_classes = 131
    # Get model
    model = get_model(num_classes)



    # Create parameter groups
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, betas=(0.9, 0.999), weight_decay=1e-2, lr = lr)
    criterion = nn.BCEWithLogitsLoss()



    torch.save(train_model(model, train_loader, val_loader, optimizer, device, criterion, batch_size),'Fefficientnet.pth')
