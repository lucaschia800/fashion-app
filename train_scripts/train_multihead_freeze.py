import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2  
import torchvision.datasets as datasets
from torchvision.models import EfficientNet_V2_M_Weights
import os
from tqdm import tqdm 
from PIL import Image
import numpy as np
import mlflow
import mlflow.pytorch
import json
import copy
import classes.multiheaded_Fefficient as custom_model
import classes.Multiheaded_dataset as custom_dataset




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
            for images, label_map in tqdm(data_loader, leave = False):



                images = images.to(device)
                labels = {k: v.to(device) for k, v in label_map.items()}

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = model(images)
                    loss, loss_dict = criterion(output, labels)
                
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
        for images, target_map in tqdm(data_loader, desc="Validating", leave=False):
            
            images = images.to(device)
            targets = {k: v.to(device) for k, v in target_map.items()}
            
           
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, targets)
            
            val_loss += loss.item()

    avg_val_loss = val_loss / len(data_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss


# Main execution
if __name__ == "__main__":

    mlflow.set_experiment("Fefficientnet_M")


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # Create datasets and instantiate dataloader
    batch_size = 75
    lr = 5e-4
    transforms = EfficientNet_V2_M_Weights.DEFAULT.transforms()
    training_data = custom_dataset.CustomDataset("imat_data/train_annos_group_relabeled.json", "Train", transforms=transforms)
    train_loader = DataLoader(training_data, batch_size= batch_size, shuffle=True, num_workers= 6)

    validation_data = custom_dataset.CustomDataset("imat_data/val_annos_group_relabeled.json" , "Val", transforms=transforms)
    val_loader = DataLoader(validation_data, batch_size = batch_size,  num_workers = 6)


    # Get model
    model = custom_model.MultiHead_FEfficientNet()
    model.freeze_backbone()



    # Create parameter groups
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, betas=(0.9, 0.999), weight_decay=1e-2, lr = lr)
    criterion = custom_model.FashionMultiHeadLoss()



    torch.save(train_model(model, train_loader, val_loader, optimizer, device, criterion),'Mefficientnet_freeze.pth')
