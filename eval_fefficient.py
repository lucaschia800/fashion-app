from torchmetrics.classification import MultilabelAveragePrecision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2  
import torchvision.datasets as datasets
from torchvision.models import efficientnet_v2_m
from torchvision.models import EfficientNet_V2_M_Weights
import torch
from tqdm import tqdm 
from PIL import Image
import json
import torch.nn as nn


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

def eval_fefficient(model, dataloader, device):
    model.eval()
    model.to(device)
    map_metric = MultilabelAveragePrecision(num_labels=131, average=None).to(device)  # Per-class AP
    map_macro = MultilabelAveragePrecision(num_labels=131, average='macro').to(device)

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            outputs = torch.sigmoid(outputs)

        # Calculate metrics
        map_metric.update(outputs, labels)
        map_macro.update(outputs, labels)

    per_class_ap = map_metric.compute()
    macro_ap = map_macro.compute()

    return per_class_ap, macro_ap


def save_metrics(per_class_ap, mean_ap, save_path):
    """Save metrics to JSON file"""
    metrics_dict = {
        "mean_average_precision": float(mean_ap),
        "per_class_average_precision": per_class_ap.cpu().tolist(),
        "num_classes": len(per_class_ap)
    }
    
    # Add some summary statistics
    metrics_dict["ap_statistics"] = {
        "min_ap": float(per_class_ap.min()),
        "max_ap": float(per_class_ap.max()),
        "std_ap": float(per_class_ap.std()),
        "median_ap": float(per_class_ap.median())
    }
    
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Metrics saved to {save_path}")
    return metrics_dict

batch_size = 128
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

validation_data = CustomDataset("imat_data/val_annos_relabel.json" , "Val", transforms=get_transform())
val_loader = DataLoader(validation_data, batch_size = batch_size,  num_workers = 6)

per_class_ap, macro_ap = eval_fefficient(get_model(path = "weights/Fefficientnet_pt2.pth"), val_loader, device)

save_metrics(per_class_ap, macro_ap, save_path="eval_res/validation_metrics.json")

