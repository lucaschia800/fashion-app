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
import classes.multiheaded_Fefficient as custom_model
import classes.Multiheaded_dataset as custom_dataset

def get_model(path = None):
    model = custom_model.MultiHead_FEfficientNet()
    if path is not None:
        model.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"Model loaded from {path}")\
        
    return model


def eval_fefficient(model, dataloader, device, categories, metric_dict):
    model.eval()
    model.to(device)


    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        with torch.no_grad():
            output_logits = model(images)


        # Calculate metrics
        for category in categories:
            map_metric, map_macro = metric_dict[category]
            # Convert outputs and labels to binary format

            probabilities = torch.sigmoid(output_logits[category])
            

            map_metric.update(probabilities, labels[category])
            map_macro.update(probabilities, labels[category])
        
    final_results = {}
    print("\nComputing final metrics...")
    for category, (map_metric, map_macro) in metric_dict.items():
        final_results[f"{category}_per_class_ap"] = map_metric.compute().cpu().tolist()
        final_results[f"{category}_macro_ap"] = map_macro.compute().cpu().item()


    return final_results

def save_metrics(results_dict, save_path):
    """
    Save the dictionary of computed metrics to a JSON file.
    
    Args:
        results_dict (dict): A dictionary containing the final computed metrics.
        save_path (str): The path to the file where metrics should be saved.
    """
    # CORRECT: This function is now designed to handle the dictionary of results.
    print(f"\nSaving metrics to {save_path}...")
    try:
        with open(save_path, 'w') as f:
            # The indent=2 argument makes the JSON file human-readable
            json.dump(results_dict, f, indent=2)
        print("Metrics saved successfully.")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    return results_dict

if __name__ == "__main__":
    batch_size = 75
    categories = {'gender' : 3, 'material' : 23, 'pattern' : 18, 'style' : 10, 'sleeve' : 4, 'category': 48, 'color' : 19}
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    validation_data = custom_dataset.CustomDataset("imat_data/val_annos_group_relabled.json" , "Val", transforms=EfficientNet_V2_M_Weights.DEFAULT.transforms())
    val_loader = DataLoader(validation_data, batch_size = batch_size,  num_workers = 6)
    
    metric_dict = {}

    for category, classes in categories.items():
        map_metric = MultilabelAveragePrecision(num_labels=classes, average=None).to(device)  # Per-class AP
        map_macro = MultilabelAveragePrecision(num_labels=classes, average='macro').to(device)

        metric_dict[category] = (map_metric, map_macro)


    final_results = eval_fefficient(get_model(path = "weights/Mefficientnet.pth"), val_loader, device, categories, metric_dict) #make sure this path is correct

    save_metrics(final_results, save_path="eval_res/validation_metrics_mefficient_no_freeze.json")

