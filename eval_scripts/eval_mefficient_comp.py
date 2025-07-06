from torchmetrics.classification import (
    MultilabelAveragePrecision, MultilabelF1Score, MultilabelAccuracy,
    MulticlassAccuracy, MulticlassF1Score
)
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
import classes.multiheaded_fefficient_v2 as custom_model
import classes.Multiheaded_dataset as custom_dataset

def get_model(path = None):
    model = custom_model.MultiHead_FEfficientNet(ckpt_path=path)
    if path is None:
        model = custom_model.MultiHead_FEfficientNet()
        
    return model

def precision_at_k(predictions, targets, k=3):
    """
    Calculate precision@k for multilabel classification
    Args:
        predictions: (batch_size, num_labels) probability scores
        targets: (batch_size, num_labels) binary targets
        k: number of top predictions to consider
    Returns:
        precision@k score
    """
    batch_size = predictions.size(0)
    
    # Get top k predictions for each sample
    _, top_k_indices = torch.topk(predictions, k, dim=1)
    
    # Create binary matrix for top-k predictions
    top_k_binary = torch.zeros_like(predictions)
    top_k_binary.scatter_(1, top_k_indices, 1)
    
    # Calculate precision@k: fraction of top-k predictions that are correct
    precision_at_k = (top_k_binary * targets).sum(dim=1) / k
    
    return precision_at_k.mean()

def eval_fefficient(model, dataloader, device, categories, metric_dict):
    model.eval()
    model.to(device)
    
    # Define category types
    multiclass_categories = ['gender', 'style', 'sleeve']
    multilabel_categories = ['color', 'pattern', 'material', 'category']
    
    # Storage for custom metrics
    precision_at_3_scores = {cat: [] for cat in multilabel_categories}

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        with torch.no_grad():
            output_logits = model(images)

        # Calculate metrics
        for category in categories:
            if category in multiclass_categories:
                # Multiclass processing
                probabilities = torch.softmax(output_logits[category], dim=1)
                true_labels = labels[category].argmax(dim=1)  # Convert one-hot to class indices
                
                # Update multiclass metrics
                for metric_name, metric in metric_dict[category].items():
                    metric.update(probabilities, true_labels)
                    
            else:
                # Multilabel processing
                probabilities = torch.sigmoid(output_logits[category])
                
                # Update multilabel metrics
                for metric_name, metric in metric_dict[category].items():
                    metric.update(probabilities, labels[category])
                
                # Calculate precision@3
                prec_at_3 = precision_at_k(probabilities, labels[category], k=3)
                precision_at_3_scores[category].append(prec_at_3)
        
    final_results = {}
    print("\nComputing final metrics...")
    
    for category in categories:
        final_results[category] = {}
        
        # Compute standard metrics
        for metric_name, metric in metric_dict[category].items():
            result = metric.compute()
            if metric_name.endswith('_per_class'):
                final_results[category][metric_name] = result.cpu().tolist()
            else:
                final_results[category][metric_name] = result.cpu().item()
        
        # Add precision@3 for multilabel categories
        if category in multilabel_categories and precision_at_3_scores[category]:
            final_results[category]['precision_at_3'] = torch.stack(precision_at_3_scores[category]).mean().item()

    return final_results

def save_metrics(results_dict, save_path):
    """
    Save the dictionary of computed metrics to a JSON file.
    
    Args:
        results_dict (dict): A dictionary containing the final computed metrics.
        save_path (str): The path to the file where metrics should be saved.
    """
    print(f"\nSaving metrics to {save_path}...")
    try:
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print("Metrics saved successfully.")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    return results_dict

if __name__ == "__main__":
    batch_size = 75
    categories = {'gender' : 3, 'material' : 23, 'pattern' : 18, 'style' : 10, 'sleeve' : 4, 'category': 48, 'color' : 19}
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define category types
    multiclass_categories = ['gender', 'style', 'sleeve']
    multilabel_categories = ['color', 'pattern', 'material', 'category']

    validation_data = custom_dataset.CustomDataset("imat_data/val_annos_group_relabeled.json" , "Val", transforms=EfficientNet_V2_M_Weights.DEFAULT.transforms(), train=False)
    val_loader = DataLoader(validation_data, batch_size = batch_size,  num_workers = 6)
    
    metric_dict = {}

    for category, num_classes in categories.items():
        metric_dict[category] = {}
        
        if category in multiclass_categories:
            # Multiclass metrics: accuracy, top3_accuracy, f1_macro
            metric_dict[category]['accuracy'] = MulticlassAccuracy(num_classes=num_classes).to(device)
            metric_dict[category]['top3_accuracy'] = MulticlassAccuracy(num_classes=num_classes, top_k=3).to(device)
            metric_dict[category]['f1_macro'] = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
            
        else:
            # Multilabel metrics: average_precision, f1_macro, subset_accuracy
            metric_dict[category]['average_precision_per_class'] = MultilabelAveragePrecision(num_labels=num_classes, average=None).to(device)
            metric_dict[category]['average_precision_macro'] = MultilabelAveragePrecision(num_labels=num_classes, average='macro').to(device)
            metric_dict[category]['f1_macro'] = MultilabelF1Score(num_labels=num_classes, average='macro').to(device)
            metric_dict[category]['subset_accuracy'] = MultilabelAccuracy(num_labels=num_classes, average='micro').to(device)

    final_results = eval_fefficient(get_model(path = "weights/Mefficientnet_v2_freeze_ckpt2_large_data.pth"), val_loader, device, categories, metric_dict)

    save_metrics(final_results, save_path="eval_res/validation_metrics_comprehensive_large_data.json")

