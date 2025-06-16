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
import json
import copy
import typing
from torchvision.models import efficientnet_v2_m



class MultiHead_FEfficientNet(nn.Module):
    """
    EfficientNet with shared backbone and multiple classifier heads
    """
    
    def __init__(
        self,
        dropout_rate: float = 0.3,
        categories = {'gender' : 3, 'material' : 23, 'pattern' : 18, 'style' : 10, 'sleeve' : 4, 'category': 48, 'color' : 19},
        ckpt_path = None
    ):
        super().__init__()
    
        # Store configuration
        self.dropout_rate = dropout_rate
        self.categories = categories
        self.ckpt_path = ckpt_path
        

        weights = EfficientNet_V2_M_Weights.DEFAULT
        self.backbone = efficientnet_v2_m(weights=weights)  
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # Remove classifier layer
        print(f"Backbone feature dimension: {self.feature_dim}")
        
        self.classifier_heads = nn.ModuleDict()
        self._build_classifier_heads()
        
        self._initialize_heads()
    
    def _build_classifier_heads(self):
        """Build classifier heads for each attribute"""
        for category, num_classes in self.categories.items():
            head = self._create_classifier_head(
                num_classes=num_classes, #131 aint right for this remember that
            )
            self.classifier_heads[category] = head
    
    def _create_classifier_head(self, num_classes):
        """
        Create a classifier head with batch norm and dropout
        """
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, num_classes)


        )
    
    def _initialize_heads(self):
        """Initialize weights for classifier heads"""
        for head in self.classifier_heads.values():
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through shared backbone and all classifier heads
        
        Args:
            x: Input tensor of shape [batch_size, 3, height, width]
            
        Returns:
            Dictionary with predictions for each attribute
        """
        # Extract features from shared backbone
        features = self.backbone(x)  # Shape: [batch_size, feature_dim]
        
        # Pass features through each classifier head
        output_logits = {}
        for category in self.categories:
            logits = self.classifier_heads[category](features)
            output_logits[category] = logits
            
            
                
        return output_logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone without classification"""
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning heads only"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """
        Get separate parameter groups for backbone and heads
        Useful for different learning rates
        """
        backbone_params = list(self.backbone.parameters())
        head_params = []
        for head in self.classifier_heads.values():
            head_params.extend(list(head.parameters()))
        
        return backbone_params, head_params
    


class FashionMultiHeadLoss(nn.Module):
    def __init__(self, weights=None, categories = {'gender' : 3, 'material' : 23, 'pattern' : 18, 'style' : 10, 'sleeve' : 4, 'category': 48, 'color' : 19}):
        super().__init__()
        self.loss_function = nn.BCEWithLogitsLoss(reduction='mean') #default is mean
        self.categories = categories
        self.weights = weights if weights is not None else [1.0] * len(categories)
    
    def forward(self, predictions, targets):
        total_loss = 0.0
        loss_dict = {}
        
        for category in self.categories:
            
            # Calculate BCE loss
            loss = self.loss_function(
                predictions[category], 
                targets[category].float()
            )
            
            # Apply weight and add to total
            weighted_loss = loss * self.weights[list(self.categories.keys()).index(category)]
            total_loss += weighted_loss
            loss_dict[f'{category}_loss'] = loss.item()
        
        return total_loss, loss_dict