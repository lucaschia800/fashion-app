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
import typing


class MultiHead_FEfficientNet(nn.Module):
    """
    EfficientNet with shared backbone and multiple classifier heads
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b3',
        attribute_configs: List[AttributeConfig] = None,
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()
    
        # Store configuration
        self.attribute_configs = attribute_configs or []
        self.dropout_rate = dropout_rate
        
        # Load pretrained EfficientNet backbon
        weights = EfficientNet_V2_M_Weights.DEFAULT
        self.backbone = efficientnet_v2_m(weights=weights)  
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.classifier[1].in_features
        print(f"Backbone feature dimension: {self.feature_dim}")
        
        # Create classifier heads
        self.classifier_heads = nn.ModuleDict()
        self._build_classifier_heads()
        
        # Initialize weights for new heads
        self._initialize_heads()
    
    def _build_classifier_heads(self):
        """Build classifier heads for each attribute"""
        for config in self.attribute_configs:
            head = self._create_classifier_head(
                num_classes=len(config.classes),
                head_name=config.name
            )
            self.classifier_heads[config.name] = head
    
    def _create_classifier_head(self, num_classes: int, head_name: str) -> nn.Module:
        """
        Create a classifier head with batch norm and dropout
        """
        return nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, num_classes),
            nn.SiLU(inplace=True),

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
        outputs = {}
        for config in self.attribute_configs:
            logits = self.classifier_heads[config.name](features)
            
            # Apply appropriate activation based on attribute type
            if config.type == 'multiclass':
                outputs[config.name] = F.softmax(logits, dim=1)
            elif config.type == 'multilabel':
                outputs[config.name] = torch.sigmoid(logits)
            else:
                # Return raw logits for custom processing
                outputs[config.name] = logits
                
        return outputs
    
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
    
    def get_trainable_parameters(self) -> Tuple[List, List]:
        """
        Get separate parameter groups for backbone and heads
        Useful for different learning rates
        """
        backbone_params = list(self.backbone.parameters())
        head_params = []
        for head in self.classifier_heads.values():
            head_params.extend(list(head.parameters()))
        
        return backbone_params, head_params