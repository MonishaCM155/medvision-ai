"""
PyTorch DenseNet-121 Architecture for Multi-Label Chest X-Ray Classification
"""

import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet121ChestXray(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(DenseNet121ChestXray, self).__init__()
        # Load DenseNet-121 backbone
        self.densenet = models.densenet121(pretrained=pretrained)
        num_features = self.densenet.classifier.in_features
        
        # Replace classifier with custom multi-label head
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.densenet(x)

    def get_gradcam_target_layer(self):
        """Returns target convolution layer for Grad-CAM extraction."""
        return self.densenet.features.denseblock4.denselayer16.conv2
