"""
PyTorch DenseNet-121 Architecture for Multi-Label Chest X-Ray Classification

NOTE on checkpoint compatibility: the inference engine
(backend/app/main.py) detects a fine-tuned checkpoint via the
'classifier.weight' key and swaps the head for a plain Linear layer of the
matching width. The head here is therefore a single Linear layer — keep it
that way so trained checkpoints load into the engine unchanged.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DenseNet121ChestXray(nn.Module):
    """DenseNet-121 with a plain Linear multi-label sigmoid head."""

    def __init__(self, num_classes=10, pretrained=True):
        super(DenseNet121ChestXray, self).__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet = models.densenet121(weights=weights)
        num_features = self.densenet.classifier.in_features

        # Plain Linear head — state_dict keys are `classifier.weight/bias`,
        # which is exactly what the engine's checkpoint loader expects.
        self.densenet.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.densenet(x)

    def get_gradcam_target_layer(self):
        """Returns target convolution layer for Grad-CAM extraction."""
        return self.densenet.features.denseblock4.denselayer16.conv2
