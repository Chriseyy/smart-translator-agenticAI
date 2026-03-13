import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 16) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
