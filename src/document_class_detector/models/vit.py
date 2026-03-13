import torch.nn as nn
import timm


def build_model(num_classes: int = 16) -> nn.Module:
    model = timm.create_model("vit_base_patch16_224", pretrained=True)

    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)

    return model
