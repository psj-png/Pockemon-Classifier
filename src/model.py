"""
model.py — 백본 팩토리

지원 백본: efficientnet_b0 / resnet50
"""
import torch.nn as nn
import torchvision.models as models


def build_model(backbone: str, pretrained: bool, num_classes: int) -> nn.Module:
    """
    backbone   : 'efficientnet_b0' | 'resnet50'
    pretrained : True  → ImageNet1K 가중치 사용
                 False → random init (inference 전용)
    num_classes: 분류할 클래스 수 (보통 150)
    """
    weights_map = {
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "resnet50"        : models.ResNet50_Weights.IMAGENET1K_V1,
    }
    w = weights_map.get(backbone) if pretrained else None

    if backbone == "efficientnet_b0":
        m    = models.efficientnet_b0(weights=w)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_f, num_classes),
        )

    elif backbone == "resnet50":
        m    = models.resnet50(weights=w)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_f, num_classes),
        )

    else:
        raise ValueError(f"지원하지 않는 backbone: {backbone!r}")

    return m
