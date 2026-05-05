"""
gradcam.py — GradCAM 시각화

모델이 예측 시 이미지의 어느 픽셀 영역에 집중했는지 히트맵으로 추출.
세 참고 레포 모두 구현하지 않은 기능.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model  = model
        self._feat: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None
        target_layer.register_forward_hook(self._hook_feat)
        target_layer.register_full_backward_hook(self._hook_grad)

    def _hook_feat(self, _, __, out):
        self._feat = out.detach()

    def _hook_grad(self, _, __, grad_out):
        self._grad = grad_out[0].detach()

    def __call__(
        self,
        tensor: torch.Tensor,          # (1, 3, H, W) normalized
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        반환: (cam_array H×W in [0,1], predicted_class_idx)
        """
        self.model.eval()
        tensor = tensor.requires_grad_(True)
        logits = self.model(tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # GAP weights × feature maps
        weights = self._grad.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self._feat).sum(dim=1).squeeze())
        cam     = cam.cpu().numpy()
        cam    -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, class_idx


def overlay_heatmap(
    original: Image.Image,
    cam: np.ndarray,
    size: int = 224,
) -> np.ndarray:
    """원본 이미지 위에 GradCAM 히트맵 오버레이 → np.ndarray (RGB)."""
    img     = np.array(original.resize((size, size))).astype(np.float32)
    heatmap = cv2.resize(cam, (size, size))
    heatmap = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = 0.55 * img + 0.45 * heatmap
    return np.clip(overlay, 0, 255).astype(np.uint8)


def get_target_layer(model: nn.Module, backbone: str) -> nn.Module:
    """백본별 마지막 conv layer 반환."""
    if backbone == "efficientnet_b0":
        return model.features[-1]
    if backbone == "resnet50":
        return model.layer4[-1]
    raise ValueError(f"Unknown backbone: {backbone!r}")
