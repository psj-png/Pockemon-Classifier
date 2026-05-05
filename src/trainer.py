"""
trainer.py — 학습 루프

- AMP(Mixed Precision) 지원으로 GPU 메모리 절약
- Label Smoothing 0.05 적용
"""
import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[float, float]:
    """
    한 epoch 학습.
    반환: (avg_loss, accuracy)
    """
    model.train()
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.05)
    total_loss = correct = total = 0

    for imgs, labels in tqdm(loader, leave=False, desc="  train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type,
                            enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    검증/테스트 루프 (gradient 없음).
    반환: (avg_loss, accuracy)
    """
    model.eval()
    criterion  = nn.CrossEntropyLoss()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total
