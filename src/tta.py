"""
tta.py — Test-Time Augmentation 추론 모듈

TTA는 동일 이미지를 N개의 다른 transform으로 변환 후
각각의 softmax 확률을 평균(soft voting)해 최종 예측을 만든다.
학습 코드를 전혀 건드리지 않고 추론 단계에서만 적용된다.

세 레포(enderpawar / zihasoo / beeean17) 모두 단일 forward pass만
사용했으며, TTA를 실험 변수로 삼은 설계는 이 프로젝트에만 존재한다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PokemonDataset, get_tta_transforms


@torch.no_grad()
def predict_standard(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    일반 단일 forward pass 추론.
    반환: (pred_labels, true_labels)
    """
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, leave=False, desc="  infer(std)"):
        preds = model(imgs.to(device)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def predict_tta(
    model: nn.Module,
    paths: list[str],
    labels: list[int],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    TTA 추론.
    각 이미지를 N개 view로 변환 → softmax 평균 → argmax
    반환: (pred_labels, true_labels)
    """
    model.eval()
    tta_tfms  = get_tta_transforms()
    n_views   = len(tta_tfms)
    prob_sum: torch.Tensor | None = None

    for tfm in tqdm(tta_tfms, desc="  infer(TTA views)", leave=False):
        ds     = PokemonDataset(paths, labels, tfm)
        loader = DataLoader(ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)
        view_probs = []
        for imgs, _ in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1).cpu()
            view_probs.append(probs)

        cat = torch.cat(view_probs, dim=0)
        prob_sum = cat if prob_sum is None else prob_sum + cat

    avg_probs = prob_sum / n_views
    return avg_probs.argmax(1).numpy(), np.array(labels)


def predict_single_tta(
    model: nn.Module,
    img: Image.Image,
    device: torch.device,
) -> np.ndarray:
    """
    단일 PIL 이미지에 TTA 적용 → (n_classes,) 확률 배열 반환.
    Streamlit app.py에서 사용.
    """
    model.eval()
    tta_tfms = get_tta_transforms()
    prob_sum: torch.Tensor | None = None

    with torch.no_grad():
        for tfm in tta_tfms:
            tensor = tfm(img).unsqueeze(0).to(device)
            probs  = torch.softmax(model(tensor), dim=1).cpu().squeeze(0)
            prob_sum = probs if prob_sum is None else prob_sum + probs

    return (prob_sum / len(tta_tfms)).numpy()


def predict_single_standard(
    model: nn.Module,
    img: Image.Image,
    device: torch.device,
    eval_transform,
) -> np.ndarray:
    """
    단일 PIL 이미지 표준 추론 → (n_classes,) 확률 배열 반환.
    Streamlit app.py에서 사용.
    """
    model.eval()
    with torch.no_grad():
        tensor = eval_transform(img).unsqueeze(0).to(device)
        probs  = torch.softmax(model(tensor), dim=1).cpu().squeeze(0)
    return probs.numpy()
