"""
dataset.py — 데이터 로딩 모듈

- sklearn stratified split으로 클래스 비율 보존 (70 / 15 / 15)
- 학습 / 검증 / 테스트 transform 분리
- TTA용 다중 view transform 제공
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import (
    BATCH_SIZE, DATA_DIR, IMG_SIZE, NUM_WORKERS, SEED, TTA_N_VIEWS
)

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)
_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── 이미지 경로 수집 ──────────────────────────────────────────
def _collect(root: str) -> tuple[list[str], list[int], list[str]]:
    root_p  = Path(root)
    classes = sorted([d.name for d in root_p.iterdir() if d.is_dir()])
    c2i     = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for cls in classes:
        for p in (root_p / cls).iterdir():
            if p.suffix.lower() in _EXTS:
                paths.append(str(p))
                labels.append(c2i[cls])
    return paths, labels, classes


# ── Transform 정의 ────────────────────────────────────────────
def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def _eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """app.py 에서 import해서 쓸 수 있도록 외부 공개."""
    return _eval_transform()


def get_tta_transforms() -> list[transforms.Compose]:
    """
    TTA용 transform 목록 (TTA_N_VIEWS개).

    원본 포함 6종의 약한 변환만 사용.
    학습 augmentation과 달리 테스트 이미지의 자연스러운
    변형 범위만 커버해 노이즈를 최소화한다.
    """
    views = [
        # 0: 원본 center crop
        transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
        ]),
        # 1: 좌우 반전
        transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
        ]),
        # 2: 살짝 밝게
        transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
        ]),
        # 3: 살짝 어둡게
        transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
            transforms.ColorJitter(brightness=-0.2),
            transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
        ]),
        # 4: 상단 편향 crop
        transforms.Compose([
            transforms.Resize(256),
            transforms.Lambda(lambda img: img.crop(
                (16, 0, img.width - 16, img.height - 32))),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
        ]),
        # 5: 하단 편향 crop
        transforms.Compose([
            transforms.Resize(256),
            transforms.Lambda(lambda img: img.crop(
                (16, 32, img.width - 16, img.height))),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
        ]),
    ]
    return views[:TTA_N_VIEWS]


# ── Dataset ───────────────────────────────────────────────────
class PokemonDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# ── DataLoader 생성 ───────────────────────────────────────────
def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    stratified 70 / 15 / 15 분할.
    반환: (train_loader, val_loader, test_loader, class_names)
    """
    paths, labels, classes = _collect(DATA_DIR)

    p_tv, p_test, y_tv, y_test = train_test_split(
        paths, labels, test_size=0.15,
        stratify=labels, random_state=SEED,
    )
    p_train, p_val, y_train, y_val = train_test_split(
        p_tv, y_tv,
        test_size=0.15 / 0.85,   # 전체 대비 약 15%
        stratify=y_tv, random_state=SEED,
    )

    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    train_loader = DataLoader(
        PokemonDataset(p_train, y_train, _train_transform()),
        shuffle=True, **kw,
    )
    val_loader = DataLoader(
        PokemonDataset(p_val, y_val, _eval_transform()),
        shuffle=False, **kw,
    )
    test_loader = DataLoader(
        PokemonDataset(p_test, y_test, _eval_transform()),
        shuffle=False, **kw,
    )

    print(f"[Dataset]  train={len(p_train)}  val={len(p_val)}  "
          f"test={len(p_test)}  classes={len(classes)}")
    return train_loader, val_loader, test_loader, classes
