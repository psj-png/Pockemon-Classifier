"""
utils.py — 공통 유틸리티

- 시드 고정
- 학습 곡선 PNG 저장
- 실험 결과 JSON 저장
- summary.json 저장
"""
from __future__ import annotations

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import RESULT_DIR


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_curves(history: dict, exp_name: str) -> None:
    """
    history 키: train_loss, train_acc, val_loss, val_acc
    results/curves/{exp_name}.png 으로 저장.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(exp_name, fontsize=11)

    for ax, key, title in [
        (axes[0], "loss", "Loss"),
        (axes[1], "acc",  "Accuracy"),
    ]:
        ax.plot(history[f"train_{key}"], label="Train", linewidth=1.8)
        ax.plot(history[f"val_{key}"],   label="Val",   linewidth=1.8,
                linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "curves", f"{exp_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ curves  → {path}")


def save_metrics(metrics: dict, exp_name: str) -> None:
    """results/metrics/{exp_name}.json 으로 저장 (report 문자열 제외)."""
    path    = os.path.join(RESULT_DIR, "metrics", f"{exp_name}.json")
    payload = {k: v for k, v in metrics.items() if k != "report"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  ✓ metrics → {path}")


def save_summary(summary: dict) -> None:
    """results/summary.json 으로 저장."""
    path = os.path.join(RESULT_DIR, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ summary → {path}")
