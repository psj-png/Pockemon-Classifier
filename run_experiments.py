"""
run_experiments.py — 4개 실험 전체 실행 스크립트

실험 축: 아키텍처(EfficientNet-B0 / ResNet-50) × TTA 유무
학습은 4개 모두 동일 조건 (동일 데이터 분할, 동일 하이퍼파라미터).
TTA 실험은 학습 완료 후 추론 단계에서만 TTA를 적용해 성능 차이를 측정.

실행:
    python run_experiments.py
"""
from __future__ import annotations

import json
import os

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler

from src.config import CKPT_DIR, EPOCHS, EXPERIMENTS, RESULT_DIR, SEED
from src.dataset import get_dataloaders
from src.evaluate import compute_metrics
from src.model import build_model
from src.trainer import evaluate_loader, train_one_epoch
from src.tta import predict_standard, predict_tta
from src.utils import fix_seed, save_curves, save_metrics, save_summary


def run_single(
    exp_name: str,
    cfg: dict,
    train_loader,
    val_loader,
    test_loader,
    test_paths: list[str],
    test_labels: list[int],
    class_names: list[str],
    device: torch.device,
) -> dict:
    print(f"\n{'='*65}")
    print(f"  {exp_name}")
    print(f"  {cfg['description']}")
    print(f"{'='*65}")

    model     = build_model(cfg["backbone"], cfg["pretrained"],
                             len(class_names)).to(device)
    optimizer = optim.AdamW(model.parameters(),
                             lr=cfg["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler(enabled=(device.type == "cuda"))
    ckpt_path = os.path.join(CKPT_DIR, f"{exp_name}.pth")

    history  = {k: [] for k in
                ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_acc = 0.0

    # ── 학습 루프 (TTA 유무 무관, 동일 조건)
    for epoch in range(1, EPOCHS + 1):
        tl, ta = train_one_epoch(model, train_loader, optimizer, scaler, device)
        vl, va = evaluate_loader(model, val_loader, device)
        scheduler.step()

        for k, v in zip(
            ["train_loss", "train_acc", "val_loss", "val_acc"],
            [tl, ta, vl, va],
        ):
            history[k].append(v)

        print(f"  [{epoch:02d}/{EPOCHS}]  "
              f"train  loss={tl:.4f}  acc={ta:.4f}  |  "
              f"val  loss={vl:.4f}  acc={va:.4f}")

        if va > best_acc:
            best_acc = va
            torch.save(
                {
                    "state_dict" : model.state_dict(),
                    "class_names": class_names,
                    "backbone"   : cfg["backbone"],
                    "use_tta"    : cfg["use_tta"],
                },
                ckpt_path,
            )

    # ── 테스트 추론 (TTA / 표준 분기)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if cfg["use_tta"]:
        print("  → TTA 추론 중...")
        preds, gts = predict_tta(model, test_paths, test_labels, device)
    else:
        print("  → 표준 추론 중...")
        preds, gts = predict_standard(model, test_loader, device)

    metrics = compute_metrics(preds, gts, class_names)
    save_curves(history, exp_name)
    save_metrics(metrics, exp_name)

    print(f"\n  [Test]  "
          f"Acc={metrics['accuracy']:.4f}  "
          f"P={metrics['precision']:.4f}  "
          f"R={metrics['recall']:.4f}  "
          f"F1={metrics['f1']:.4f}")
    return metrics


def main() -> None:
    fix_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    # TTA 추론에 필요한 test 경로/레이블 직접 접근
    test_ds     = test_loader.dataset
    test_paths  = test_ds.paths
    test_labels = test_ds.labels

    summary: dict[str, dict] = {}

    for name, cfg in EXPERIMENTS.items():
        m = run_single(
            name, cfg,
            train_loader, val_loader, test_loader,
            test_paths, test_labels,
            class_names, device,
        )
        summary[name] = {
            k: round(float(v), 4)
            for k, v in m.items()
            if k != "report"
        }

    save_summary(summary)

    # ── 최종 비교표 출력
    print("\n\n" + "=" * 65)
    print(f"{'실험':<44} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 65)
    for name, m in summary.items():
        tta_tag = " ← TTA" if EXPERIMENTS[name]["use_tta"] else ""
        print(
            f"{name:<44} "
            f"{m['accuracy']:>7.4f} "
            f"{m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} "
            f"{m['f1']:>7.4f}"
            f"{tta_tag}"
        )


if __name__ == "__main__":
    main()
