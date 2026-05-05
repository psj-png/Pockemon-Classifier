"""
config.py — 경로, 하이퍼파라미터, 4개 실험 정의

실험 축: 아키텍처(EfficientNet-B0 vs ResNet-50) × TTA 적용 여부
  - 학습 조건은 4개 실험 모두 동일
  - TTA 실험(exp2, exp4)은 추론 단계에서만 차이 발생
"""
import os

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data", "PokemonData")
CKPT_DIR   = os.path.join(ROOT_DIR, "checkpoints")
RESULT_DIR = os.path.join(ROOT_DIR, "results")

for _d in [
    CKPT_DIR,
    os.path.join(RESULT_DIR, "curves"),
    os.path.join(RESULT_DIR, "metrics"),
]:
    os.makedirs(_d, exist_ok=True)

# ── 공통 하이퍼파라미터
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 15
SEED        = 42
NUM_WORKERS = 0   # Windows 안전값; Linux 환경에서는 4 권장

# ── TTA 설정
TTA_N_VIEWS = 6   # 원본 포함 총 몇 개 view를 평균낼지

# ── 4가지 실험 정의
EXPERIMENTS = {
    "exp1_effnet_b0_baseline": {
        "backbone"   : "efficientnet_b0",
        "pretrained" : True,
        "use_tta"    : False,
        "lr"         : 1e-4,
        "description": "EfficientNet-B0 | Pretrained | No TTA",
    },
    "exp2_effnet_b0_tta": {
        "backbone"   : "efficientnet_b0",
        "pretrained" : True,
        "use_tta"    : True,
        "lr"         : 1e-4,
        "description": "EfficientNet-B0 | Pretrained | TTA",
    },
    "exp3_resnet50_baseline": {
        "backbone"   : "resnet50",
        "pretrained" : True,
        "use_tta"    : False,
        "lr"         : 1e-4,
        "description": "ResNet-50 | Pretrained | No TTA",
    },
    "exp4_resnet50_tta": {
        "backbone"   : "resnet50",
        "pretrained" : True,
        "use_tta"    : True,
        "lr"         : 1e-4,
        "description": "ResNet-50 | Pretrained | TTA",
    },
}
