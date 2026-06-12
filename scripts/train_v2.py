"""
YOLO11s-seg V2 파인튜닝 스크립트
=================================
목적: v1 best.pt에서 이어받아 dog 성능 개선에 집중한 재학습.
      v1 대비 변경: 시작 가중치를 v1 best.pt로 교체, epochs 150으로 증가.
입력: /home1/Jwson08/autolabel/dataset/data.yaml
출력: runs/segment/train_v2/weights/best.pt
"""

import sys
from pathlib import Path
from ultralytics import YOLO


# ===== [설정] =====
DATA_YAML  = Path("/home1/Jwson08/autolabel/dataset/data.yaml")

# v1 best.pt에서 시작 — COCO pretrained 대비 이미 person/dog 특징 학습된 상태
MODEL_NAME = "/home1/Jwson08/autolabel/models/yolo11s_seg_v1_20260601/best.pt"

EPOCHS  = 150   # v1(100) 대비 50 epoch 추가 — dog mAP 추가 개선 여유
IMGSZ   = 640
BATCH   = 32
DEVICE  = [0, 1]
PROJECT = "runs/segment"
NAME    = "train_v2"


# ===== [Step 1. 경로 확인] =====
print("=" * 55)
print("[Step 1] 경로 확인")

if not DATA_YAML.exists():
    print(f"  ❌ data.yaml 없음: {DATA_YAML}")
    sys.exit(1)
print(f"  ✅ data.yaml: {DATA_YAML}")

if not Path(MODEL_NAME).exists():
    print(f"  ❌ v1 모델 없음: {MODEL_NAME}")
    sys.exit(1)
print(f"  ✅ 시작 모델: {MODEL_NAME}")


# ===== [Step 2. 모델 로드] =====
print(f"\n[Step 2] v1 best.pt 로드")
model = YOLO(MODEL_NAME)
print(f"  task: {model.task}")


# ===== [Step 3. 학습] =====
print(f"\n[Step 3] V2 학습 시작")
print(f"  epochs={EPOCHS}, imgsz={IMGSZ}, batch={BATCH}, device=cuda:{DEVICE}")
print(f"  결과 저장: {PROJECT}/{NAME}/\n")

results = model.train(
    data     = str(DATA_YAML),
    epochs   = EPOCHS,
    imgsz    = IMGSZ,
    batch    = BATCH,
    device   = DEVICE,
    project  = PROJECT,
    name     = NAME,
    exist_ok = True,
    workers  = 4,
    verbose  = True,
)


# ===== [Step 4. 결과] =====
print("\n" + "=" * 55)
print("[Step 4] V2 학습 완료")

best_pt = Path(PROJECT) / NAME / "weights" / "best.pt"
if best_pt.exists():
    print(f"  best.pt: {best_pt.resolve()}")
else:
    print(f"  ⚠️  best.pt 없음: {best_pt}")

metrics  = results.results_dict
map50    = metrics.get("metrics/mAP50(M)",    "N/A")
map5095  = metrics.get("metrics/mAP50-95(M)", "N/A")

print(f"  mAP50   (mask): {map50:.4f}"   if isinstance(map50,   float) else f"  mAP50  : {map50}")
print(f"  mAP50-95(mask): {map5095:.4f}" if isinstance(map5095, float) else f"  mAP50-95: {map5095}")
print("=" * 55)
