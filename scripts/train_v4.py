"""
YOLO11s-seg V4 파인튜닝 스크립트
=================================
목적: v3 best.pt에서 이어서 데이터 보강(incoming GT 667장 추가)된
      train set으로 추가 학습 후 v3와 성능 비교.
입력: /home1/Jwson08/autolabel/dataset/data.yaml (train 12644장, val/test는 v3와 동일)
출력: runs/segment/train_v4/weights/best.pt
"""

import sys
from pathlib import Path
from ultralytics import YOLO


# ===== [설정] =====
DATA_YAML  = Path("/home1/Jwson08/autolabel/dataset/data.yaml")

# v3 best.pt에서 시작
MODEL_NAME = "/home1/Jwson08/autolabel/runs/segment/runs/segment/train_v3/weights/best.pt"

EPOCHS     = 100
IMGSZ      = 640
BATCH      = 64    # v3(8) → v4(64): GPU 4장, GPU당 16장. v3에서 GPU당 4장에 3.65GB만 사용해 여유 충분
MASK_RATIO = 1     # v3와 동일 (풀 해상도 마스크)
DEVICE     = [0, 1, 2, 3]  # DDP 4 GPU
PROJECT    = "runs/segment"
NAME       = "train_v4"


# ===== [Step 1. 경로 확인] =====
print("=" * 55)
print("[Step 1] 경로 확인")

if not DATA_YAML.exists():
    print(f"  ❌ data.yaml 없음: {DATA_YAML}")
    sys.exit(1)
print(f"  ✅ data.yaml: {DATA_YAML}")

if not Path(MODEL_NAME).exists():
    print(f"  ❌ v3 모델 없음: {MODEL_NAME}")
    sys.exit(1)
print(f"  ✅ 시작 모델: {MODEL_NAME}")


# ===== [Step 2. 모델 로드] =====
print(f"\n[Step 2] v3 best.pt 로드")
model = YOLO(MODEL_NAME)
print(f"  task: {model.task}")


# ===== [Step 3. 학습] =====
print(f"\n[Step 3] V4 학습 시작")
print(f"  epochs={EPOCHS}, imgsz={IMGSZ}, batch={BATCH}, mask_ratio={MASK_RATIO}")
print(f"  device=cuda:{DEVICE}, GPU: {__import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
print(f"  결과 저장: {PROJECT}/{NAME}/\n")

results = model.train(
    data       = str(DATA_YAML),
    epochs     = EPOCHS,
    imgsz      = IMGSZ,
    batch      = BATCH,
    mask_ratio = MASK_RATIO,
    device     = DEVICE,
    project    = PROJECT,
    name       = NAME,
    exist_ok   = True,
    workers    = 4,
    verbose    = True,
)


# ===== [Step 4. 결과] =====
print("\n" + "=" * 55)
print("[Step 4] V4 학습 완료")

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
