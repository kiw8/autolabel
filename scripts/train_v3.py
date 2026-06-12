"""
YOLO11s-seg V3 파인튜닝 스크립트
=================================
목적: v2 best.pt에서 이어받아 고해상도 학습으로 mask 품질 개선.
      v2 대비 변경: imgsz 640→1280, mask_ratio 4→1 (풀 해상도 마스크)
입력: /home1/Jwson08/autolabel/dataset/data.yaml
출력: runs/segment/train_v3/weights/best.pt
"""

import sys
from pathlib import Path
from ultralytics import YOLO


# ===== [설정] =====
DATA_YAML  = Path("/home1/Jwson08/autolabel/dataset/data.yaml")

# v2 best.pt에서 시작
MODEL_NAME = "/home1/Jwson08/autolabel/runs/segment/runs/segment/train_v2/weights/best.pt"

EPOCHS     = 100
IMGSZ      = 640    # v2와 동일 해상도 유지 — 탐지 정확도 이미 충분 (conf 0.96/0.92)
BATCH      = 8      # imgsz=640으로 메모리 여유 생겨 4→8로 증가, 학습 속도 향상
MASK_RATIO = 1      # v2(4) → v3(1): 풀 해상도 마스크, 경계 부정확 문제 해결 ← 핵심 변경
DEVICE     = [0, 1] # DDP 2 GPU
PROJECT    = "runs/segment"
NAME       = "train_v3"


# ===== [Step 1. 경로 확인] =====
print("=" * 55)
print("[Step 1] 경로 확인")

if not DATA_YAML.exists():
    print(f"  ❌ data.yaml 없음: {DATA_YAML}")
    sys.exit(1)
print(f"  ✅ data.yaml: {DATA_YAML}")

if not Path(MODEL_NAME).exists():
    print(f"  ❌ v2 모델 없음: {MODEL_NAME}")
    sys.exit(1)
print(f"  ✅ 시작 모델: {MODEL_NAME}")


# ===== [Step 2. 모델 로드] =====
print(f"\n[Step 2] v2 best.pt 로드")
model = YOLO(MODEL_NAME)
print(f"  task: {model.task}")


# ===== [Step 3. 학습] =====
print(f"\n[Step 3] V3 학습 시작")
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
print("[Step 4] V3 학습 완료")

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
