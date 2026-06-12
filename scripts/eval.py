"""
학습 결과 평가 스크립트
=======================
목적: 학습된 best.pt 모델의 성능을 두 가지 방법으로 평가한다.
     1) 정량 평가 — val 전체 데이터셋 mAP 측정
     2) 정성 평가 — val 이미지 샘플에 추론을 돌려 마스크 품질 시각화

입력: runs/segment/runs/segment/train_v1/weights/best.pt
출력: outputs/eval_metrics.txt  (mAP 수치)
      outputs/eval_samples/     (시각화 이미지 20장)
"""

# ===== [import] =====
import random
import shutil
from pathlib import Path

import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ===== [0. 디바이스 자동 선택] =====
device = 0 if torch.cuda.is_available() else "cpu"
print(f"[Step 0] 디바이스: {device}")


# ===== [1. 설정] =====
WEIGHTS     = "runs/segment/runs/segment/train_v4/weights/best.pt"
DATA_YAML   = "/home1/Jwson08/autolabel/dataset/data.yaml"
VAL_DIR     = Path("/home1/Jwson08/autolabel/dataset/images/val")
OUTPUT_DIR  = Path("/home1/Jwson08/autolabel/outputs/eval_samples_v4")
METRICS_TXT = Path("/home1/Jwson08/autolabel/outputs/eval_metrics_v4.txt")
NUM_SAMPLES = 20    # 시각화할 val 이미지 수
CONF_THRESH = 0.25

CLASS_COLORS = {
    0: (255, 80,  80),   # person — 빨강 계열
    1: (80,  200, 80),   # dog    — 초록 계열
}
CLASS_NAMES = {0: "person", 1: "dog"}


# ===== [2. 모델 로드] =====
print("\n[Step 1] 모델 로드...")
model = YOLO(WEIGHTS)
print(f"         weights: {WEIGHTS}")


# ===== [3. 정량 평가 — val 전체 mAP] =====
print("\n[Step 2] val 전체 mAP 측정 중... (시간이 걸릴 수 있습니다)")
metrics = model.val(
    data=DATA_YAML,
    split="val",
    device=device,
    verbose=False,
)

map50_b   = metrics.box.map50
map5095_b = metrics.box.map
map50_m   = metrics.seg.map50
map5095_m = metrics.seg.map

print("\n" + "=" * 50)
print("[정량 평가 결과]")
print(f"  Box  mAP@0.5      : {map50_b:.4f}")
print(f"  Box  mAP@0.5:0.95 : {map5095_b:.4f}")
print(f"  Mask mAP@0.5      : {map50_m:.4f}  ← 핵심 지표")
print(f"  Mask mAP@0.5:0.95 : {map5095_m:.4f}")
print("=" * 50)

# 수치 파일 저장
METRICS_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(METRICS_TXT, "w") as f:
    f.write(f"weights: {WEIGHTS}\n\n")
    f.write(f"Box  mAP@0.5      : {map50_b:.4f}\n")
    f.write(f"Box  mAP@0.5:0.95 : {map5095_b:.4f}\n")
    f.write(f"Mask mAP@0.5      : {map50_m:.4f}\n")
    f.write(f"Mask mAP@0.5:0.95 : {map5095_m:.4f}\n")
print(f"\n  수치 저장: {METRICS_TXT}")


# ===== [4. 정성 평가 — 샘플 이미지 시각화] =====
print(f"\n[Step 3] 샘플 {NUM_SAMPLES}장 시각화 중...")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

val_images = sorted(VAL_DIR.glob("*.jpg"))
samples = random.sample(val_images, min(NUM_SAMPLES, len(val_images)))

for i, img_path in enumerate(samples):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue
    H, W = img_bgr.shape[:2]
    overlay = img_bgr.copy()

    results = model.predict(
        source=str(img_path),
        conf=CONF_THRESH,
        device=device,
        verbose=False,
    )
    result = results[0]

    person_cnt = 0
    dog_cnt = 0

    if result.masks is not None:
        for mask_xy, cls, conf in zip(
            result.masks.xy,
            result.boxes.cls,
            result.boxes.conf,
        ):
            class_id = int(cls.item())
            confidence = float(conf.item())
            color = CLASS_COLORS.get(class_id, (180, 180, 180))
            name  = CLASS_NAMES.get(class_id, "unknown")

            if class_id == 0:
                person_cnt += 1
            elif class_id == 1:
                dog_cnt += 1

            # 마스크 채우기 (반투명)
            pts = mask_xy.astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)

            # 외곽선
            cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=2)

            # 레이블
            if len(pts) > 0:
                cx, cy = pts[0]
                cv2.putText(
                    img_bgr,
                    f"{name} {confidence:.2f}",
                    (cx, max(cy - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

    # 반투명 마스크 합성
    img_bgr = cv2.addWeighted(overlay, 0.35, img_bgr, 0.65, 0)

    # 파일명에 감지 수 표기
    tag = f"p{person_cnt}d{dog_cnt}"
    out_name = f"{i+1:02d}_{img_path.stem}_{tag}.jpg"
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), img_bgr)

    print(f"  [{i+1:2d}/{NUM_SAMPLES}] {img_path.name} → person {person_cnt}, dog {dog_cnt}")

print(f"\n  시각화 저장: {OUTPUT_DIR}/")


# ===== [5. 최종 요약] =====
print("\n" + "=" * 50)
print("[평가 완료]")
print(f"  Mask mAP@0.5      : {map50_m:.4f}")
print(f"  Mask mAP@0.5:0.95 : {map5095_m:.4f}")
print(f"  샘플 이미지       : {OUTPUT_DIR}/")
print(f"  수치 파일         : {METRICS_TXT}")
print("=" * 50)

if map50_m >= 0.85:
    print("\n  ✅ 성능 양호 — inference.py 작성 및 NAS 배포 진행 가능")
else:
    print(f"\n  ⚠️  Mask mAP@0.5 = {map50_m:.4f} — 샘플 이미지 확인 후 추가 학습 여부 판단")
