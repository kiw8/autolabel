"""
YOLO 단독 진단 스크립트
========================
목적: YOLO bbox(탐지 정확도) vs mask(경계 정확도)를 분리해서 비교.
      bbox는 정확한데 mask만 나쁘면 → YOLO seg 마스크 한계
      bbox 자체가 나쁘면 → 모델 감지 성능 문제

출력:
  outputs/diagnose/01_bbox_only.jpg  ← bbox만 표시
  outputs/diagnose/02_mask_only.jpg  ← mask만 표시
  outputs/diagnose/03_combined.jpg   ← bbox + mask 합친 것
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ===== [설정] =====
MODEL_PATH  = "/home1/Jwson08/autolabel/runs/segment/runs/segment/train_v3/weights/best.pt"
IMAGE_PATH  = "/home1/Jwson08/autolabel/data/images/GX010097[1]_011_0000.jpg"
OUTPUT_DIR  = Path("/home1/Jwson08/autolabel/outputs/diagnose_v3")
CONF        = 0.25   # 낮게 설정해서 감지 가능한 것 최대한 확인

CLASS_NAMES  = {0: "person", 1: "dog"}
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}  # BGR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== [모델 로드 & 추론] =====
model   = YOLO(MODEL_PATH)
results = model.predict(source=IMAGE_PATH, conf=CONF, iou=0.5, device=0, verbose=False)
result  = results[0]

img_orig = cv2.imread(IMAGE_PATH)
H, W     = img_orig.shape[:2]

print(f"이미지 크기: {W}x{H}")
print(f"감지된 객체 수: {len(result.boxes)}")

# ===== [1. BBOX만] =====
img_bbox = img_orig.copy()
for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
    x1, y1, x2, y2 = map(int, box.tolist())
    class_id  = int(cls.item())
    confidence= float(conf.item())
    color     = CLASS_COLORS.get(class_id, (180,180,180))
    name      = CLASS_NAMES.get(class_id, "unknown")

    cv2.rectangle(img_bbox, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img_bbox, f"{name} {confidence:.2f}",
                (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    print(f"  bbox: {name} conf={confidence:.2f} [{x1},{y1},{x2},{y2}]")

cv2.imwrite(str(OUTPUT_DIR/"01_bbox_only.jpg"), img_bbox)
print(f"저장: {OUTPUT_DIR}/01_bbox_only.jpg")

# ===== [2. MASK만] =====
img_mask = img_orig.copy()
overlay  = img_orig.copy()

if result.masks is not None:
    for mask_xy, cls, conf in zip(result.masks.xy, result.boxes.cls, result.boxes.conf):
        class_id  = int(cls.item())
        confidence= float(conf.item())
        color     = CLASS_COLORS.get(class_id, (180,180,180))
        name      = CLASS_NAMES.get(class_id, "unknown")

        pts = mask_xy.astype(np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img_mask, [pts], True, color, 2)
        print(f"  mask: {name} conf={confidence:.2f} polygon={len(pts)}점")

    img_mask = cv2.addWeighted(overlay, 0.4, img_mask, 0.6, 0)
else:
    print("  마스크 없음")

cv2.imwrite(str(OUTPUT_DIR/"02_mask_only.jpg"), img_mask)
print(f"저장: {OUTPUT_DIR}/02_mask_only.jpg")

# ===== [3. BBOX + MASK 합치기] =====
img_combined = img_mask.copy()
for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
    x1, y1, x2, y2 = map(int, box.tolist())
    class_id  = int(cls.item())
    confidence= float(conf.item())
    color     = CLASS_COLORS.get(class_id, (180,180,180))
    name      = CLASS_NAMES.get(class_id, "unknown")

    cv2.rectangle(img_combined, (x1,y1), (x2,y2), color, 2)
    cv2.putText(img_combined, f"{name} {confidence:.2f}",
                (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite(str(OUTPUT_DIR/"03_combined.jpg"), img_combined)
print(f"저장: {OUTPUT_DIR}/03_combined.jpg")

# ===== [판단 기준 출력] =====
print("\n" + "="*50)
print("[판단 기준]")
print("01_bbox_only.jpg → bbox가 정확히 객체를 감싸는지 확인")
print("02_mask_only.jpg → mask 경계가 얼마나 정밀한지 확인")
print("03_combined.jpg  → bbox vs mask 차이 확인")
print()
print("bbox는 정확 + mask가 투박  → YOLO seg 마스크 해상도 문제 (V3로 해결 예정)")
print("bbox 자체가 부정확          → 모델 감지 성능 문제 (학습 데이터/모델 크기)")
print("="*50)
