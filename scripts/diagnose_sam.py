"""
SAM 단독 진단 스크립트 (Step 3)
================================
목적: SAM3를 단독으로 실행해 polygon 품질을 YOLO(Step 2) 결과와 비교.
      클릭 기반 vs bbox 기반 두 가지 방식 모두 테스트.

Step 2 결과 (YOLO V2, 동일 이미지):
  person: conf=0.96, polygon=300점, bbox=[952,412,1769,860]
  dog   : conf=0.92, polygon=74점,  bbox=[934,363,1231,540]

출력:
  outputs/diagnose/sam_01_click.jpg   ← 클릭 기반 SAM
  outputs/diagnose/sam_02_bbox.jpg    ← bbox 기반 SAM
  outputs/diagnose/sam_03_compare.jpg ← YOLO mask vs SAM 나란히 비교
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import SAM

# ===== [설정] =====
SAM_PATH   = "/nas03/models/labeling-project/sam3.pt"
IMAGE_PATH = "/home1/Jwson08/autolabel/data/images/GX010097[1]_011_0000.jpg"
OUTPUT_DIR = Path("/home1/Jwson08/autolabel/outputs/diagnose")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 2 YOLO 결과에서 가져온 bbox 좌표
YOLO_BBOXES = [
    {"name": "person", "class_id": 0, "bbox": [952, 412, 1769, 860], "color": (0, 0, 255)},
    {"name": "dog",    "class_id": 1, "bbox": [934, 363, 1231, 540], "color": (0, 255, 0)},
]

# 클릭 좌표: bbox 중심점, label=1(foreground)
def bbox_center(bbox):
    return [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

_DP_EPSILON = 0.3  # sam_inference.py와 동일 설정

def mask_to_polygon(mask_data):
    mask_np = mask_data.cpu().numpy()
    mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = _DP_EPSILON * cv2.arcLength(largest, closed=True) / 100
    approx  = cv2.approxPolyDP(largest, epsilon, closed=True)
    if len(approx) < 3:
        return None
    return approx

def draw_polygon(img, approx, color, label):
    overlay = img.copy()
    cv2.fillPoly(overlay, [approx], color)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.polylines(img, [approx], True, color, 2)
    cx, cy = approx[0][0]
    cv2.putText(img, label, (cx, max(cy - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img

# ===== [모델 로드] =====
print("SAM3 로드 중...")
model = SAM(SAM_PATH)
img_orig = cv2.imread(IMAGE_PATH)
H, W = img_orig.shape[:2]
print(f"이미지 크기: {W}x{H}")

# ===== [1. 클릭 기반 SAM] =====
print("\n[1] 클릭 기반 SAM 추론")
img_click = img_orig.copy()
click_results = []

for obj in YOLO_BBOXES:
    cx, cy = bbox_center(obj["bbox"])
    pts    = [[cx, cy]]
    labels = [1]

    results = model.predict(
        source=IMAGE_PATH,
        points=pts,
        labels=labels,
        device=0,
        verbose=False,
    )
    r = results[0]
    if r.masks is None or len(r.masks) == 0:
        print(f"  {obj['name']}: 마스크 없음")
        click_results.append(None)
        continue

    scores   = r.boxes.conf if r.boxes is not None and len(r.boxes) > 0 else None
    best_idx = int(scores.argmax()) if scores is not None and len(scores) > 0 else 0
    approx   = mask_to_polygon(r.masks.data[best_idx])

    if approx is None:
        print(f"  {obj['name']}: polygon 변환 실패")
        click_results.append(None)
        continue

    n_pts = len(approx)
    print(f"  {obj['name']}: click=({cx},{cy}), polygon={n_pts}점")
    click_results.append({"approx": approx, "n_pts": n_pts})

    label_str = f"{obj['name']} {n_pts}pts"
    img_click = draw_polygon(img_click, approx, obj["color"], label_str)

cv2.imwrite(str(OUTPUT_DIR / "sam_01_click.jpg"), img_click)
print(f"저장: {OUTPUT_DIR}/sam_01_click.jpg")

# ===== [2. bbox 기반 SAM] =====
print("\n[2] bbox 기반 SAM 추론")
img_bbox = img_orig.copy()
bbox_results = []

for obj in YOLO_BBOXES:
    results = model.predict(
        source=IMAGE_PATH,
        bboxes=[obj["bbox"]],
        device=0,
        verbose=False,
    )
    r = results[0]
    if r.masks is None or len(r.masks) == 0:
        print(f"  {obj['name']}: 마스크 없음")
        bbox_results.append(None)
        continue

    scores   = r.boxes.conf if r.boxes is not None and len(r.boxes) > 0 else None
    best_idx = int(scores.argmax()) if scores is not None and len(scores) > 0 else 0
    approx   = mask_to_polygon(r.masks.data[best_idx])

    if approx is None:
        print(f"  {obj['name']}: polygon 변환 실패")
        bbox_results.append(None)
        continue

    n_pts = len(approx)
    print(f"  {obj['name']}: bbox={obj['bbox']}, polygon={n_pts}점")
    bbox_results.append({"approx": approx, "n_pts": n_pts})

    label_str = f"{obj['name']} {n_pts}pts"
    img_bbox = draw_polygon(img_bbox, approx, obj["color"], label_str)

cv2.imwrite(str(OUTPUT_DIR / "sam_02_bbox.jpg"), img_bbox)
print(f"저장: {OUTPUT_DIR}/sam_02_bbox.jpg")

# ===== [3. YOLO mask vs SAM 나란히 비교] =====
print("\n[3] YOLO mask vs SAM 비교 이미지 생성")
yolo_img = cv2.imread(str(OUTPUT_DIR / "02_mask_only.jpg"))
sam_img   = img_bbox.copy()  # bbox 기반 SAM을 대표로 사용

if yolo_img is not None:
    # 제목 텍스트 추가
    cv2.putText(yolo_img, "YOLO V2 mask (mask_ratio=4)", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    cv2.putText(sam_img,  "SAM3 bbox-based", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    compare = np.hstack([yolo_img, sam_img])
    cv2.imwrite(str(OUTPUT_DIR / "sam_03_compare.jpg"), compare)
    print(f"저장: {OUTPUT_DIR}/sam_03_compare.jpg")
else:
    print("  YOLO mask 이미지 없음 — Step 2 먼저 실행 필요")

# ===== [요약] =====
print("\n" + "=" * 55)
print("[Step 3 SAM 단독 검증 결과 요약]")
print(f"{'객체':<8} {'YOLO polygon':>14} {'SAM click':>12} {'SAM bbox':>12}")
print("-" * 55)

yolo_pts = {"person": 300, "dog": 74}  # Step 2 결과
for i, obj in enumerate(YOLO_BBOXES):
    name     = obj["name"]
    yolo_n   = yolo_pts[name]
    click_n  = click_results[i]["n_pts"] if click_results[i] else "실패"
    bbox_n   = bbox_results[i]["n_pts"]  if bbox_results[i]  else "실패"
    print(f"{name:<8} {str(yolo_n):>14} {str(click_n):>12} {str(bbox_n):>12}")

print("=" * 55)
print()
print("[판단 기준]")
print("SAM polygon 점 수가 적고 육안으로 정확 → SAM 품질 양호")
print("SAM polygon이 YOLO보다 훨씬 적은 점수 → YOLO mask 해상도 문제 확인")
print("SAM bbox > SAM click 품질              → hybrid 모드에서 bbox 입력 권장")
print("=" * 55)
