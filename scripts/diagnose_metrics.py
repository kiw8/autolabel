"""
Step 5 — 수치화 스크립트
========================
목적: 진단 이미지 기준 IoU / polygon 점수 / 추론 시간 / val mAP를 한 파일로 집계.

측정 항목:
  1) YOLO mask vs SAM mask  IoU (per object)   ← 두 마스크가 얼마나 다른지
  2) Polygon 점수 비교 (Steps 2~4 결과 집계)
  3) 추론 시간 비교 (Step 4 결과 재측정)
  4) YOLO V2 val mAP (eval_metrics_v2.txt)

출력:
  outputs/diagnose/step5_metrics.txt  ← 브리핑용 수치 파일
  outputs/diagnose/step5_iou.jpg      ← IoU 시각화 (겹침 영역)
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, SAM

sys.path.insert(0, "/home1/Jwson08/work/labeling-project")

# ===== [설정] =====
YOLO_PATH  = "/home1/Jwson08/autolabel/runs/segment/runs/segment/train_v2/weights/best.pt"
SAM_PATH   = "/nas03/models/labeling-project/sam3.pt"
IMAGE_PATH = "/home1/Jwson08/autolabel/data/images/GX010097[1]_011_0000.jpg"
OUTPUT_DIR = Path("/home1/Jwson08/autolabel/outputs/diagnose")
METRICS_TXT = OUTPUT_DIR / "step5_metrics.txt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_DP_EPSILON = 0.3

# Step 2에서 확인한 YOLO bbox 좌표 (SAM 클릭/bbox 프롬프트용)
OBJECTS = [
    {"name": "person", "class_id": 0, "bbox": [952, 412, 1769, 860],
     "click": [1360, 636], "color_yolo": (0, 0, 255), "color_sam": (255, 100, 0)},
    {"name": "dog",    "class_id": 1, "bbox": [934, 363, 1231, 540],
     "click": [1082, 451], "color_yolo": (0, 200, 0), "color_sam": (0, 200, 200)},
]

# ===== [헬퍼] =====
def mask_to_polygon(mask_data):
    mask_np = mask_data.cpu().numpy()
    mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = _DP_EPSILON * cv2.arcLength(largest, closed=True) / 100
    approx  = cv2.approxPolyDP(largest, epsilon, closed=True)
    return approx if len(approx) >= 3 else None

def poly_to_mask(pts, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask

def compute_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return round(float(inter) / float(union), 4) if union > 0 else 0.0

# ===== [이미지 로드] =====
img_orig = cv2.imread(IMAGE_PATH)
H, W = img_orig.shape[:2]
print(f"이미지: {W}x{H}\n")

# ===== [1. YOLO 추론] =====
print("[1] YOLO V2 추론")
yolo_model = YOLO(YOLO_PATH)

t0 = time.perf_counter()
yolo_results = yolo_model.predict(source=IMAGE_PATH, conf=0.25, iou=0.5, device=0, verbose=False)
t_yolo = round((time.perf_counter() - t0) * 1000, 1)

yolo_r = yolo_results[0]
yolo_masks_raw = {}  # 원본 저해상도 mask (IoU 계산용)
yolo_poly_pts  = {}  # polygon 점수용

if yolo_r.masks is not None:
    for mask_xy, mask_data, cls_t, conf_t in zip(
        yolo_r.masks.xy, yolo_r.masks.data, yolo_r.boxes.cls, yolo_r.boxes.conf
    ):
        cid = int(cls_t.item())
        name = OBJECTS[cid]["name"] if cid < len(OBJECTS) else f"class{cid}"
        n_pts = len(mask_xy)
        yolo_poly_pts[name] = n_pts

        # 이미지 크기로 리사이즈된 마스크 생성
        mask_np = mask_data.cpu().numpy()
        mask_full = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_LINEAR)
        yolo_masks_raw[name] = (mask_full > 0.5).astype(np.uint8) * 255

        print(f"  {name}: conf={conf_t.item():.2f}, polygon={n_pts}점")

print(f"  YOLO 추론시간: {t_yolo}ms\n")

# ===== [2. SAM 추론 (bbox 기반)] =====
print("[2] SAM3 추론 (bbox 기반)")
sam_model = SAM(SAM_PATH)

sam_poly_pts  = {}
sam_masks_raw = {}

t_sam_total = 0.0
for obj in OBJECTS:
    t0 = time.perf_counter()
    results = sam_model.predict(
        source=IMAGE_PATH,
        bboxes=[obj["bbox"]],
        device=0,
        verbose=False,
    )
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    t_sam_total += elapsed

    r = results[0]
    if r.masks is None or len(r.masks) == 0:
        print(f"  {obj['name']}: 마스크 없음")
        continue

    scores   = r.boxes.conf if r.boxes is not None and len(r.boxes) > 0 else None
    best_idx = int(scores.argmax()) if scores is not None and len(scores) > 0 else 0
    approx   = mask_to_polygon(r.masks.data[best_idx])

    if approx is None:
        print(f"  {obj['name']}: polygon 변환 실패")
        continue

    n_pts = len(approx)
    sam_poly_pts[obj["name"]] = n_pts

    # polygon → 풀 해상도 마스크
    sam_masks_raw[obj["name"]] = poly_to_mask(approx.reshape(-1, 2), H, W)

    print(f"  {obj['name']}: polygon={n_pts}점 ({elapsed}ms)")

t_sam_total = round(t_sam_total, 1)
print(f"  SAM 총 추론시간: {t_sam_total}ms\n")

# ===== [3. IoU 계산] =====
print("[3] IoU 계산 (YOLO mask vs SAM mask)")
ious = {}
for obj in OBJECTS:
    name = obj["name"]
    if name in yolo_masks_raw and name in sam_masks_raw:
        iou = compute_iou(yolo_masks_raw[name], sam_masks_raw[name])
        ious[name] = iou
        print(f"  {name}: IoU = {iou:.4f}")
    else:
        print(f"  {name}: 계산 불가 (마스크 없음)")

# ===== [4. IoU 시각화] =====
print("\n[4] IoU 시각화 생성")
img_iou = img_orig.copy()
overlay = img_orig.copy()

for obj in OBJECTS:
    name = obj["name"]
    if name not in yolo_masks_raw or name not in sam_masks_raw:
        continue

    mask_y = yolo_masks_raw[name]
    mask_s = sam_masks_raw[name]

    # YOLO만 있는 영역 (파랑 계열)
    only_yolo = np.logical_and(mask_y > 0, mask_s == 0)
    # SAM만 있는 영역 (초록 계열)
    only_sam  = np.logical_and(mask_y == 0, mask_s > 0)
    # 겹치는 영역 (노랑)
    both      = np.logical_and(mask_y > 0, mask_s > 0)

    overlay[only_yolo] = obj["color_yolo"]
    overlay[only_sam]  = obj["color_sam"]
    overlay[both]      = (0, 255, 255)  # 노랑 = 겹침

img_iou = cv2.addWeighted(overlay, 0.45, img_iou, 0.55, 0)

# 범례
legend_y = 60
for obj in OBJECTS:
    name = obj["name"]
    if name in ious:
        cv2.putText(img_iou, f"{name} IoU={ious[name]:.3f}",
                    (30, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        legend_y += 40

cv2.putText(img_iou, "Yellow=overlap  Blue=YOLO_only  Green=SAM_only",
            (30, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

cv2.imwrite(str(OUTPUT_DIR / "step5_iou.jpg"), img_iou)
print(f"  저장: step5_iou.jpg\n")

# ===== [5. 전체 수치 파일 저장] =====
lines = []
lines.append("=" * 60)
lines.append("Step 5 — 수치화 결과 (브리핑용)")
lines.append("=" * 60)
lines.append(f"진단 이미지: GX010097[1]_011_0000.jpg  ({W}x{H})")
lines.append("")

lines.append("[A] YOLO V2 val 전체 데이터셋 mAP")
lines.append("-" * 40)
lines.append("  Box  mAP@0.5      : 0.9111")
lines.append("  Box  mAP@0.5:0.95 : 0.8142")
lines.append("  Mask mAP@0.5      : 0.9175  ← 탐지 정확도 지표")
lines.append("  Mask mAP@0.5:0.95 : 0.7786  ← 경계 정밀도 지표")
lines.append("  * mAP@0.5:0.95 낮음 → 경계 정밀도 개선 여지 있음")
lines.append("")

lines.append("[B] Polygon 점수 비교 (낮을수록 매끄러운 경계)")
lines.append("-" * 40)
header = f"  {'객체':<8} {'YOLO V2':>10} {'SAM bbox':>10} {'SAM click':>10} {'감소율':>8}"
lines.append(header)
lines.append("  " + "-" * 44)
for obj in OBJECTS:
    name   = obj["name"]
    y_pts  = yolo_poly_pts.get(name, "N/A")
    s_pts  = sam_poly_pts.get(name, "N/A")
    c_pts  = {"person": 40, "dog": 32}[name]  # Step 3 click 결과
    if isinstance(y_pts, int) and isinstance(s_pts, int):
        ratio = round((1 - s_pts / y_pts) * 100, 1)
        lines.append(f"  {name:<8} {y_pts:>10}점 {s_pts:>10}점 {c_pts:>10}점 {ratio:>7}%↓")
    else:
        lines.append(f"  {name:<8} {str(y_pts):>10} {str(s_pts):>10} {str(c_pts):>10}")
lines.append("")

lines.append("[C] IoU (YOLO mask vs SAM mask, 동일 이미지·동일 객체)")
lines.append("-" * 40)
lines.append("  IoU = 겹치는 픽셀 / 전체 픽셀 (1.0=완전 일치)")
for obj in OBJECTS:
    name = obj["name"]
    iou  = ious.get(name, "N/A")
    if isinstance(iou, float):
        grade = "양호" if iou >= 0.85 else ("주의" if iou >= 0.70 else "불량")
        lines.append(f"  {name:<8}: {iou:.4f}  [{grade}]")
    else:
        lines.append(f"  {name:<8}: {iou}")
lines.append("  * IoU < 0.85 이면 YOLO와 SAM 경계가 의미 있게 다름")
lines.append("")

lines.append("[D] 추론 시간 비교")
lines.append("-" * 40)
lines.append(f"  YOLO 단독   : {t_yolo}ms")
lines.append(f"  SAM  단독   : {t_sam_total}ms  (첫 호출 포함, 이미지 인코딩 포함)")
lines.append(f"  Hybrid      : ~21,000ms  (Step 4 측정값 — YOLO + SAM 인코딩)")
lines.append(f"  SAM 2회차   : ~1,000ms   (같은 이미지, 캐시 적용 시)")
lines.append("")

lines.append("[E] 종합 진단")
lines.append("-" * 40)
lines.append("  YOLO 탐지 정확도 : Mask mAP@0.5 = 0.9175  → 정상")
lines.append("  YOLO 경계 정밀도 : Mask mAP@0.5:0.95 = 0.7786  → 개선 여지")
lines.append("  YOLO polygon     : person 300점 (과도) → mask_ratio=4 저해상도 원인")
lines.append("  SAM polygon      : 34~40점 → 매끄러운 경계, SAM 정상 동작")
lines.append("  Hybrid 시간      : ~21초  → UX 병목 가능성 (캐시 전략 필요)")
lines.append("")
lines.append("  결론: YOLO 탐지는 정상, 경계 품질이 병목")
lines.append("        V3(mask_ratio=1)로 YOLO 경계 개선 + hybrid 시 SAM이 보완")
lines.append("=" * 60)

report = "\n".join(lines)
print(report)

METRICS_TXT.write_text(report, encoding="utf-8")
print(f"\n저장: {METRICS_TXT}")
