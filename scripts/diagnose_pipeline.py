"""
파이프라인 비교 진단 스크립트 (Step 4)
=======================================
목적: Predictor 3가지 모드를 동일 이미지에서 실행해 polygon 품질 / 추론 시간 비교.

  mode="yolo"   : YOLO polygon 그대로 저장
  mode="hybrid" : YOLO bbox → SAM bbox 프롬프트 → SAM polygon
  mode="sam"    : 수동 클릭 → SAM click 프롬프트 → SAM polygon

이전 단계 결과 (동일 이미지 기준):
  Step 2 (YOLO V2)     : person=300점, dog=74점
  Step 3 (SAM bbox)    : person=34점,  dog=35점
  Step 3 (SAM click)   : person=40점,  dog=32점

출력:
  outputs/diagnose/pipeline_01_yolo.jpg    ← yolo 모드
  outputs/diagnose/pipeline_02_hybrid.jpg  ← hybrid 모드
  outputs/diagnose/pipeline_03_sam.jpg     ← sam 모드
  outputs/diagnose/pipeline_04_compare.jpg ← 3가지 나란히 비교
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# predictor.py import를 위해 labeling-project 루트를 경로에 추가
sys.path.insert(0, "/home1/Jwson08/work/labeling-project")

from ai.predictor import Predictor

# ===== [설정] =====
YOLO_PATH  = "/nas03/models/labeling-project/elevator_v2.pt"
SAM_PATH   = "/nas03/models/labeling-project/sam3.pt"
IMAGE_PATH = "/home1/Jwson08/autolabel/data/images/GX010097[1]_011_0000.jpg"
OUTPUT_DIR = Path("/home1/Jwson08/autolabel/outputs/diagnose")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Step 2 결과 좌표: person bbox 중심점 클릭 (SAM 모드용)
CLICK_PERSON = [[1360, 636, 1]]  # foreground
CLICK_DOG    = [[1082, 451, 1]]  # foreground

CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}  # BGR: person=red, dog=green

# ===== [헬퍼: polygon 시각화] =====
def draw_result(img_orig, objects, title):
    img = img_orig.copy()
    overlay = img_orig.copy()

    for obj in objects:
        cid   = obj["class_id"]
        name  = obj["class_name"]
        conf  = obj["confidence"]
        poly  = obj["polygon"]
        color = CLASS_COLORS.get(cid, (180, 180, 180))

        if len(poly) < 3:
            continue

        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2)

        cx, cy = pts[0]
        cv2.putText(img, f"{name} {conf:.2f} ({len(poly)}pts)",
                    (cx, max(cy - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.putText(img, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    return img

# ===== [Predictor 로드] =====
print("Predictor 로드 중...")
predictor = Predictor(
    model_path=YOLO_PATH,
    sam_model_path=SAM_PATH,
    device=0,
)
print("로드 완료\n")

results_summary = []

# ===== [1. YOLO 모드] =====
print("[1] mode=yolo")
t0 = time.perf_counter()
r_yolo = predictor.predict(IMAGE_PATH, mode="yolo", conf_threshold=0.25)
t_yolo = round((time.perf_counter() - t0) * 1000, 1)

for obj in r_yolo["objects"]:
    n = len(obj["polygon"])
    print(f"  {obj['class_name']}: conf={obj['confidence']:.2f}, polygon={n}점")

img_yolo = draw_result(cv2.imread(IMAGE_PATH), r_yolo["objects"], f"YOLO only  ({t_yolo}ms)")
cv2.imwrite(str(OUTPUT_DIR / "pipeline_01_yolo.jpg"), img_yolo)
print(f"  추론시간: {t_yolo}ms → 저장: pipeline_01_yolo.jpg\n")
results_summary.append(("yolo", r_yolo["objects"], t_yolo))

# ===== [2. Hybrid 모드] =====
print("[2] mode=hybrid  (YOLO bbox → SAM polygon)")
t0 = time.perf_counter()
r_hybrid = predictor.predict(IMAGE_PATH, mode="hybrid", conf_threshold=0.25)
t_hybrid = round((time.perf_counter() - t0) * 1000, 1)

for obj in r_hybrid["objects"]:
    n = len(obj["polygon"])
    print(f"  {obj['class_name']}: conf={obj['confidence']:.2f}, polygon={n}점")

img_hybrid = draw_result(cv2.imread(IMAGE_PATH), r_hybrid["objects"], f"Hybrid YOLO+SAM  ({t_hybrid}ms)")
cv2.imwrite(str(OUTPUT_DIR / "pipeline_02_hybrid.jpg"), img_hybrid)
print(f"  추론시간: {t_hybrid}ms → 저장: pipeline_02_hybrid.jpg\n")
results_summary.append(("hybrid", r_hybrid["objects"], t_hybrid))

# ===== [3. SAM 모드 (클릭)] =====
print("[3] mode=sam  (person 클릭)")
t0 = time.perf_counter()
r_sam_person = predictor.predict(
    IMAGE_PATH, mode="sam",
    click_points=CLICK_PERSON, click_class_id=0,
)
t0b = time.perf_counter()
r_sam_dog = predictor.predict(
    IMAGE_PATH, mode="sam",
    click_points=CLICK_DOG, click_class_id=1,
)
t_sam = round((time.perf_counter() - t0) * 1000, 1)

sam_objects = r_sam_person["objects"] + r_sam_dog["objects"]
for obj in sam_objects:
    n = len(obj["polygon"])
    print(f"  {obj['class_name']}: conf={obj['confidence']:.2f}, polygon={n}점")

img_sam = draw_result(cv2.imread(IMAGE_PATH), sam_objects, f"SAM click  ({t_sam}ms)")
cv2.imwrite(str(OUTPUT_DIR / "pipeline_03_sam.jpg"), img_sam)
print(f"  추론시간: {t_sam}ms → 저장: pipeline_03_sam.jpg\n")
results_summary.append(("sam(click)", sam_objects, t_sam))

# ===== [4. 나란히 비교 이미지] =====
print("[4] 3모드 비교 이미지 생성")
compare = np.hstack([img_yolo, img_hybrid, img_sam])
cv2.imwrite(str(OUTPUT_DIR / "pipeline_04_compare.jpg"), compare)
print(f"  저장: pipeline_04_compare.jpg\n")

# ===== [최종 요약] =====
print("=" * 65)
print(f"[Step 4 파이프라인 비교 결과]")
print(f"{'모드':<14} {'객체':<8} {'polygon 점수':>12} {'추론시간':>10}")
print("-" * 65)

for mode_name, objects, elapsed in results_summary:
    for obj in objects:
        n = len(obj["polygon"])
        print(f"{mode_name:<14} {obj['class_name']:<8} {n:>12}점 {elapsed:>9}ms")

print("=" * 65)
print()
print("[판단 기준]")
print("yolo polygon이 많고 hybrid/sam이 적음  → hybrid/sam 품질 우수")
print("hybrid 추론시간이 yolo보다 크게 늘면   → SAM 추가 비용 vs 품질 트레이드오프 확인")
print("hybrid ≈ sam(click) 품질               → 자동화(hybrid)로 수동 클릭 대체 가능")
print("=" * 65)
