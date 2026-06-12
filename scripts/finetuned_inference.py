"""
파인튜닝 모델 추론 스크립트
============================
목적: 학습된 v1 모델로 추론하고, 트랙 #3 baseline과 비교 가능한 자료 생성
입력: models/yolo11s_seg_v1_20260601/best.pt, data/images/GX010097[1]_011_0000.jpg
출력: outputs/finetuned_result.jpg
"""

# ===== [import] =====
import cv2                            # 이미지 읽기·polygon 그리기·파일 저장
import numpy as np                    # polygon 좌표 타입 변환 (float → int32)
import matplotlib                     # use()는 pyplot import 이전에 호출해야 적용됨
matplotlib.use("Agg")                 # 원격 서버 디스플레이 없는 환경에서 파일 저장 전용 백엔드
import matplotlib.pyplot as plt       # (현재 직접 사용하지 않으나 향후 확장용으로 유지)
from ultralytics import YOLO          # YOLO 모델 로드·추론 인터페이스


# ===== [설정] =====
MODEL_PATH  = "/home1/Jwson08/autolabel/models/yolo11s_seg_v1_20260601/best.pt"
IMAGE_PATH  = "/home1/Jwson08/autolabel/data/images/GX010097[1]_011_0000.jpg"
OUTPUT_PATH = "/home1/Jwson08/autolabel/outputs/finetuned_result.jpg"

# 파인튜닝 모델은 클래스 ID가 COCO가 아닌 우리 데이터셋 기준 (0=person, 1=dog)
CLASS_NAMES  = {0: "person", 1: "dog"}
CLASS_COLORS = {
    0: (0,   0,   255),   # person — 빨강 (BGR)
    1: (0,   255, 0  ),   # dog    — 초록 (BGR)
}


# ===== [모델 로드] =====
model = YOLO(MODEL_PATH)              # 학습된 가중치 로드 (COCO pretrained 아닌 elevator 파인튜닝 버전)

model_name = "/".join(MODEL_PATH.split("/")[-2:])   # 경로에서 "yolo11s_seg_v1_20260601/best.pt" 부분만 추출
image_name = IMAGE_PATH.split("/")[-1]

print(f"모델: {model_name}")
print(f"디바이스: cuda:0")            # CUDA_VISIBLE_DEVICES=6,7 환경에서 index 0 = 물리 GPU 6번
print(f"이미지: {image_name}")


# ===== [추론] =====
results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,                        # baseline(트랙 #3)과 동일한 threshold로 비교 가능하게 통일
    device=0,                         # CUDA_VISIBLE_DEVICES로 이미 GPU 지정됨, index 0 사용
    verbose=False,                    # ultralytics 내부 로그 억제 (우리 형식으로만 출력)
)
result = results[0]                   # predict는 배치 리스트 반환, 단일 이미지라 첫 번째만 사용

print(f"감지된 객체 수: {len(result.boxes)}")


# ===== [시각화 & 콘솔 출력] =====
img = cv2.imread(IMAGE_PATH)

if result.masks is None:
    print("⚠️  Segmentation 마스크 없음 — conf threshold를 낮춰보세요")
else:
    for idx, (mask_xy, cls, conf) in enumerate(zip(
        result.masks.xy,      # polygon 좌표 리스트 (픽셀 단위, float)
        result.boxes.cls,     # 클래스 ID tensor
        result.boxes.conf,    # confidence tensor
    )):
        class_id   = int(cls.item())
        confidence = float(conf.item())
        class_name = CLASS_NAMES.get(class_id, f"class{class_id}")
        color      = CLASS_COLORS.get(class_id, (180, 180, 180))

        # 트랙 #3과 동일한 출력 형식으로 baseline과 나란히 비교 가능
        print(f"객체 {idx + 1}: {class_name} (conf={confidence:.2f}, polygon 점 {len(mask_xy)}개)")

        # polygon 외곽선
        pts = mask_xy.astype(np.int32)          # cv2는 int32 요구
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        # 클래스 라벨 텍스트 (polygon 첫 점 위에 표시)
        if len(pts) > 0:
            cx, cy = pts[0]
            cv2.putText(
                img,
                f"{class_name} {confidence:.2f}",
                (cx, max(cy - 10, 20)),         # 화면 위로 넘어가지 않게 최소 y=20 보정
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2,
            )


# ===== [저장] =====
cv2.imwrite(OUTPUT_PATH, img)
print(f"저장 완료: {OUTPUT_PATH}")
