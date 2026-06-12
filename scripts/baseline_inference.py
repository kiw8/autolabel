"""
Baseline 추론 스크립트
=====================
목적: 사전학습된 YOLO11-seg 모델로 우리 데이터에 추론을 돌려서
     "파인튜닝 전" 성능을 측정한다. 도메인 시프트가 얼마나 큰지 보는 단계.
입력: 트랙 #2에서 시각화했던 같은 이미지
출력: baseline 추론 결과 (시각화 + 콘솔 분석)
"""

import torch  # ← 추가: GPU 가용 여부를 확인하기 위해 임포트
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ===== 0. 디바이스 자동 선택 =====
# ← 추가: CUDA_VISIBLE_DEVICES=6,7 환경에서 index 0이 실제 물리 GPU 6번에 매핑됨
# ← 추가: CUDA 없으면 cpu로 fallback해서 환경 이식성 확보
device = 0 if torch.cuda.is_available() else "cpu"
print(f"[디바이스 선택] torch.cuda.is_available()={torch.cuda.is_available()} → device={device}")

# ===== 1. 설정 =====
IMAGE_PATH = "/home1/Jwson08/autolabel/data/images/GX010097[1]_011_0000.jpg"
OUTPUT_PATH = "/home1/Jwson08/autolabel/outputs/baseline_result.jpg"

# YOLO11 nano segmentation 모델
# 첫 실행 시 자동 다운로드됨 (약 6MB)
MODEL_NAME = "yolo11n-seg.pt"

# COCO 80 클래스 중 우리가 관심 있는 것만 (사전학습 모델은 COCO 기준)
TARGET_COCO_CLASSES = {
    0: "person",   # COCO ID 0 = person
    16: "dog",     # COCO ID 16 = dog
}

# 표시할 색상 (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),    # 빨강 - person
    16: (0, 255, 0),   # 초록 - dog
}

# 다른 클래스도 detection되면 회색으로 표시 (참고용)
OTHER_COLOR = (128, 128, 128)


# ===== 2. 모델 로드 (첫 실행 시 자동 다운로드) =====
print("[1/4] 모델 로드 중...")
model = YOLO(MODEL_NAME)
print(f"      모델: {MODEL_NAME}")
print(f"      디바이스: {'cuda:' + str(device) if isinstance(device, int) else device}")  # ← 수정: model.device는 predict 전엔 cpu로 표시돼서 선택한 device 변수를 직접 출력


# ===== 3. 추론 실행 =====
print("\n[2/4] 추론 실행 중...")
results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,        # confidence threshold (0.25 = 신뢰도 25% 이상만)
    verbose=False,    # 콘솔 로그 줄이기
    device=device,    # ← 추가: 위에서 선택한 device(0=GPU, "cpu")를 명시적으로 전달
)

# results는 리스트로 반환됨 (배치 처리 가능해서). 첫 번째 결과만 사용
result = results[0]

print(f"      감지된 객체 수: {len(result.boxes)}")


# ===== 4. 결과 분석 =====
print("\n[3/4] 결과 분석...")
print("=" * 60)

img = cv2.imread(IMAGE_PATH)
H, W = img.shape[:2]

target_count = {"person": 0, "dog": 0}
other_count = 0

if result.masks is None:
    print("      ⚠️ Segmentation 마스크가 감지되지 않음")
else:
    # 각 detection 결과를 순회
    for idx, (mask, box, cls, conf) in enumerate(zip(
        result.masks.xy,     # polygon 좌표 (픽셀 단위 리스트)
        result.boxes.xyxy,   # bbox
        result.boxes.cls,    # 클래스 ID
        result.boxes.conf,   # 신뢰도
    )):
        class_id = int(cls.item())
        confidence = float(conf.item())

        # 우리 관심 클래스인지 확인
        if class_id in TARGET_COCO_CLASSES:
            class_name = TARGET_COCO_CLASSES[class_id]
            color = CLASS_COLORS[class_id]
            target_count[class_name] += 1
            print(f"  ✅ 객체 {idx+1}: {class_name} (conf={confidence:.2f}, polygon 점 {len(mask)}개)")
        else:
            # 관심 외 클래스 (참고용)
            class_name = result.names[class_id]  # COCO 이름
            color = OTHER_COLOR
            other_count += 1
            print(f"  ⚠️ 객체 {idx+1}: {class_name} (conf={confidence:.2f}) - 관심 외 클래스")

        # polygon 그리기
        points_np = mask.astype(np.int32)
        cv2.polylines(img, [points_np], isClosed=True, color=color, thickness=2)

        # 클래스 + confidence 표시
        if len(points_np) > 0:
            cx, cy = points_np[0]
            label_text = f"{class_name} {confidence:.2f}"
            cv2.putText(
                img, label_text, (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

print("=" * 60)
print(f"\n[Baseline 성능 요약]")
print(f"  - person 감지: {target_count['person']}개")
print(f"  - dog 감지:    {target_count['dog']}개")
print(f"  - 관심 외 객체: {other_count}개")


# ===== 5. 결과 시각화 =====
print("\n[4/4] 결과 시각화 및 저장...")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14, 10))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Baseline (pretrained YOLO11n-seg) Result")
plt.tight_layout()
plt.show()

cv2.imwrite(OUTPUT_PATH, img)
print(f"      저장 완료: {OUTPUT_PATH}")
print("\n완료. 트랙 #2 결과(visualization_check.jpg)와 비교해보세요.")
