"""
시각화 검증 스크립트
=====================
목적: CVAT 라벨이 실제 이미지의 객체와 정확히 매칭되는지 시각적으로 확인
입력: 이미지 1장 + 그에 매칭되는 .txt 라벨 1개
출력: polygon이 그려진 이미지 (화면 표시 + 파일 저장)
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')   # GUI 없는 원격 서버용 백엔드
import matplotlib.pyplot as plt

# ===== 1. 설정 =====
IMAGE_PATH = "/home1/Jwson08/autolabel/data/images/GX010097[1]_011_0000.jpg"
LABEL_PATH = "/home1/Jwson08/autolabel/data/labels/GX010097[1]_011_0000.txt"
OUTPUT_PATH = "/home1/Jwson08/autolabel/outputs/visualization_check.jpg"

# 클래스 매핑 (CVAT 작업 시 person을 먼저 추가했음 → 0)
CLASS_NAMES = {
    0: "person",
    1: "dog",
}

# 클래스별 색상 (BGR 순서, OpenCV 기본)
CLASS_COLORS = {
    0: (0, 0, 255),    # 빨강 — 사람
    1: (0, 255, 0),    # 초록 — 강아지
}


# ===== 2. 이미지 로드 =====
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {IMAGE_PATH}")

H, W = img.shape[:2]
print(f"이미지 크기: 가로 {W} × 세로 {H}")


# ===== 3. 라벨 파일 읽기 =====
with open(LABEL_PATH, "r") as f:
    lines = f.readlines()

print(f"객체 개수: {len(lines)}")


# ===== 4. 각 라인 파싱 → polygon 그리기 =====
for idx, line in enumerate(lines):
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = list(map(float, parts[1:]))  # 정규화된 polygon 좌표 (x1,y1,x2,y2,...)

    # 정규화 좌표(0~1) → 픽셀 좌표 변환
    points = []
    for i in range(0, len(coords), 2):
        x_pixel = int(coords[i] * W)
        y_pixel = int(coords[i + 1] * H)
        points.append([x_pixel, y_pixel])

    points_np = np.array(points, dtype=np.int32)

    # polygon 외곽선 그리기
    color = CLASS_COLORS.get(class_id, (255, 255, 255))  # 모르는 클래스면 흰색
    cv2.polylines(img, [points_np], isClosed=True, color=color, thickness=2)

    # 클래스 이름 표시 (polygon 첫 점 근처)
    cx, cy = points_np[0]
    label_text = CLASS_NAMES.get(class_id, f"class_{class_id}")
    cv2.putText(
        img,
        label_text,
        (cx, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    print(f"  객체 {idx+1}: {label_text} (polygon 점 {len(points)}개)")


# ===== 5. 결과 시각화 =====
# OpenCV는 BGR 순서, matplotlib는 RGB 순서라서 변환 필요
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14, 10))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Label Visualization Check")
plt.tight_layout()
plt.show()


# ===== 6. 결과 파일 저장 =====
cv2.imwrite(OUTPUT_PATH, img)
print(f"\n결과 이미지 저장 완료: {OUTPUT_PATH}")
