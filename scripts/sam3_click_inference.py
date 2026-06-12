"""
SAM 3 클릭 추론 스크립트
========================
목적: 사람이 클릭한 좌표(점)를 SAM 3에 프롬프트로 넘겨
     해당 객체의 세그멘테이션 마스크를 polygon으로 뽑는다.
     트랙 #4: YOLO baseline 결과와 SAM 3 결과를 비교하는 단계.
입력: CVAT 라벨 파일 (정규화 polygon 좌표)
출력: outputs/sam3_click_result.jpg (polygon 시각화)
"""

# ===== [import] =====
import torch                          # ← 추가: GPU 가용 여부 확인을 위해 최상단 임포트
import cv2                            # ← 추가: 이미지 읽기·그리기·polygon 변환에 사용
import numpy as np                    # ← 추가: 마스크(float32) → uint8 변환 등 배열 연산용
import matplotlib                     # ← 추가: 백엔드 설정을 use() 전에 임포트해야 함
matplotlib.use("Agg")                 # ← 추가: 원격 서버는 디스플레이가 없어서 파일 저장 전용 백엔드 강제
import matplotlib.pyplot as plt       # ← 추가: 시각화 결과를 파일로 렌더링하기 위해 임포트
from ultralytics import SAM           # ← 추가: ultralytics SAM 3 인터페이스 (sam3_b.pt 로드에 사용)


# ===== [0. 디바이스 자동 선택] =====
# ← 추가: CUDA_VISIBLE_DEVICES=6,7 환경에서 index 0 = 물리 GPU 6번
# ← 추가: CUDA 없으면 "cpu"로 fallback → 로컬 CPU 환경에서도 실행 가능
device = 0 if torch.cuda.is_available() else "cpu"
print(f"[Step 0] 디바이스 선택: torch.cuda.is_available()={torch.cuda.is_available()} → device={device}")


# ===== [1. 설정] =====
IMAGE_PATH   = "/home1/Jwson08/autolabel/data/images/GX010097[1]_011_0000.jpg"
OUTPUT_PATH  = "/home1/Jwson08/autolabel/outputs/sam3_click_result.jpg"
MODEL_NAME   = "/home1/Jwson08/autolabel/models/sam3.pt"
LABEL_PATH   = "/home1/Jwson08/autolabel/data/labels/GX010097[1]_011_0000.txt"
# ← 추가: CVAT에서 내보낸 라벨 파일 경로. 각 줄 = 하나의 객체 (class_id + 정규화 polygon 좌표)

# 클래스 ID별 이름과 색상 (BGR 순서)
CLASS_NAMES  = {0: "person", 1: "dog"}
# ← 변경: 기존 CLICK_POINTS 인덱스 기반 → class_id 기반 dict로 교체 (라벨 파일에서 class_id를 읽으므로)
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}
# ← 변경: 0=person 빨강, 1=dog 초록. dict 조회로 class_id와 색상이 항상 1:1 매핑됨

CLICK_DOT_COLOR  = (0, 255, 255)      # ← 설정: 클릭 점 색상 = 노란색 (BGR에서 노랑은 0,255,255)
CLICK_DOT_RADIUS = 8                  # ← 설정: 클릭 점 반지름(픽셀)
DP_EPSILON       = 1.5                # ← 수정: 2.0 → 1.5. 허용 오차를 줄여 polygon이 객체 윤곽을 더 촘촘히 따라가게


# ===== [2. SAM 3 모델 로드] =====
print(f"\n[Step 1/4] SAM 3 모델 로드 중: {MODEL_NAME}")
model = SAM(MODEL_NAME)               # ← 핵심: ultralytics가 model 이름에 'sam3'이 있으면 SAM3Predictor로 자동 분기
device_str = f"cuda:{device}" if isinstance(device, int) else device
print(f"           디바이스: {device_str}")


# ===== [3. 이미지 로드 + 크기 출력] =====
print(f"\n[Step 2/4] 이미지 로드: {IMAGE_PATH}")
img = cv2.imread(IMAGE_PATH)          # ← 추가: OpenCV는 BGR 순서로 읽음
if img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없음: {IMAGE_PATH}")
H, W = img.shape[:2]
print(f"           이미지 크기: width={W}, height={H}")
# ← 수정: 기존에도 출력했지만 width/height 레이블을 명시적으로 붙여 좌표 변환 계산 검증에 활용


# ===== [4. 라벨 파일에서 클릭 좌표 자동 추출] =====
print(f"\n[Step 3/4] 라벨 파일에서 클릭 좌표 추출: {LABEL_PATH}")
# ← 추가: 기존 하드코딩 CLICK_POINTS 대신 라벨 파일의 첫 polygon 점을 클릭 좌표로 자동 사용

click_objects = []  # [(class_id, [[px, py], ...]), ...]
# ← 변경: 단일 점(px, py) 대신 점 목록으로 저장 → person은 2점, dog는 1점을 클래스별로 다르게 처리

with open(LABEL_PATH) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        class_id = int(parts[0])

        xs = [float(v) for v in parts[1::2]]
        ys = [float(v) for v in parts[2::2]]
        # ← 추가: parts[1::2]는 홀수 인덱스(x좌표들), parts[2::2]는 짝수 인덱스(y좌표들)
        #         라벨 형식이 class_id x1 y1 x2 y2 ... 이므로 슬라이싱으로 x/y를 한번에 분리

        px_first = int(xs[0] * W)
        py_first = int(ys[0] * H)
        # ← 추가: 첫 polygon 점 → 픽셀 변환. 기존과 동일하게 SAM의 첫 번째 프롬프트 점으로 사용

        label_name = CLASS_NAMES.get(class_id, f"class{class_id}")

        if class_id == 0:
            cx = int(np.mean(xs) * W)
            cy = int(np.mean(ys) * H)
            # ← 유지: polygon 무게중심 = 상반신 포함용 두 번째 힌트

            # 다리/발 추가 점: x가 가장 낮은 하위 15% 점들에서 추출
            # ← 추가: 이 탑뷰 이미지에서 x가 낮을수록 이미지 왼쪽 = 다리·발 방향이므로
            #         낮은 x 구간 안에서 y 최솟값(무릎) · y 최댓값(발끝)을 각각 1점씩 뽑음
            low_x_region = sorted(zip(xs, ys), key=lambda p: p[0])
            cutoff = max(2, len(low_x_region) // 7)   # 하위 ~15%
            low_x_region = low_x_region[:cutoff]
            leg_nx,  leg_ny  = min(low_x_region, key=lambda p: p[1])  # y 최소 = 상부 다리
            foot_nx, foot_ny = max(low_x_region, key=lambda p: p[1])  # y 최대 = 발끝
            leg_px,  leg_py  = int(leg_nx  * W), int(leg_ny  * H)
            foot_px, foot_py = int(foot_nx * W), int(foot_ny * H)

            pts = [[px_first, py_first], [cx, cy], [leg_px, leg_py], [foot_px, foot_py]]
            print(f"           {label_name} 클릭 좌표: 첫점({px_first},{py_first}), 중심({cx},{cy}), 다리({leg_px},{leg_py}), 발({foot_px},{foot_py})")
        else:
            pts = [[px_first, py_first]]
            # ← 유지: dog(class 1)는 1점으로도 충분히 잡혔으므로 기존 방식 그대로
            print(f"           {label_name} 클릭 좌표: ({px_first}, {py_first})")

        click_objects.append((class_id, pts))
        # ← 추가: 정규화 원본값과 변환된 픽셀값을 함께 출력해 좌표 변환이 올바른지 바로 확인 가능


# ===== [5. 클릭 좌표별 SAM 추론 + polygon 변환] =====
print(f"\n[Step 4/4] SAM 3 클릭 추론 시작 ({len(click_objects)}개 객체)")

for idx, (class_id, pts) in enumerate(click_objects):
    # ← 변경: (class_id, x, y) → (class_id, pts). pts는 [[x,y], ...] 리스트
    color      = CLASS_COLORS.get(class_id, (255, 255, 255))
    label_name = CLASS_NAMES.get(class_id, f"class{class_id}")
    print(f"\n  [{idx+1}/{len(click_objects)}] {label_name}(class {class_id}) 클릭 좌표: {pts}")

    # --- SAM 3 추론 ---
    results = model.predict(
        source=IMAGE_PATH,
        points=pts,                  # ← 변경: [[x,y]] 고정 → pts 변수. person은 2점, dog는 1점이 자동 적용
        labels=[1] * len(pts),       # ← 변경: [1] 고정 → 점 수만큼 foreground(1) 레이블 동적 생성
        device=device,
        imgsz=1008,
        verbose=False,
    )
    result = results[0]

    # --- 마스크 유효성 확인 ---
    if result.masks is None or len(result.masks) == 0:
        print(f"  ⚠️  마스크가 생성되지 않음 — 라벨의 첫 점이 객체 경계 위일 수 있음")
        continue

    masks_tensor = result.masks.data
    scores = result.boxes.conf if (result.boxes is not None and len(result.boxes) > 0) else None

    if scores is not None and len(scores) > 0:
        best_idx = int(scores.argmax())
    else:
        best_idx = 0

    mask_np   = masks_tensor[best_idx].cpu().numpy()
    mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255

    # --- 마스크 → polygon ---
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        print(f"  ⚠️  contour 추출 실패 (마스크가 너무 작거나 비어 있음)")
        continue

    largest = max(contours, key=cv2.contourArea)
    epsilon  = DP_EPSILON * cv2.arcLength(largest, closed=True) / 100
    approx   = cv2.approxPolyDP(largest, epsilon, closed=True)

    print(f"  ✅ 마스크 생성 성공")
    print(f"     원본 contour 점 수: {len(largest)}")
    print(f"     단순화 후 점 수:    {len(approx)}  (epsilon={epsilon:.2f}px)")

    # --- polygon 그리기 ---
    cv2.polylines(img, [approx], isClosed=True, color=color, thickness=2)

    # --- 클릭 점 표시 (노란 원, 모든 점) ---
    for px, py in pts:
        cv2.circle(img, (px, py), CLICK_DOT_RADIUS, CLICK_DOT_COLOR, -1)
    # ← 변경: 단일 circle → pts 순회. person은 첫점+중심 2개가 모두 표시됨

    # --- 클래스 이름 텍스트 표시 (첫 번째 점 기준) ---
    x0, y0 = pts[0]
    text_x = x0 + CLICK_DOT_RADIUS + 4
    text_y = y0 + 6
    # ← 변경: x, y 대신 pts[0] 사용. 점이 여러 개일 때도 텍스트 위치가 첫 점에 고정됨
    cv2.putText(
        img,
        f"{class_id}:{label_name}",   # ← 추가: "0:person" / "1:dog" 형식으로 class_id와 이름 함께 표시
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,                          # ← 추가: 폰트 크기 0.8 — 이미지가 크므로 너무 작으면 안 보임
        color,                        # ← 추가: 텍스트 색상도 polygon과 동일하게 맞춰 어느 객체 라벨인지 직관적
        2,                            # ← 추가: 선 두께 2 — 배경과 대비가 충분히 나도록
        cv2.LINE_AA,                  # ← 추가: 안티앨리어싱으로 텍스트 가장자리를 부드럽게
    )


# ===== [6. 결과 저장] =====
print(f"\n[Step 5/5] 결과 이미지 저장...")
cv2.imwrite(OUTPUT_PATH, img)
print(f"           저장 완료: {OUTPUT_PATH}")
print(f"\n완료. baseline_result.jpg와 sam3_click_result.jpg를 비교해보세요.")
