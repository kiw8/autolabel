"""
추론 속도 측정 스크립트
========================
목적: 학습된 v1 모델의 추론 속도 측정
입력: models/yolo11s_seg_v1_20260601/best.pt, dataset/images/val
출력: 콘솔 (평균/최소/최대/표준편차/FPS)
"""

# ===== [import] =====
import time    # perf_counter — 고해상도 타이머, time.time()보다 정밀
import glob    # 이미지 경로 리스트 수집
import random  # 측정 이미지 무작위 샘플링
import os      # 경로 존재 여부 확인
from ultralytics import YOLO


# ===== [설정] =====
MODEL_PATH = "/home1/Jwson08/autolabel/models/yolo11s_seg_v1_20260601/best.pt"
IMAGE_DIR  = "/home1/Jwson08/autolabel/dataset/images/val"
NUM_WARMUP = 5    # GPU 캐시·CUDA 커널 안정화용, 측정에서 제외
NUM_RUNS   = 100  # 통계적으로 신뢰할 수 있는 샘플 수


# ===== [모델 로드] =====
model = YOLO(MODEL_PATH)

model_name = "/".join(MODEL_PATH.split("/")[-2:])  # 경로에서 마지막 2단계만 추출
print(f"모델: {model_name}")
print(f"디바이스: cuda:0")


# ===== [이미지 샘플링] =====
all_images = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
assert len(all_images) >= NUM_WARMUP + NUM_RUNS, (
    f"val 이미지 수({len(all_images)})가 필요한 수({NUM_WARMUP + NUM_RUNS})보다 적습니다"
)

random.seed(42)  # 재현성 확보 — 같은 시드면 항상 같은 이미지 선택
samples = random.sample(all_images, NUM_WARMUP + NUM_RUNS)

warmup_images = samples[:NUM_WARMUP]
measure_images = samples[NUM_WARMUP:]

print(f"측정: val {NUM_RUNS}장 (워밍업 {NUM_WARMUP}장 제외)")


# ===== [워밍업] =====
# GPU는 첫 추론 시 CUDA 커널 컴파일·캐시 적재로 시간이 튀므로 워밍업 필수
for img_path in warmup_images:
    model.predict(img_path, device=0, verbose=False)


# ===== [속도 측정] =====
elapsed_times = []

for img_path in measure_images:
    t_start = time.perf_counter()          # CPU 기준 고해상도 타이머 시작
    model.predict(img_path, device=0, verbose=False)
    t_end = time.perf_counter()            # 추론 완료 후 타이머 종료
    elapsed_times.append((t_end - t_start) * 1000)  # 초 → ms 변환


# ===== [통계 계산 & 출력] =====
avg_ms = sum(elapsed_times) / len(elapsed_times)
min_ms = min(elapsed_times)
max_ms = max(elapsed_times)
std_ms = (sum((t - avg_ms) ** 2 for t in elapsed_times) / len(elapsed_times)) ** 0.5
fps    = 1000 / avg_ms  # 평균 ms 기준 이론 FPS (배치 처리 아닌 단일 이미지 기준)

print("---")
print(f"평균: {avg_ms:.1f} ms")
print(f"최소: {min_ms:.1f} ms")
print(f"최대: {max_ms:.1f} ms")
print(f"표준편차: {std_ms:.1f} ms")
print("---")
print(f"FPS (이론치): {fps:.1f} fps")
