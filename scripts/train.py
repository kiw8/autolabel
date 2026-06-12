"""
YOLO11s-seg 파인튜닝 스크립트 (트랙 #6)
=========================================
목적: 사전학습된 YOLO11s-seg를 엘리베이터 person/dog 데이터셋으로 파인튜닝한다.
      이번 실행은 10 epoch 짧은 검증 학습 — 파이프라인 정상 작동 확인용.
입력: /home1/Jwson08/autolabel/dataset/data.yaml
출력: runs/segment/train_check/weights/best.pt
"""

# ===== [import] =====
import sys
from pathlib import Path
from ultralytics import YOLO
# ← ultralytics: YOLO 모델 로드·학습·평가를 한 인터페이스로 제공하는 패키지


# ===== [설정] =====
DATA_YAML  = Path("/home1/Jwson08/autolabel/dataset/data.yaml")
MODEL_NAME = "yolo11s-seg.pt"
# ← yolo11s-seg.pt: nano(n)보다 파라미터가 많아 정확도 높음.
#   로컬에 없으면 ultralytics가 자동으로 공식 서버에서 다운로드

EPOCHS   = 100
# ← 검증(10) → 본 학습(100). 10 epoch에서 mAP50=0.87로 아직 상승 추세였으므로 충분히 늘림

IMGSZ    = 640
# ← YOLO 표준 입력 해상도. 원본 1920×1080을 640×640으로 리사이즈해서 학습

BATCH    = 32
# ← 단일 GPU 16 → 2 GPU 병렬 시 32로 증가. GPU 수만큼 배치를 늘려야 처리 효율이 높아짐

DEVICE   = [0, 1]
# ← 검증(단일 GPU 0) → 본 학습([0,1]). CUDA_VISIBLE_DEVICES=6,7 환경에서
#   0=물리 GPU 6번, 1=물리 GPU 7번을 동시에 사용하는 DDP(분산 학습) 모드

PROJECT  = "runs/segment"
# ← 학습 결과 저장 루트 디렉토리. ultralytics가 자동 생성

NAME     = "train_v1"
# ← 검증(train_check) → 본 학습(train_v1). 결과는 runs/segment/train_v1/ 아래 저장됨


# ===== [Step 1. data.yaml 확인] =====
print("=" * 55)
print("[Step 1] data.yaml 경로 확인")
if not DATA_YAML.exists():
    print(f"  ❌ 파일 없음: {DATA_YAML}")
    print("     split_dataset.py를 먼저 실행하세요.")
    sys.exit(1)
    # ← data.yaml이 없으면 학습 자체가 불가능하므로 명확한 에러 메시지 후 즉시 종료
print(f"  ✅ {DATA_YAML}")


# ===== [Step 2. 모델 로드] =====
print(f"\n[Step 2] 모델 로드: {MODEL_NAME}")
model = YOLO(MODEL_NAME)
# ← 사전학습 가중치를 로드. COCO 80클래스로 학습된 가중치를 우리 2클래스에 맞게 파인튜닝할 예정
# ← 파일이 없으면 ultralytics가 자동 다운로드 (인터넷 연결 필요, 약 22MB)
print(f"  모델 태스크: {model.task}")
# ← task가 'segment'여야 instance segmentation 학습 진행


# ===== [Step 3. 학습] =====
print(f"\n[Step 3] 학습 시작")
print(f"  epochs={EPOCHS}, imgsz={IMGSZ}, batch={BATCH}, device=cuda:{DEVICE}")
print(f"  결과 저장: {PROJECT}/{NAME}/\n")

results = model.train(
    data    = str(DATA_YAML),   # ← str()로 변환: ultralytics가 문자열 경로를 요구
    epochs  = EPOCHS,
    imgsz   = IMGSZ,
    batch   = BATCH,
    device  = DEVICE,
    project = PROJECT,
    name    = NAME,
    exist_ok= True,
    # ← exist_ok=True: 같은 이름 폴더가 있어도 덮어쓰기 허용.
    #   False면 train_check2, train_check3... 처럼 번호가 붙어 결과 위치가 매번 달라짐
    workers = 4,
    # ← 데이터 로딩 병렬 워커 수. 너무 크면 메모리 과부하, 4가 안정적인 기본값
    verbose = True,
    # ← 에포크별 loss/mAP를 콘솔에 출력. 학습 진행 상황을 실시간으로 확인하기 위해
)


# ===== [Step 4. 결과 출력] =====
print("\n" + "=" * 55)
print("[Step 4] 학습 완료 — 결과 요약")

# best.pt 경로
best_pt = Path(PROJECT) / NAME / "weights" / "best.pt"
if best_pt.exists():
    print(f"  best.pt : {best_pt.resolve()}")
else:
    print(f"  ⚠️  best.pt 를 찾을 수 없음: {best_pt}")
    # ← 10 epoch처럼 짧은 학습에서도 best.pt는 생성되어야 함. 없으면 학습 중 오류 가능성

# 최종 mAP 출력 (segmentation mask 기준)
metrics = results.results_dict
# ← results_dict: {'metrics/precision(M)': ..., 'metrics/mAP50(M)': ..., ...}
# ← (M)은 Mask, (B)는 Box. segmentation 품질은 (M) 기준으로 봐야 함

map50   = metrics.get("metrics/mAP50(M)",    "N/A")
map5095 = metrics.get("metrics/mAP50-95(M)", "N/A")

print(f"  mAP50   (mask): {map50:.4f}"   if isinstance(map50,   float) else f"  mAP50   : {map50}")
print(f"  mAP50-95(mask): {map5095:.4f}" if isinstance(map5095, float) else f"  mAP50-95: {map5095}")
# ← mAP50: IoU 0.5 기준 평균 정밀도. 10 epoch 검증 학습에서는 0.3~0.6 정도면 정상
# ← mAP50-95: IoU 0.5~0.95 기준. 더 엄격한 지표, 본 학습 성능 평가에 사용

print(f"\n검증 완료. 정상 작동 확인 시 epochs를 늘려 본 학습 진행.")
print("=" * 55)
