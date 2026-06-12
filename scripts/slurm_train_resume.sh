#!/bin/bash
# ===================================================
# YOLO11s-seg 학습 재개 Slurm 배치 스크립트
# 사용법: sbatch slurm_train_resume.sh
# ===================================================

#SBATCH --job-name=yolo_train_v1
# ← 작업 이름. squeue 명령어로 조회할 때 표시됨

#SBATCH --partition=debug
# ← 이 서버의 파티션 이름 (sinfo로 확인한 값)

#SBATCH --gres=gpu:A6000:2
# ← A6000 GPU 2개 요청. gpu-107 (Blackwell/p6000, sm_120)은 PyTorch 2.5.1과 비호환이므로 A6000 명시

#SBATCH --time=04:00:00
# ← 최대 실행 시간 (4시간). 남은 40 epoch × 약 4.5분 = 약 180분 기준 여유 있게 설정

#SBATCH --output=/home1/Jwson08/autolabel/slurm_%j.log
# ← %j = 배치 작업 ID. 로그가 slurm_12345.log 형태로 저장됨
# ← stdout과 stderr 둘 다 이 파일에 합쳐서 저장

#SBATCH --ntasks=1
# ← 단일 태스크. DDP는 Python 내부에서 torch.distributed로 처리하므로 1로 설정

#SBATCH --cpus-per-task=8
# ← 데이터 로딩 워커(workers=4) × GPU 수(2) = 8 코어 요청

# ===== [환경 설정] =====
source /home1/Jwson08/anaconda3/etc/profile.d/conda.sh
# ← conda 명령어를 쉘에서 사용 가능하게 초기화

conda activate autolabel
# ← ultralytics, torch 등이 설치된 가상환경 활성화

cd /home1/Jwson08/autolabel
# ← 작업 디렉토리 이동. runs/segment/... 상대경로가 여기서부터 시작됨

echo "========================================"
echo "작업 시작: $(date)"
echo "노드: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ===== [학습 재개] =====
python -c "
from ultralytics import YOLO
model = YOLO('runs/segment/runs/segment/train_v1/weights/last.pt')
# last.pt: 마지막으로 저장된 체크포인트 (현재 epoch 58)
# resume=True: epoch/optimizer 상태까지 이어받아 epoch 59부터 재개
model.train(resume=True)
"

echo "========================================"
echo "작업 완료: $(date)"
echo "========================================"
