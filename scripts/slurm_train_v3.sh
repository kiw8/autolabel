#!/bin/bash
#SBATCH --job-name=train_v3
#SBATCH --nodelist=gpu-106
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=logs/train_v3_%j.log
#SBATCH --error=logs/train_v3_%j.log

echo "=============================="
echo "JOB ID    : $SLURM_JOB_ID"
echo "NODE      : $SLURMD_NODENAME"
echo "GPU       : $CUDA_VISIBLE_DEVICES"
echo "START     : $(date)"
echo "=============================="

# conda 환경 활성화
source /home1/Jwson08/anaconda3/etc/profile.d/conda.sh
conda activate autolabel

cd /home1/Jwson08/autolabel

# V3 학습 실행
python scripts/train_v3.py

echo ""
echo "END: $(date)"
