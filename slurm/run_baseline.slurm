#!/bin/bash
#SBATCH --job-name=baseline_run
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=hpg-ai           
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=ic3                
#SBATCH --qos=ic3                    

# ---------------------------------------------------------------------
# 1. Environment Setup
# ---------------------------------------------------------------------
echo "Job started on $(hostname) at $(date)"

module load conda
conda activate ic3_env

# ---------------------------------------------------------------------
# 2. Dynamic Path Routing
# ---------------------------------------------------------------------
PREDICTION_OUTPUT="predictions/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.jsonl"

# ---------------------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------------------
echo "Starting inference run: $SLURM_JOB_NAME"
echo "Output will be saved to: $PREDICTION_OUTPUT"

python src/inference.py \
    --data_path data/starter_val.jsonl \
    --output_path $PREDICTION_OUTPUT \
    --batch_size 8

echo "Job finished successfully at $(date)"