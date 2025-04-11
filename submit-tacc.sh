#!/bin/bash
#SBATCH -J thresh_lsh_eval
#SBATCH -o outfiles/%x-%j.out
#SBATCH -e outfiles/%x-%j.err
#SBATCH -p gh       
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=nkoley@umd.edu

# module load python3/3.9.7
source ../loki-venv/bin/activate

# Scratch and cache paths
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"
export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_CONFIG_DIR="$SCRATCH/.cache/wandb_config"

# Distributed env
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1
export JOBID=${SLURM_JOB_ID}
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}

srun python -u evaluate_tasks.py \
    --sequence-length 4096 \
    --model-id meta-llama/Llama-2-7b-hf \
    --model-type llama \
    --dataset wikitext-valid \
    --use-thresh \
    --no-json \
    --lm-harness-eval

echo "Finished at: $(date)"
