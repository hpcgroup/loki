#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --account=m4641_g
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH -J pca_topk
#SBATCH --output=outfiles/%x-%j.out

module load conda
conda activate loki
module load pytorch/2.1.0-cu12
module load cudatoolkit/12.2
conda activate loki
huggingface-cli login --token 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

export CUDA_DEVICE_MAX_CONNECTIONS=1

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET="AWS Libfabric"
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_OFLOW_BUF_SIZE=1073741824
export FI_CXI_OFLOW_BUF_COUNT=1
export MPICH_GPU_SUPPORT_ENABLED=0 

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"

export WANDB_DIR="$SCRATCH/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_CONFIG_DIR="$SCRATCH/.cache/wandb_config"

srun --ntasks=1 --gpus=4 --nodes=1 ./set_env_vars_slurm.sh python -u evaluate_tasks.py --sequence-length 4096 --model-id mlabonne/Meta-Llama-3-8B --model-type llama --use-pca-topk --top-r 32 --top-k 0.25 --rotary-type postrotary --temp 0.5 --dataset wikitext-test --transform-dataset wikitext --use-axonn --use-wandb