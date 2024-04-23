#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm80g
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --account=m4641_g
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:50:00


# Runs a "10B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1


NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 1 ))
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

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"

MODEL=$1
MODEL_TYPE=$2
SEQ_LEN=$3
MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
TOPR=$4
EVAL=$5

OUT_FILE_PATH="experiments/exp-pca/${MODEL_NAME}"
mkdir -p $OUT_FILE_PATH

echo "Model: ${MODEL}"
echo "Model Name: ${MODEL_NAME}"
echo "Sequence Length: ${SEQ_LEN}"
echo "Output Path: ${OUT_FILE_PATH}"
echo "Running model ${MODEL} with PCA Attention and top-r ${TOPR}"

run_cmd="srun -C gpu -N ${NNODES} -n ${GPUS} -c 32 --cpu-bind=cores --gpus-per-node=4 python -u eval_ppl.py --sequence-length ${SEQ_LEN}\
        --model-id ${MODEL} --model-type ${MODEL_TYPE}\
        --use-pca --top-r ${TOPR} ${EVAL}| tee ${OUT_FILE_PATH}/out_${MODEL_NAME}_${TOPR}${EVAL}.out 2>&1"


echo ${run_cmd}
eval ${run_cmd}
