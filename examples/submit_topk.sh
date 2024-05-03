#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm80g
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --account=m4641_g
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH -J topk
#SBATCH --output=outfiles/%x-%j.out


# Runs a "10B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1


#NNODES=$SLURM_JOB_NUM_NODES
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
export WORLD_SIZE=$GPUS


export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"

MODEL=$1
MODEL_TYPE=$2
SEQ_LEN=$3
MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
TOPK=$4
DATASET=$5
EVAL=$6
SAVE=false

OUT_FILE_PATH="experiments/exp-topk/${MODEL_NAME}"
mkdir -p $OUT_FILE_PATH


echo "Model: ${MODEL}"
echo "Model Name: ${MODEL_NAME}"
echo "Sequence Length: ${SEQ_LEN}"
echo "Output Path: ${OUT_FILE_PATH}"
echo "Running model ${MODEL} with top-k ${TOPK}"

SAVE_ARGS=""
if [ "$SAVE" = true ]; then
  OUT_TENSOR_DATA_PATH="${SCRATCH}/InferenceData/topk/${MODEL_NAME}/${TOPK}/${DATASET}/"
  mkdir -p $OUT_TENSOR_DATA_PATH
  SAVE_ARGS="--save-tensors --tensors-dir ${OUT_TENSOR_DATA_PATH}"
fi

run_cmd="srun -C gpu -N ${NNODES} -n ${GPUS} -c 32 --cpu-bind=cores --gpus-per-node=4 ./set_env_vars_slurm.sh python -u eval_ppl.py --use-axonn --sequence-length ${SEQ_LEN}\
        --model-id ${MODEL} --model-type ${MODEL_TYPE} --dataset ${DATASET}\
        ${SAVE_ARGS}\
        --dataset ${DATASET}
        --use-axonn --use-topk --top-k ${TOPK} ${EVAL}| tee ${OUT_FILE_PATH}/out_${MODEL_NAME}_${TOPK}${EVAL}.out 2>&1"

echo ${run_cmd}
eval ${run_cmd}
