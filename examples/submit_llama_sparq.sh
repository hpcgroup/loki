#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --account=m2404_g
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00


# Runs a "10B" parameter model

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

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"

MODEL="meta-llama/Llama-2-13b-hf"
MODEL_NAME="llama2_13b"

for TOPR in 16 32 64
do
	for TOPK in 128
	do
		run_cmd="srun -C gpu -N ${NNODES} -n ${GPUS} -c 32 --cpu-bind=cores --gpus-per-node=4 python -u eval_top_k.py --sequence-length 4096 --top-k ${TOPK} --top-r ${TOPR} --model-id ${MODEL} --model-type llama --use-axonn --use-spar --use-query | tee exp-spar/out_Q_${MODEL_NAME}_${TOPR}_${TOPK}.out"
		echo ${run_cmd}
		eval ${run_cmd}
	done
done
