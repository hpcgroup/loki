#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=gpu&hbm80g
#SBATCH -N {nodes}
#SBATCH --gpus-per-node={gpus}
#SBATCH --account=m4641_g
#SBATCH --ntasks-per-node={gpus}
#SBATCH --time={time}
#SBATCH -J pca_topk
#SBATCH --output={output_file}
#SBATCH --error={output_file}

export CUDA_DEVICE_MAX_CONNECTIONS=1

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * {gpus} ))
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

export WANDB_DIR="$SCRATCH/InferenceData/wandb"
export WANDB_CACHE_DIR="$SCRATCH/.cache/wandb"
export WANDB_CONFIG_DIR="$SCRATCH/.cache/wandb_config"

MODEL={model_path}
MODEL_TYPE={model_type}
SEQ_LEN={seqlen}
MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
TOPR={top_r}
TOPK={top_k}
ROTARY={rotary_type}
AXONN_ARGS={axonn_args}
EVAL_ARGS={eval_args}
WANDB_ARGS={wandb_args}
EVAL_DATASET={tdataset}
TRANSFORM_DATASET={transform_dataset}


TRANSFORM_DATASET_ARGS=""
if [ "$TRANSFORM_DATASET" != "" ]; then
    TRANSFORM_DATASET_ARGS="--transform-dataset $TRANSFORM_DATASET"
fi

echo "Model: $MODEL"
echo "Model Name: $MODEL_NAME"
echo "Sequence Length: $SEQ_LEN"
echo "Running model $MODEL with PCA Attention and top-r $TOPR and top-k $TOPK"

run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 ./set_env_vars_slurm.sh\
        python -u evaluate_tasks.py --sequence-length $SEQ_LEN --model-id $MODEL --model-type $MODEL_TYPE\
        --use-pca-topk --top-r $TOPR --top-k $TOPK --dataset $EVAL_DATASET --rotary-type $ROTARY\
        $TRANSFORM_DATASET_ARGS $WANDB_ARGS $AXONN_ARGS $EVAL_ARGS"


      
echo $run_cmd
eval $run_cmd
