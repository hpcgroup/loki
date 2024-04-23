#!/bin/bash

set -x
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 2048
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 1024
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 512
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 32 2048
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 32 1024
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 32 512
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 16 2048
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 16 1024
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 16 512
set +x
