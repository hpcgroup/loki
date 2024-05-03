#!/bin/bash

set -x
#sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 2048
#sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 1024
#sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 512
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 0.5
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 0.25
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 0.125

sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 32 0.5
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 32 0.25
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 32 0.125
#sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 32 512
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 16 0.5
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 16 0.25
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 16 0.125
#sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 16 512

sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 0.5 --lm-harness-eval
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 0.25 --lm-harness-eval
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 64 0.125 --lm-harness-eval

sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 32 0.5 --lm-harness-eval
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 32 0.25 --lm-harness-eval
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 32 0.125 --lm-harness-eval
#sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 32 512
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 16 0.5 --lm-harness-eval
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 16 0.25 --lm-harness-eval
sbatch examples/submit_pca_topk.sh meta-llama/Llama-2-7b-hf llama 4096 16 0.125 --lm-harness-eval
set +x
