#!/bin/bash

set -x
#sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 32 0.5
#sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 32 0.25
#sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 32 0.125
#
#sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 16 0.5
#sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 16 0.25
#sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 16 0.125
#
#sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 8 0.5
#sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 8 0.25
#sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 8 0.125

sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 32 0.5 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 32 0.25 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 32 0.125 --lm-harness-eval

sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 16 0.5 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 16 0.25 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 16 0.125 --lm-harness-eval

sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 8 0.5 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 8 0.25 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 8 0.125 --lm-harness-eval
set +x
