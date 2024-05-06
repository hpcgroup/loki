#!/bin/bash

set -x
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 64 0.5
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 64 0.25
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 64 0.125

sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 32 0.5
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 32 0.25
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 32 0.125

sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 16 0.5
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 16 0.25
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 16 0.125

sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 64 0.5 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 64 0.25 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 64 0.125 --lm-harness-eval

sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 32 0.5 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 32 0.25 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 32 0.125 --lm-harness-eval

sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 16 0.5 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 16 0.25 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 16 0.125 --lm-harness-eval
set +x
