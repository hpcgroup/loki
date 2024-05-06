#!/bin/bash

set -x
#sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 128 --lm-harness-eval
#sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 64 --lm-harness-eval
#sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 32 --lm-harness-eval
#sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 128
#sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 64
#sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 32
sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 96
sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 0.97 --lm-harness-eval
sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 0.95 --lm-harness-eval
sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 0.90 --lm-harness-eval
sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 0.85 --lm-harness-eval
sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 0.75 --lm-harness-eval
#sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 0.97
#sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 0.95
#sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 0.90
#sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 0.85
#sbatch examples/pca/submit_pca.sh meta-llama/Llama-2-7b-hf llama 4096 0.75
set +x
