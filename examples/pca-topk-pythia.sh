#!/bin/bash

set -x
#sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 64 2048
#sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 64 1024
#sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 64 512
#sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 32 2048
#sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 32 1024
#sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 32 512
#sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 16 2048
#sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 16 1024
#sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 16 512
#sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 32 512
#sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 16 512


sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 64 0.5
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 64 0.25
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 64 0.125

sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 32 0.5
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 32 0.25
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 32 0.125

sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 16 0.5
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 16 0.25
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 16 0.125

sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 64 0.5 --lm-harness-eval
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 64 0.25 --lm-harness-eval
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 64 0.125 --lm-harness-eval

sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 32 0.5 --lm-harness-eval
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 32 0.25 --lm-harness-eval
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 32 0.125 --lm-harness-eval

sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 16 0.5 --lm-harness-eval
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 16 0.25 --lm-harness-eval
sbatch examples/submit_pca_topk.sh EleutherAI/pythia-6.9b gptneox 2048 16 0.125 --lm-harness-eval
set +x
