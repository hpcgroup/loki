#!/bin/bash

set -x
#sbatch examples/submit_h2o.sh EleutherAI/pythia-6.9b gptneox 2048 0.0625
#sbatch examples/submit_h2o.sh meta-llama/Llama-2-13b-hf llama 4096 0.0625
#sbatch examples/submit_h2o.sh meta-llama/Llama-2-13b-hf llama 4096 0.125
#sbatch examples/submit_h2o.sh meta-llama/Llama-2-13b-hf llama 4096 0.25
#sbatch examples/submit_h2o.sh meta-llama/Llama-2-70b-hf llama 4096 0.0625
#sbatch examples/submit_h2o.sh meta-llama/Llama-2-70b-hf llama 4096 0.125
#sbatch examples/submit_h2o.sh meta-llama/Llama-2-70b-hf llama 4096 0.25
sbatch examples/submit_h2o_1gpu.sh EleutherAI/pythia-6.9b gptneox 2048 0.125
sbatch examples/submit_h2o_1gpu.sh EleutherAI/pythia-6.9b gptneox 2048 0.25
sbatch examples/submit_h2o_1gpu.sh EleutherAI/pythia-6.9b gptneox 2048 0.125 --lm-harness-eval
sbatch examples/submit_h2o_1gpu.sh EleutherAI/pythia-6.9b gptneox 2048 0.25 --lm-harness-eval
set +x
