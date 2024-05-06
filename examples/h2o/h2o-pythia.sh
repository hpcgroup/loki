#!/bin/bash

set -x
sbatch examples/h2o/submit_h2o_1gpu.sh EleutherAI/pythia-6.9b gptneox 2048 0.125
sbatch examples/h2o/submit_h2o_1gpu.sh EleutherAI/pythia-6.9b gptneox 2048 0.25
sbatch examples/h2o/submit_h2o_1gpu.sh EleutherAI/pythia-6.9b gptneox 2048 0.125 --lm-harness-eval
sbatch examples/h2o/submit_h2o_1gpu.sh EleutherAI/pythia-6.9b gptneox 2048 0.25 --lm-harness-eval
set +x
