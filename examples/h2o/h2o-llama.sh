#!/bin/bash

set -x
sbatch examples/h2o/submit_h2o.sh meta-llama/Llama-2-7b-hf llama 4096 0.125
sbatch examples/h2o/submit_h2o.sh meta-llama/Llama-2-7b-hf llama 4096 0.25
sbatch examples/h2o/submit_h2o.sh meta-llama/Llama-2-7b-hf llama 4096 0.125 --lm-harness-eval
sbatch examples/h2o/submit_h2o.sh meta-llama/Llama-2-7b-hf llama 4096 0.25 --lm-harness-eval
set +x
