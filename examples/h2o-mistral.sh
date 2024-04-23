#!/bin/bash

set -x
sbatch examples/submit_h2o_1gpu.sh mistralai/Mistral-7B-v0.1 mistral 4096 0.125
sbatch examples/submit_h2o_1gpu.sh mistralai/Mistral-7B-v0.1 mistral 4096 0.25
sbatch examples/submit_h2o_1gpu.sh mistralai/Mistral-7B-v0.1 mistral 4096 0.125 --lm-harness-eval
sbatch examples/submit_h2o_1gpu.sh mistralai/Mistral-7B-v0.1 mistral 4096 0.25 --lm-harness-eval
set +x
