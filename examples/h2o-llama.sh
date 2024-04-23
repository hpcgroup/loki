#!/bin/bash

set -x
sbatch examples/submit_h2o.sh meta-llama/Llama-2-7b-hf llama 4096 0.0625
sbatch examples/submit_h2o.sh meta-llama/Llama-2-7b-hf llama 4096 0.125
sbatch examples/submit_h2o.sh meta-llama/Llama-2-7b-hf llama 4096 0.25
sbatch examples/submit_h2o.sh meta-llama/Llama-2-13b-hf llama 4096 0.0625
#sbatch examples/submit_h2o.sh meta-llama/Llama-2-13b-hf llama 4096 0.125
#sbatch examples/submit_h2o.sh meta-llama/Llama-2-13b-hf llama 4096 0.25
sbatch examples/submit_h2o.sh meta-llama/Llama-2-70b-hf llama 4096 0.0625
#sbatch examples/submit_h2o.sh meta-llama/Llama-2-70b-hf llama 4096 0.125
#sbatch examples/submit_h2o.sh meta-llama/Llama-2-70b-hf llama 4096 0.25
set +x
