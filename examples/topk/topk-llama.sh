#!/bin/bash

set -x
#sbatch examples/topk/submit_topk.sh meta-llama/Llama-2-7b-hf llama 4096 0.125
#sbatch examples/topk/submit_h2o.sh meta-llama/Llama-2-7b-hf llama 4096 0.25
#sbatch examples/topk/submit_h2o.sh meta-llama/Llama-2-13b-hf llama 4096 0.125
#sbatch examples/topk/submit_h2o.sh meta-llama/Llama-2-13b-hf llama 4096 0.25
#sbatch examples/topk/submit_h2o.sh meta-llama/Llama-2-70b-hf llama 4096 0.125
sbatch examples/topk/submit_topk.sh meta-llama/Llama-2-70b-hf llama 4096 1024
sbatch examples/topk/submit_topk.sh meta-llama/Llama-2-70b-hf llama 4096 4096
set +x
