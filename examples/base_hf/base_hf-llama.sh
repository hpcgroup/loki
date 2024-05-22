#!/bin/bash

#MODEL=$1
#MODEL_TYPE=$2
#SEQLEN=$3

set -x
#sbatch examples/base_hf/submit_base_hf.sh meta-llama/Llama-2-7b-hf llama 4096
#sbatch examples/base_hf/submit_base_hf.sh meta-llama/Llama-2-7b-hf llama 4096 --lm-harness-eval
#
#sbatch examples/base_hf/submit_base_hf.sh meta-llama/Llama-2-13b-hf llama 4096
#sbatch examples/base_hf/submit_base_hf.sh meta-llama/Llama-2-13b-hf llama 4096 --lm-harness-eval
#

#sbatch examples/base_hf/submit_base_hf.sh meta-llama/Llama-2-70b-hf llama 4096
sbatch examples/base_hf/submit_base_hf.sh meta-llama/Llama-2-70b-hf llama 4096 --lm-harness-eval
#
#sbatch examples/base_hf/submit_base_hf.sh meta-llama/Meta-Llama-3-8B llama 8192
#sbatch examples/base_hf/submit_base_hf.sh meta-llama/Meta-Llama-3-8B llama 8192 --lm-harness-eval
#

#sbatch examples/base_hf/submit_base_hf.sh meta-llama/Meta-Llama-3-70B llama 8192
sbatch examples/base_hf/submit_base_hf.sh meta-llama/Meta-Llama-3-70B llama 8192 --lm-harness-eval
#
#sbatch examples/base_hf/submit_base_hf.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 
#sbatch examples/base_hf/submit_base_hf.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 llama 2048 --lm-harness-eval

#sbatch examples/base_hf/submit_base_hf.sh mistralai/Mistral-7B-v0.1 mistral 4096 
#sbatch examples/base_hf/submit_base_hf.sh  mistralai/Mistral-7B-v0.1 mistral 4096 --lm-harness-eval
#

#sbatch examples/base_hf/submit_base_hf.sh mistralai/Mixtral-8x7B-v0.1 mistral 4096 
sbatch examples/base_hf/submit_base_hf.sh mistralai/Mixtral-8x7B-v0.1 mistral 4096 --lm-harness-eval

#sbatch examples/base_hf/submit_base_hf.sh EleutherAI/pythia-6.9b gptneox 2048
#sbatch examples/base_hf/submit_base_hf.sh EleutherAI/pythia-6.9b gptneox 2048 --lm-harness-eval

set +x
