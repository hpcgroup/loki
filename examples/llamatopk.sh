#!/bin/bash

set -x
srun -n 2 python -u eval_top_k.py --top-k 512 --model-id huggyllama/llama-13b --model-type llama --use-axonn | tee out512.txt
srun -n 2 python -u eval_top_k.py --top-k 256 --model-id huggyllama/llama-13b --model-type llama --use-axonn | tee out256.txt
srun -n 2 python -u eval_top_k.py --top-k 128 --model-id huggyllama/llama-13b --model-type llama --use-axonn | tee out128.txt
srun -n 2 python -u eval_top_k.py --top-k 64 --model-id huggyllama/llama-13b --model-type llama --use-axonn | tee out64.txt
srun -n 2 python -u eval_top_k.py --top-k 32 --model-id huggyllama/llama-13b --model-type llama --use-axonn | tee out32.txt
set +x
