#!/bin/bash

MODEL=$1
MODEL_TYPE=$2
SEQLEN=$3

set -x
sbatch examples/topk/submit_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.5
sbatch examples/topk/submit_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.25
sbatch examples/topk/submit_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.125


sbatch examples/topk/submit_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.5 --lm-harness-eval
sbatch examples/topk/submit_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.25 --lm-harness-eval
sbatch examples/topk/submit_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.125 --lm-harness-eval
set +x
