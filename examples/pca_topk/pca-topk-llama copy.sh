#!/bin/bash

MODEL=$1
MODEL_TYPE=$2
SEQLEN=$3
ROTARY=$4
TDATA=$5

set -x
#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 64 0.5
sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 64 0.25 ${ROTARY} "${TDATA}"
#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 64 0.125

#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 32 0.5
sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 32 0.25 ${ROTARY} "${TDATA}"
#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 32 0.125

#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 16 0.5
sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 16 0.25 ${ROTARY} "${TDATA}"
#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 16 0.125

#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 64 0.5 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 64 0.25 ${ROTARY} --lm-harness-eval "${TDATA}"
#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 64 0.125 --lm-harness-eval

#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 32 0.5 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 32 0.25 ${ROTARY} --lm-harness-eval "${TDATA}"
#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 32 0.125 --lm-harness-eval

#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 16 0.5 --lm-harness-eval
sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 16 0.25 ${ROTARY} --lm-harness-eval "${TDATA}"
#sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 16 0.125 --lm-harness-eval
set +x