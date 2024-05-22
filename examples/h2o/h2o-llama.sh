#!/bin/bash

MODEL=$1
MODEL_TYPE=$2
SEQLEN=$3

set -x
#sbatch examples/h2o/submit_h2o_1gpu.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.25
#sbatch examples/h2o/submit_h2o_1gpu.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.125
#sbatch examples/h2o/submit_h2o_1gpu.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.0625
#sbatch examples/h2o/submit_h2o_1gpu_lmeval.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.25 --lm-harness-eval
#sbatch examples/h2o/submit_h2o_1gpu_lmeval.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.125 --lm-harness-eval
#sbatch examples/h2o/submit_h2o_1gpu_lmeval.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.0625 --lm-harness-eval
sbatch examples/h2o/submit_h2o.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.25
sbatch examples/h2o/submit_h2o.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.125
sbatch examples/h2o/submit_h2o.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.0625
sbatch examples/h2o/submit_h2o.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.25 --lm-harness-eval
sbatch examples/h2o/submit_h2o.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.125 --lm-harness-eval
sbatch examples/h2o/submit_h2o.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} 0.0625 --lm-harness-eval
set +x
