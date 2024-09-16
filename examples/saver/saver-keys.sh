#!/bin/bash
MODEL_ID=$1
MODEL_TYPE=$2
SEQ_LEN=$3
TIME=$4

echo "MODEL_ID: $MODEL_ID"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "SEQ_LEN: $SEQ_LEN"

set -x
sbatch examples/saver/submit_saver.sh ${MODEL_ID} ${MODEL_TYPE} ${SEQ_LEN} 1 wikitext-valid
sbatch examples/saver/submit_saver.sh ${MODEL_ID} ${MODEL_TYPE} ${SEQ_LEN} 1 bookcorpus
sbatch examples/saver/submit_saver.sh ${MODEL_ID} ${MODEL_TYPE} ${SEQ_LEN} 1 c4
set +x
