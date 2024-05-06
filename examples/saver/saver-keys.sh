#!/bin/bash
MODEL_ID=$1
MODEL_TYPE=$2

echo "MODEL_ID: $MODEL_ID"
echo "MODEL_TYPE: $MODEL_TYPE"

set -x
bash examples/saver/submit_saver.sh ${MODEL_ID} ${MODEL_TYPE} 2048 1 wikitext-valid
#sbatch examples/saver/submit_saver.sh ${MODEL_ID} ${MODEL_TYPE} 2048 1 bookcorpus
#sbatch examples/saver/submit_saver.sh ${MODEL_ID} ${MODEL_TYPE} 2048 1 c4
set +x
