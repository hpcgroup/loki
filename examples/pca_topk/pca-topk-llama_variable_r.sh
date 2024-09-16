#!/bin/bash

MODEL=$1
MODEL_TYPE=$2
SEQLEN=$3
TDATA=$4

set -x

for k in 0.5 0.25 0.125
do 
    for r in postrotary prerotary
    do
        for d in 0.50 0.60 0.70 0.80
        do
            sbatch examples/pca_topk/submit_pca_topk.sh ${MODEL} ${MODEL_TYPE} ${SEQLEN} ${d} ${k} ${r} --lm-harness-eval "${TDATA}"
        done
    done
done

set +x
