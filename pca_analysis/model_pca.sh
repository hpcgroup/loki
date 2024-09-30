#!/bin/bash

MODEL=$1
NUM_LAYERS=$2


set -x
sbatch submit_pca.sh key ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/wikitext-valid/postrotary ${MODEL}-PCA/wikitext/postrotary
sbatch submit_pca.sh key ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/wikitext-valid/prerotary ${MODEL}-PCA/wikitext/prerotary
sbatch submit_pca.sh key ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/c4/postrotary ${MODEL}-PCA/c4/postrotary
sbatch submit_pca.sh key ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/c4/prerotary ${MODEL}-PCA/c4/prerotary
sbatch submit_pca.sh key ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/bookcorpus/postrotary ${MODEL}-PCA/bookcorpus/postrotary
sbatch submit_pca.sh key ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/bookcorpus/prerotary ${MODEL}-PCA/bookcorpus/prerotary

sbatch submit_pca.sh query ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/wikitext-valid/postrotary ${MODEL}-PCA/wikitext/postrotary
sbatch submit_pca.sh query ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/wikitext-valid/prerotary ${MODEL}-PCA/wikitext/prerotary
sbatch submit_pca.sh query ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/c4/postrotary ${MODEL}-PCA/c4/postrotary
sbatch submit_pca.sh query ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/c4/prerotary ${MODEL}-PCA/c4/prerotary
sbatch submit_pca.sh query ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/bookcorpus/postrotary ${MODEL}-PCA/bookcorpus/postrotary
sbatch submit_pca.sh query ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/bookcorpus/prerotary ${MODEL}-PCA/bookcorpus/prerotary

sbatch submit_pca.sh value ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/wikitext-valid/prerotary ${MODEL}-PCA/wikitext/prerotary
sbatch submit_pca.sh value ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/c4/prerotary ${MODEL}-PCA/c4/prerotary
sbatch submit_pca.sh value ${NUM_LAYERS} ${CFS}/m4641/ApproxAttn/raw_tensors/${MODEL}/1/bookcorpus/prerotary ${MODEL}-PCA/bookcorpus/prerotary
set +x
