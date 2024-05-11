#!/bin/bash

set -x
sbatch examples/topk/submit_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 1 wikitext-test
sbatch examples/topk/submit_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 0.25 wikitext-test
sbatch examples/topk/submit_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 0.50 wikitext-test

sbatch examples/topk/submit_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 1 wikitext-test --lm-harness-eval
sbatch examples/topk/submit_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 0.25 wikitext-test --lm-harness-eval
sbatch examples/topk/submit_topk.sh mistralai/Mistral-7B-v0.1 mistral 4096 0.50 wikitext-test --lm-harness-eval
set +x
