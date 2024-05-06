#!/bin/bash

set -x
sbatch examples/topk/submit_topk.sh EleutherAI/pythia-6.9b gptneox 2048 1 wikitext-test
sbatch examples/topk/submit_topk.sh EleutherAI/pythia-6.9b gptneox 2048 0.25 wikitext-test
sbatch examples/topk/submit_topk.sh EleutherAI/pythia-6.9b gptneox 2048 0.50 wikitext-test

sbatch examples/topk/submit_topk.sh EleutherAI/pythia-6.9b gptneox 2048 1 wikitext-test --lm-harness-eval
sbatch examples/topk/submit_topk.sh EleutherAI/pythia-6.9b gptneox 2048 0.25 wikitext-test --lm-harness-eval
sbatch examples/topk/submit_topk.sh EleutherAI/pythia-6.9b gptneox 2048 0.50 wikitext-test --lm-harness-eval
set +x
