# approximate-attention

## Installation
You need to install the [llm-perplexity-eval](https://github.com/axonn-ai/llm-perplexity-eval) package first.
Follow the instructions in their readme.

## Usage

Say you want to find the perplexity for H2O on a llama-2-7b model

```
python -u evaluate_tasks.py --sequence-length 4096 --model-id meta-llama/Llama-2-7b-hf --model-type llama --use-h2o --heavy-ratio 0.1
```

### With AxoNN's tensor parallelism

Additionally, you can add `--use-axonn` flag to shard a large model like llama-13b over multiple GPUs.
For this you will need to launch the code using mpirun


```
mpirun -np 2 python -u evaluate_tasks.py --sequence-length 4096 --model-id meta-llama/Llama-2-13b-hf --model-type llama --use-h2o --heavy-ratio 0.1 --use-axonn
```


