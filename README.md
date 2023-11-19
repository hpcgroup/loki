# approximate-attention

## Installation
You need to install the [llm-perplexity-eval](https://github.com/axonn-ai/llm-perplexity-eval) package first.
Follow the instructions in their readme.

## Usage

Say you want to only consider the top 128 tokens for opt-350m.

```
python eval_top_k.py --top-k 128 --model-id facebook/opt-350m
```

To get the original perplexity you can set `--top-k -1`. This will bypass the top-k changes


### With AxoNN's tensor parallelism

Additionally, you can add `--use-axonn` flag to shard a large model like opt-13b over multiple GPUs.
For this you will need to launch the code using mpirun


```
mpirun -np 2 python eval_top_k.py --top-k 128 --model-id facebook/opt-350m

```


