# Loki

This repository contains the code related to the experiments in the paper [Loki: Low-Rank Keys for Efficient Sparse Attention](https://arxiv.org/abs/2406.02542).
We provide the code to compute the PCA of the keys for various models, baseline method implementations and kernels for Loki used in the paper, along with scripts to evaluate the methods on perplexity evaluation and downstream tasks.

## Installation
You need to install the requirements as follows:

```
pip install -r requirements.txt
```

Note: The code requires specific versions of the huggingface transformers library present in the requirements.txt file. It will not work with other versions.

## Usage

#### Compute PCA of keys for a model
Say you want to compute the PCA transform for the keys of Llama-2-7b model. You can do this by following the steps below:

- Run perplexity evaluation on the model on a target dataset to save the keys, queries and values tensors.
  ```bash
  # The --use-axonn flag is optional and is used to shard the model over multiple GPUs using AxoNN

  python -u evaluate_tasks.py --sequence-length 4096 --model-id meta-llama/Llama-2-7b-hf --model-type llama --dataset wikitext-valid --save-tensors --tensor-dir <Directory to save tensors> --use-topk --top-k 1 [--use-axonn]
  ```
  List of possible datasets - wikitext-valid, bookcorpus, c4

- Compute the PCA of the generated keys: In the `pca_analysis` directory, run the following command:

  ```bash
  python pca.py key <NUM_LAYERS> <Path to saved key tensors> <Path to output the PCA transforms>
  ```
Verify that the PCA transform are saved in the output directory. Do not modify the subdirectory structure of the output directory as it is used by the downstream tasks evaluation code.

#### Running the ML evaluations
Once the PCA transform is computed, we can run the ML evaluations using Loki. The following command runs the evaluation on the downstream tasks using the PCA transform computed in the previous step:

```bash
python -u evaluate_tasks.py \
  --sequence-length 4096 \
  --model-id meta-llama/Llama-2-7b-hf \
  --model-type llama 
  --use-pca-topk --top-r <16/32/64> --top-k <0.125/0.25/0.5> \
  --rotary-type <prerotary/postrotary> \
  --dataset <Dataset to compute perplexity on, Default: wikitext-test>\
  --transform-dataset <Dataset used to compute PCA keys: wikitext/bookcorpus/c4, Default:wikitext>\
  [--lm-harness-eval]\ # Flag to evaluate the model on the LM Harness Tasks
  [--use-wandb]\ # Optional flag to log the results to wandb
  [--use-axonn] # Optional flag to shard the model over multiple GPUs using AxoNN
```


<!---
#### Reproducing the results
We have provided slurm scripts to evaluate the baseline methods and Loki on the downstream tasks. You can run the scripts as follows:

- Generating the keys for the models:
First, you need to modify the template saver script based on your machine slurm configuration. You also need to modify the OUT_TENSOR_DATA_PATH in the script to save the keys to the desired location. Then you can run the script as follows:

```
# Generate batch scripts for the "saver" experiment
<>

# Run the batch scripts for the particular model
<>

```

- Compute the PCA transform for the generated keys:
```

```

### With AxoNN's tensor parallelism

Additionally, you can add `--use-axonn` flag to shard a large model like llama-13b over multiple GPUs.
For this you will need to launch the code using mpirun


```
mpirun -np 2 python -u evaluate_tasks.py --sequence-length 4096 --model-id meta-llama/Llama-2-13b-hf --model-type llama --use-h2o --heavy-ratio 0.1 --use-axonn
```
--->


