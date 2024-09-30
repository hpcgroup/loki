import os
import argparse
import numpy as np

models = {
  "Llama-2-7B" : {"path": "meta-llama/Llama-2-7b-hf", 
                  "type": "llama",
                  "seq_len": 4096, 
                  "num_gpus": 1,
                  "category": "small"},
  "Llama-2-13B" : {"path": "meta-llama/Llama-2-13b-hf",
                  "type": "llama", 
                  "seq_len": 4096, 
                  "num_gpus": 1, 
                  "category": "small"},
  "Llama-2-70B" : {"path": "meta-llama/Llama-2-70b-hf",
                  "type": "llama",
                  "seq_len": 4096, 
                  "num_gpus": 4, 
                  "category": "large"},
  "Llama-3-8B" : {"path": "meta-llama/Meta-Llama-3-8B",
                  "type": "llama",
                  "seq_len": 8192, 
                  "num_gpus": 1, 
                  "category": "small"},
  "Llama-3-70B" : {"path": "meta-llama/Meta-Llama-3-70B",
                  "type": "llama",
                  "seq_len": 8192,
                  "num_gpus": 4, 
                  "category": "large"},
  "TinyLlama-1.1B" : {"path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  "type": "llama",
                  "seq_len": 2048,
                  "num_gpus": 1, 
                  "category": "small"},
  "Mistral-7B" : {"path": "mistralai/Mistral-7B-v0.1",
                  "type": "mistral",
                  "seq_len": 4096, 
                  "num_gpus": 1, 
                  "category": "small"},
  "Mixtral-8x7B" : {"path": "mistralai/Mixtral-8x7B-v0.1",
                    "type": "mistral",
                    "seq_len": 4096, 
                    "num_gpus": 4,
                    "category": "large"},
  "Mixtral-8x22B" : {"path": "mistralai/Mixtral-8x22B-v0.1",
                  "type": "mistral",
                  "seq_len": 4096, 
                  "num_gpus": 8,
                  "category": "large"},
  "Pythia-6.9B" : {"path": "EleutherAI/pythia-6.9b",
                  "type": "gptneox",
                  "seq_len": 2048, 
                  "num_gpus": 1, 
                  "category": "small"},
}

EXPERIMENTS = ["base_hf",  "h2o", "topk", "saver", "pca_topk"]
TOPK_VALUES = [0.5, 0.25, 0.125]
TOPD_VALUES = [64, 32, 16] # Assuming the head dimension is 128 as is the case for most models
ROTARY_VALUES = ["prerotary", "postrotary"]
DATASETS = ["wikitext-test", "wikitext-valid", "bookcorpus", "c4"]

TIMES = {
    "small": {
        "ppl": {
            "base_hf": "01:00:00",
            "topk": "01:00:00",
            "saver": "01:00:00",
            "h2o": "01:00:00",
            "pca_topk": "01:00:00"
        },
        "lm_harness": {
            "base_hf": "02:00:00",
            "topk": "02:00:00",
            "h2o": "04:00:00",
            "pca_topk": "02:00:00"
        }
    },
    "large": {
        "ppl": {
            "base_hf": "02:30:00",
            "topk": "02:30:00",
            "saver": "03:00:00",
            "h2o": "04:00:00",
            "pca_topk": "03:00:00"
        },
        "lm_harness": {
            "base_hf": "04:00:00",
            "topk": "04:00:00",
            "h2o": "08:00:00",
            "pca_topk": "04:30:00"
        }
    },
    "very-large": {
        "ppl": {
            "base_hf": "03:30:00",
            "topk": "03:30:00",
            "saver": "04:00:00",
            "h2o": "05:00:00",
            "pca_topk": "04:00:00"
        },
        "lm_harness": {
            "base_hf": "05:00:00",
            "topk": "05:00:00",
            "h2o": "10:00:00",
            "pca_topk": "05:30:00"
        }
    }

}

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, choices=EXPERIMENTS)
parser.add_argument('--models', type=str, nargs='+', choices=models.keys(), default=models.keys())
parser.add_argument('--eval-task', type=str, choices=["ppl", "lm_harness"]) 
parser.add_argument('--use-wandb', action='store_true', default=False, help="use wandb")
parser.add_argument('--use-axonn', action='store_true', default=False, help="shard a model using AxoNN")
parser.add_argument('--eval-dataset', type=str, choices=DATASETS, default="")
parser.add_argument('--transform-dataset', type=str, choices=DATASETS, default="")
parser.add_argument('--topk-list', type=float, nargs='+', default=TOPK_VALUES)
parser.add_argument('--topd-list', type=int, nargs='+', default=TOPD_VALUES)
args = parser.parse_args()

LOKI_HOME = "/global/homes/p/prajwal/Inference/approximate-attention/"
examples_dir = os.path.join(LOKI_HOME, "examples")

template_file = os.path.join(examples_dir, f"{args.exp}", "job_template_perlmutter.sh")

def get_config_list(args):
    base_dict = {}
    if args.eval_dataset != "":
        base_dict["tdataset"] = args.eval_dataset
    if args.exp == "base_hf" or args.exp == "saver":
        config = [base_dict]
    if args.exp == "topk":
        config = [base_dict | {"top_k": topk} for topk in args.topk_list]
    if args.exp == "h2o":
        config = [base_dict | {"heavy_ratio": heavy_ratio} for heavy_ratio in args.topk_list]
    if args.exp == "pca_topk":
        if args.transform_dataset != "":
            base_dict["transform_dataset"] = args.transform_dataset
        config = [base_dict | {"top_k": topk, "top_r": topd, "rotary_type": rotary} for topk in args.topk_list for topd in args.topd_list for rotary in ROTARY_VALUES]
    return config
  

# Some basic assertions
if args.exp == "saver" or args.exp == "pca_topk":
    assert args.eval_dataset != "", "Target dataset is required for saver and pca_topk experiments"
else:
    assert args.eval_dataset == "", "Target dataset is not required for this experiment"

if args.exp == "saver":
    assert args.eval_task != "lm_harness", "lm_harness is not supported for saver"

# Output file path for the experiment logs
out_file_prefix= os.path.join(LOKI_HOME, "logs", "perlmutter", f"{args.exp}", f"{args.eval_task}")
if not os.path.exists(out_file_prefix):
    os.makedirs(out_file_prefix)

# Path where the batch files will be stored
batch_file_prefix = os.path.join(LOKI_HOME, "examples", f"{args.exp}", "perlmutter", f"{args.eval_task}")
if not os.path.exists(batch_file_prefix):
    os.makedirs(batch_file_prefix)

configs = get_config_list(args)

for model in args.models:
    total_gpus = models[model]["num_gpus"]
    nodes = (total_gpus + 3) // 4 # Assuming 4 GPUs per node
    gpus = min(total_gpus, 4)
    seqlen = models[model]["seq_len"]
    model_path = models[model]["path"]
    model_type = models[model]["type"]
    time = TIMES[models[model]["category"]][args.eval_task][args.exp]

    batch_file_dir = os.path.join(batch_file_prefix, model)

    if not os.path.exists(batch_file_dir):
        os.makedirs(batch_file_dir)

    with open(template_file) as f:
        template = f.read()

    # Set experiment specific parameters
    axonn_args = ""
    if args.use_axonn:
        axonn_args = "--use-axonn"
    
    wandb_args = ""
    if args.use_wandb:
        wandb_args = "--use-wandb"

    if args.eval_task == "ppl":
        eval_args = ""
    elif args.eval_task == "lm_harness":
        eval_args = "--lm-harness-eval"

    for config in configs:
        outfile_name = f"{model}"
        for key, value in config.items():
            outfile_name = outfile_name + f"_{key.upper()}_{value}"
        output_file = os.path.join(out_file_prefix, outfile_name + ".log")

        # Format the template script with the args and config parameters
        script = template.format(
            nodes=nodes,
            gpus=gpus,
            model_path=model_path,
            model_type=model_type,
            seqlen=seqlen,
            output_file=output_file,
            axonn_args=axonn_args,
            wandb_args=wandb_args,
            eval_args=eval_args,
            time=time,
            **config
        )

        script_file = os.path.join(batch_file_dir, outfile_name + ".sh")
        with open(script_file, "w") as f:
            f.write(script)

        # TODO: Add time
        print(f"sbatch {script_file}")
        #if args.run:
        #    import subprocess
        #    subprocess.run(["sbatch", "-t" , f"{args.time}", "--reservation", "axonn_gb1" ,f"{script_file}"])
