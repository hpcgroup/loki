from transformers import AutoTokenizer, AutoModelForCausalLM
from axonn.models.transformers import parallelize 
from axonn import axonn as ax
import torch
import random
import numpy as np
import argparse

from methods import init_tensor_saver
from methods.common.configure_model import get_h2o_args, get_topk_args, get_spar_args, get_pca_args, get_save_tensor_args
from methods.common.configure_model import get_modifier
from methods import init_logger, finish_logger
import methods

OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
ENDC = '\033[0m'

def init_everything():
    torch.distributed.init_process_group(backend='nccl')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(f"Going to distribute the model over {world_size} GPUs")
    ax.init(G_data=1, G_inter=1, G_intra_r=world_size, G_intra_c=1, G_intra_d=1)

def set_seed(seed=123456):
    # Extremely important for AxoNN
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="huggyllama/llama-7b", help="huggingface model to use")
    return parser

if  __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    model_id = args.model_id
    dtype = torch.float16

    init_everything()
    set_seed()
    with parallelize(model_id):
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_id)


    with open("data/input.txt", "r") as f:
        prompts = [p.strip() for p in f.readlines()]
      

    total_generated_tokens = 0
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    generations = []

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.autocast(device_type='cuda', dtype=dtype):
            outputs = model.generate(input_ids.cuda(), do_sample=True, max_new_tokens=60, num_beams=4)

        generated_tokens = outputs.numel() -  input_ids.numel()
        total_generated_tokens += generated_tokens

        generations.append(outputs[:, input_ids.numel():])

    end_event.record()
    
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event)
    tput = total_generated_tokens * 1000 / total_time

    if torch.distributed.get_rank() == 0:
        for prompt, generation in zip(prompts,generations):
            print(f"{OKBLUE}[PROMPT]: {prompt}{ENDC}")
            print(f"{OKGREEN}[GENERATION]: = {tokenizer.batch_decode(generation)[0]}{ENDC}")
            print("=====")
        print(f"Tput = {tput} generated tokens / second")

