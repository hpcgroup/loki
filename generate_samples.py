from methods import init_tensor_saver
from methods.common.configure_model import get_h2o_args, get_topk_args, get_spar_args, get_pca_args, get_save_tensor_args
from methods.common.configure_model import get_modifier
import numpy as np
import torch

import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RANK"] = os.getenv("SLURM_PROCID", "0")

from transformers import pipeline
from huggingface_hub import login


login("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")

#torch.manual_seed(42)
#torch.cuda.manual_seed(42)
#np.random.seed(42)


def generate_response(prompt, model_id):
    # Load the model and tokenizer
    generator = pipeline("text-generation", model=model_id, token='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', device="cuda")

    # Generate response to the prompt
    response = generator(prompt, max_length=512, do_sample=True, temperature=0.9)

    return response[0]['generated_text']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="facebook/opt-350m", help="huggingface model to use")
    parser.add_argument("--model-type", type=str, default="opt", help="model type - opt, llama, gpt-neo")
    parser.add_argument("--sequence-length", type=int, default=4096, help="sequence length")
    parser.add_argument("--use-axonn", action='store_true', default=False, help="shard a model using AxoNN")

    parser = get_h2o_args(parser)
    parser = get_topk_args(parser)
    parser = get_spar_args(parser)
    parser = get_pca_args(parser)
    args = parser.parse_args()

    rank = 0
    world_size = 1
    if args.use_axonn:
        world_size = os.getenv("WORLD_SIZE")
        rank = int(os.getenv("RANK"))


    modifier_method = get_modifier(args)
    #if modifier_method is None:
    #    raise ValueError("Modifier method not found")

    print (modifier_method)

    cache = None
    if args.use_topk:
        modifier_method(args.top_k)
    elif args.use_h2o:
        modifier_method(args.heavy_ratio)
    elif args.use_sparq or args.use_spark:
        modifier_method(args.top_r, args.top_k)
    elif args.use_spar_hat:
        cache = modifier_method(args.top_r)
    elif args.use_pca:
        modifier_method(args.top_r)
        args.use_axonn = False
    elif args.use_pca_topk:
        modifier_method(args.top_r, args.top_k)
        args.use_axonn = False

    
    prompts = ["In a world where technology has advanced beyond imagination, society grapples with the consequences of its own creations. The integration of artificial intelligence into everyday life has revolutionized how we live, work, and interact. However, with great power comes great responsibility, and ethical dilemmas abound. Governments struggle to regulate the use of AI, while corporations push the boundaries of what is possible in the pursuit of profit. Meanwhile, individuals navigate a landscape where privacy is increasingly scarce, and the line between human and machine blurs. Against this backdrop, a new generation of thinkers emerges, questioning the very nature of consciousness and what it means to be human. As tensions rise and alliances shift, the fate of humanity hangs in the balance, waiting to be written by those bold enough to seize the pen of destiny"]

    for prompt in prompts:
        response = generate_response(prompt, args.model_id)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("\n")


