from lm_perplexity_eval import evaluate
from top_k.modify_opt import make_attention_top_k_opt
from top_k.modify_llama import make_attention_top_k_llama, make_attention_spar_llama
import argparse

MODEL_TYPE_FUNC_MAP = {
  'opt' : make_attention_top_k_opt,
  'llama' : make_attention_top_k_llama
}

parser = argparse.ArgumentParser()
parser.add_argument("--top-k", type=int, default=-1, help="top k tokens to consider," 
                    "set to -1 to use all tokens")
parser.add_argument("--model-id", type=str, default="facebook/opt-350m", help="huggingface model to use")
parser.add_argument("--model-type", type=str, default="opt", help="model type - opt, llama, gpt-neo")
parser.add_argument("--sequence-length", type=int, default=2048, help="sequence length")
parser.add_argument("--use-axonn", action='store_true', default=False, help="shard a model using AxoNN")
parser.add_argument("--use-spar", action='store_true', default=False, help="use the Spar algos")
parser.add_argument("--use-query", action='store_true', default=False, help="use the Spar-Q algo")
parser.add_argument("--top-r", type=int, default=-1, help="top r tokens to consider," "set to -1 to use all tokens")

args = parser.parse_args()

if args.top_k !=-1:
    if args.use_spar:
        if args.model_type == "llama":
            make_attention_spar_llama(top_r=args.top_r, top_k=args.top_k, use_keys=(not args.use_query))
    else:
        if args.model_type == "opt":
            make_attention_top_k_opt(top_k=args.top_k)
        elif args.model_type == "llama":
            make_attention_top_k_llama(top_k=args.top_k)

ppl = evaluate(model_id=args.model_id,
            dataset="wikitext",
            sequence_length=args.sequence_length,
            use_axonn=args.use_axonn)

print(ppl)
