from lm_perplexity_eval import evaluate
from top_k.modify_opt import make_attention_top_k 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--top-k", type=int, default=-1, help="top k tokens to consider," 
                    "set to -1 to use all tokens")
parser.add_argument("--model-id", type=str, default="facebook/opt-350m", help="huggingface model to use")
parser.add_argument("--sequence-length", type=int, default=2048, help="sequence length")
parser.add_argument("--use-axonn", action='store_true', default=False, help="shard a model using AxoNN")

args = parser.parse_args()

if args.top_k !=-1:
    make_attention_top_k(top_k=args.top_k)

ppl = evaluate(model_id=args.model_id,
            dataset="wikitext",
            sequence_length=args.sequence_length,
            use_axonn=args.use_axonn)

print(ppl)
