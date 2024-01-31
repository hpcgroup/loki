from lm_perplexity_eval import evaluate
from opt_top_k.modify_llama import make_s_hat_attention
from opt_top_k.cache_utils import SparHatCache
import argparse

MODEL_TYPE_FUNC_MAP = {
  'llama' : make_s_hat_attention
}

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str, default="facebook/opt-350m", help="huggingface model to use")
parser.add_argument("--model-type", type=str, default="opt", help="model type - opt, llama, gpt-neo")
parser.add_argument("--sequence-length", type=int, default=2048, help="sequence length")
parser.add_argument("--use-axonn", action='store_true', default=False, help="shard a model using AxoNN")
parser.add_argument("--top-r", type=int, default=-1, help="top r tokens to consider," "set to -1 to use all tokens")

args = parser.parse_args()

make_s_hat_attention()

ppl = evaluate(model_id=args.model_id,
            dataset="wikitext",
            sequence_length=args.sequence_length,
            use_axonn=args.use_axonn,
            past_key_value_cache=SparHatCache(args.top_r),)


print(ppl)
