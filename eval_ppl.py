from lm_perplexity_eval import evaluate
import methods
#from methods import init_tensor_saver
from methods import (
  make_llama_attention_h2o, make_llama_attention_top_k, make_llama_attention_sparq, make_llama_attention_spark, make_llama_attention_sparhat, make_llama_attention_histh2o,
  make_opt_attention_h2o, make_opt_attention_top_k
)
from methods import SparHatCache
import argparse


def get_h2o_args(parser):
    parser.add_argument("--use-h2o", action='store_true', default=False, help="use the H2O algos")
    parser.add_argument("--heavy-ratio", type=float, default=0.1, help="H2O heavy ratio," "set to 0.1 by default")
    parser.add_argument("--recent-ratio", type=float, default=0.1, help="H2O recent ratio," "set to 0.1 by default")
    return parser

def get_topk_args(parser):
    parser.add_argument("--use-topk", action='store_true', default=False, help="use the H2O algos")
    parser.add_argument("--top-k", type=int, default=-1, help="top k tokens to consider," "set to -1 to use all tokens")
    return parser

def get_spar_args(parser):
    parser.add_argument("--use-sparq", action='store_true', default=False, help="use the Spar algos")
    parser.add_argument("--use-spark", action='store_true', default=False, help="use the Spar algos")
    parser.add_argument("--use-spar-hat", action='store_true', default=False, help="use the Spar Hat algo")
    parser.add_argument("--top-r", type=int, default=-1, help="top r channels to consider," "set to -1 to use all channels")
    return parser

H2O_TYPE_FUNC_MAP = {
  'llama' : make_llama_attention_histh2o,
  'opt' : make_opt_attention_h2o
}

TOPK_TYPE_FUNC_MAP = {
  'llama' : make_llama_attention_top_k,
  'opt' : make_opt_attention_top_k
}

SPARQ_TYPE_FUNC_MAP = {
  'llama' : make_llama_attention_sparq
}

SPARK_TYPE_FUNC_MAP = {
  'llama' : make_llama_attention_spark
}

SPARHAT_TYPE_FUNC_MAP = {
  'llama' : make_llama_attention_sparhat
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="facebook/opt-350m", help="huggingface model to use")
    parser.add_argument("--model-type", type=str, default="opt", help="model type - opt, llama, gpt-neo")
    parser.add_argument("--sequence-length", type=int, default=4096, help="sequence length")
    parser.add_argument("--use-axonn", action='store_true', default=False, help="shard a model using AxoNN")
    #parser.add_argument("--save-tensors", action='store_true', default=False, help="save tensors to disk")
    #parser.add_argument("--output-dir", type=str, default="output", help="output directory for saving tensors")

    parser = get_h2o_args(parser)
    parser = get_topk_args(parser)
    parser = get_spar_args(parser)
    args = parser.parse_args()

    #if args.save_tensors:
    #    init_tensor_saver(args.output_dir)

    cache = None
    if args.use_topk:
        TOPK_TYPE_FUNC_MAP[args.model_type](args.top_k)
    elif args.use_h2o:
        H2O_TYPE_FUNC_MAP[args.model_type](args.heavy_ratio)
    elif args.use_sparq:
        SPARQ_TYPE_FUNC_MAP[args.model_type](args.top_r, args.top_k)
    elif args.use_spark:
        SPARK_TYPE_FUNC_MAP[args.model_type](args.top_r, args.top_k)
    elif args.use_spar_hat:
        SPARHAT_TYPE_FUNC_MAP[args.model_type]()
        #SPARHAT_TYPE_FUNC_MAP[args.model_type](args.top_k)
        cache = SparHatCache(args.top_r)

    ppl = evaluate(model_id=args.model_id,
                dataset="wikitext",
                sequence_length=args.sequence_length,
                use_axonn=args.use_axonn,
                past_key_values=cache,)


    print(ppl)
