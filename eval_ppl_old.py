import lm_eval
from lm_perplexity_eval import evaluate
from methods import init_tensor_saver
from methods import (
  make_llama_attention_h2o, make_llama_attention_top_k, make_llama_attention_sparq, make_llama_attention_spark, make_llama_attention_sparhat, make_llama_attention_pca, make_llama_attention_pca_topk,
  make_opt_attention_h2o, make_opt_attention_top_k,
  make_mistral_attention_h2o, make_mistral_attention_top_k, make_mistral_attention_pca, make_mistral_attention_pca_topk,
  #make_gemma_attention_top_k
  make_gptneox_attention_top_k
)
from configure_model import get_h2o_args, get_topk_args, get_spar_args, get_pca_args, get_save_tensor_args
from configure_model import get_modifier

from methods import SparHatCache
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


H2O_TYPE_FUNC_MAP = {
  'llama' : make_llama_attention_h2o,
  'opt' : make_opt_attention_h2o,
  'mistral' : make_mistral_attention_h2o
}

TOPK_TYPE_FUNC_MAP = {
  'llama' : make_llama_attention_top_k,
  'opt' : make_opt_attention_top_k,
  'mistral' : make_mistral_attention_top_k,
  'gpt-neox' : make_gptneox_attention_top_k
# 'gemma' : make_gemma_attention_top_k
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

PCA_TYPE_FUNC_MAP = {
  'llama' : make_llama_attention_pca,
  'mistral' : make_mistral_attention_pca
}

PCA_TOPK_TYPE_FUNC_MAP = {
  'llama' : make_llama_attention_pca_topk,
  'mistral': make_mistral_attention_pca_topk
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="facebook/opt-350m", help="huggingface model to use")
    parser.add_argument("--model-type", type=str, default="opt", help="model type - opt, llama, gpt-neo")
    parser.add_argument("--sequence-length", type=int, default=4096, help="sequence length")
    parser.add_argument("--use-axonn", action='store_true', default=False, help="shard a model using AxoNN")
    parser.add_argument("--lm-harness-eval", action='store_true', default=False, help="use lm harness eval")

    parser = get_h2o_args(parser)
    parser = get_topk_args(parser)
    parser = get_spar_args(parser)
    parser = get_pca_args(parser)
    parser = get_save_tensor_args(parser)
    args = parser.parse_args()

    if args.save_tensors:
        init_tensor_saver(args.tensors_dir)

    modifier_method = get_modifier(args)
    print (modifier_method)

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
        cache = SPARHAT_TYPE_FUNC_MAP[args.model_type](args.top_r)
    elif args.use_pca:
        PCA_TYPE_FUNC_MAP[args.model_type](args.top_r)
        args.use_axonn = False
    elif args.use_pca_topk:
        PCA_TOPK_TYPE_FUNC_MAP[args.model_type](args.top_r, args.top_k)
        args.use_axonn = False

    if args.lm_harness_eval:
        results = lm_eval.simple_evaluate(
            model = "hf",
            model_args=f"pretrained={args.model_id}",
            tasks = ["copa", "rte", "openbookqa", "mathqa", "winogrande", "hellaswag"],
            #tasks = ["hellaswag"],
            log_samples=False,
        )

        print(results["results"])
    else:
        ppl = evaluate(model_id=args.model_id,
                    dataset="wikitext",
                    sequence_length=args.sequence_length,
                    use_axonn=args.use_axonn,
                    past_key_values=cache,)

        print(ppl)
