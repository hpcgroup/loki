from methods import init_tensor_saver
from configure_model import get_h2o_args, get_topk_args, get_spar_args, get_pca_args, get_save_tensor_args
from configure_model import get_modifier

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    if modifier_method is None:
        raise ValueError("Modifier method not found")

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

    if args.lm_harness_eval:
        import lm_eval
        results = lm_eval.simple_evaluate(
            model = "hf",
            model_args=f"pretrained={args.model_id}",
            tasks = ["copa", "rte", "openbookqa", "mathqa", "winogrande", "hellaswag"],
            #tasks = ["hellaswag"],
            log_samples=False,
        )

        print(results["results"])
    else:
        from lm_perplexity_eval import evaluate
        ppl = evaluate(model_id=args.model_id,
                    dataset="wikitext-test",
                    sequence_length=args.sequence_length,
                    use_axonn=args.use_axonn,
                    past_key_values=cache,)

        print(ppl)
