from methods import init_tensor_saver
from methods.common.configure_model import get_h2o_args, get_topk_args, get_spar_args, get_pca_args, get_save_tensor_args
from methods.common.configure_model import get_modifier
from methods import init_logger, finish_logger
import methods

import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#LM_HARNESS_VALID_TASKS = ["hellaswag", "winogrande", "gsm8k", "mmlu", "truthfulqa_mc2", "arc_challenge"]
LM_HARNESS_TASKS = {
  "mmlu" : "acc,none",
  "gsm8k" : "exact_match,strict-match",
  "hellaswag" : "acc_norm,none",
  "winogrande" : "acc,none",
  "truthfulqa_mc2" : "acc,none",
  "arc_challenge" : "acc_norm,none"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="facebook/opt-350m", help="huggingface model to use")
    parser.add_argument("--model-type", type=str, default="opt", help="model type - opt, llama, gpt-neo")
    parser.add_argument("--sequence-length", type=int, default=4096, help="sequence length")
    parser.add_argument("--use-axonn", action='store_true', default=False, help="shard a model using AxoNN")
    parser.add_argument("--lm-harness-eval", action='store_true', default=False, help="use lm harness eval")
    parser.add_argument("--dataset", type=str, default="wikitext-test", help="dataset - wikitext, bookcorpus, c4")
    parser.add_argument("--use-wandb", action='store_true', default=False, help="use wandb")
    #parser.add_argument("--task", type=str, default="perplexity", help="task - perplexity, <lm_harness_tasks>")

    parser = get_h2o_args(parser)
    parser = get_topk_args(parser)
    parser = get_spar_args(parser)
    parser = get_pca_args(parser)
    parser = get_save_tensor_args(parser)
    args = parser.parse_args()

    if args.save_tensors:
        init_tensor_saver(args.tensors_dir)

    init_logger(args)

    modifier_method = get_modifier(args)
    if modifier_method is None:
        raise ValueError("Modifier method not found")

    print (modifier_method)

    cache = None
    if args.use_topk or args.use_h2o or args.use_pca_topk:
        modifier_method(args)
    elif args.use_sparq or args.use_spark:
        modifier_method(args.top_r, args.top_k)
    elif args.use_spar_hat:
        cache = modifier_method(args.top_r)
    elif args.use_pca:
        modifier_method(args.top_r)
        args.use_axonn = False
    
    if args.lm_harness_eval:
        import lm_eval
        from lm_perplexity_eval import evaluate
        model = evaluate(model_id=args.model_id,
                    dataset=args.dataset,
                    sequence_length=args.sequence_length,
                    use_axonn=args.use_axonn,
                    past_key_values=cache,
                    axonn_low_level_api=True,
                    return_model=True)
        results = lm_eval.simple_evaluate(
            model = "hf",
            #model_args=f"pretrained={args.model_id}",
            #model_args={"pretrained": model, "parallelize": True},
            model_args={"pretrained": model},
            tasks = LM_HARNESS_TASKS.keys(),
            log_samples=False,
            batch_size=16
        )

        if results is not None:
            print(results["results"])
            if methods.LOGGER is not None:
                methods.LOGGER.log_lm_harness_results(LM_HARNESS_TASKS, results["results"])
    else:
        from lm_perplexity_eval import evaluate
        print(args.use_axonn)
        ppl = evaluate(model_id=args.model_id,
                    dataset=args.dataset,
                    sequence_length=args.sequence_length,
                    use_axonn=args.use_axonn,
                    past_key_values=cache,
                    axonn_low_level_api=True)

        print(ppl)
        if methods.LOGGER is not None:
            methods.LOGGER.log_ppl(ppl)
    
    finish_logger()
