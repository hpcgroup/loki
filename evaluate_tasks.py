from methods import init_tensor_saver
from methods.common.configure_model import get_h2o_args, get_topk_args, get_pca_args, get_save_tensor_args
from methods.common.configure_model import get_modifier
from methods import init_logger, finish_logger
from methods.common.ppl import get_model, evaluate_ppl
import methods

import argparse
import os

# Required to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Hugging Face OpenLLM Tasks and associated metrics from May 2024
# https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard
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
    parser.add_argument("--dataset", type=str, default="wikitext-test", help="dataset - wikitext-test, wikitext-valid, bookcorpus, c4")
    parser.add_argument("--use-wandb", action='store_true', default=False, help="use wandb")

    # Get Method Specific Arguments
    parser = get_h2o_args(parser)
    parser = get_topk_args(parser)
    parser = get_pca_args(parser)
    parser = get_save_tensor_args(parser)
    args = parser.parse_args()

    if args.save_tensors:
        init_tensor_saver(args.tensors_dir)

    init_logger(args)

    modifier_method = get_modifier(args)
    if modifier_method is None:
        print ("[INFO] Running Base HF Model without any modification")
    else:
        print (modifier_method)
        modifier_method(args)
    
    if args.lm_harness_eval: # Use lm harness eval
        import lm_eval
        use_axonn_low_level_api = True

        if args.model_type == "gptneox":
            print ("[INFO] Disabling Axonn for GPT-NeoX")
            args.use_axonn = False
        if args.model_id == "mistralai/Mixtral-8x22B-v0.1" or args.use_h2o:
            print ("[INFO] Disabling Low Level Axonn API for Mixtral-8x22B")
            use_axonn_low_level_api = False
        
        if args.use_h2o:
            # H2O modification for lm_eval does not work for gsm8k. Need to modify their hh code
            del LM_HARNESS_TASKS["gsm8k"]

        if args.use_axonn:
            # If using axonn, we need to load the model and shard it before passing it to lm_eval
            model = get_model(model_id=args.model_id,
                        use_axonn=args.use_axonn,
                        axonn_low_level_api=use_axonn_low_level_api)

            results = lm_eval.simple_evaluate(
                model = "hf",
                model_args={"pretrained": model},
                tasks = LM_HARNESS_TASKS.keys(),
                log_samples=False,
                batch_size=8
            )
        else:
            results = lm_eval.simple_evaluate(
                model = "hf",
                model_args=f"pretrained={args.model_id}",
                tasks = LM_HARNESS_TASKS.keys(),
                log_samples=False,
                batch_size=16
            )

        if results is not None:
            print(results["results"])
            if methods.LOGGER is not None:
                methods.LOGGER.log_lm_harness_results(LM_HARNESS_TASKS, results["results"])
    else: # Use PPL Evaluation
        # Some issue with loading pythia with axonn
        use_axonn_low_level_api = True
        if args.model_type == "gptneox":
            args.use_axonn = False

        # Low level API not available for Mixtral or in the H2O modification
        if args.model_id == "mistralai/Mixtral-8x22B-v0.1" or args.use_h2o:
            use_axonn_low_level_api = False

        ppl = evaluate_ppl(model_id=args.model_id,
                    dataset=args.dataset,
                    sequence_length=args.sequence_length,
                    use_axonn=args.use_axonn,
                    axonn_low_level_api=use_axonn_low_level_api)

        print(ppl)
        if methods.LOGGER is not None:
            methods.LOGGER.log_ppl(ppl)
    
    finish_logger()
