from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import copy
from datasets import Dataset, DatasetDict


try:
    from axonn import axonn as ax
    from axonn.intra_layer import auto_parallelize
    from axonn.models.transformers import parallelize
    AXONN_AVAILABLE=True
except ImportError as e:
    print(e)
    AXONN_AVAILABLE=False


def get_model(
    model_id="facebook/opt-350m",
    device="cuda",
    dtype=torch.float16,
    use_axonn=True,
    axonn_low_level_api=True,
):
    # Use the AxoNN library to shard the model
    if use_axonn:
        assert AXONN_AVAILABLE, "axonn is not installed"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if rank == 0:
            print(f"Going to distribute the model over {world_size} GPUs")

        if model_id == "mistralai/Mixtral-8x22B-v0.1":
            ax.init(G_data=1, G_inter=1, G_intra_r=1, G_intra_c=1, G_intra_d=world_size)
        else:
            ax.init(G_data=1, G_inter=1, G_intra_r=world_size, G_intra_c=1, G_intra_d=1)
        
        axonn_low_level_api_success = False

        # Try to parallelize the model using the fast low level API
        if axonn_low_level_api:
            try:
                with parallelize(model_id):
                    if rank == 0:
                        print("Attempting to parallelize with fast low level API.")
                    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True).to(device)
                    if rank == 0:
                        print("Parallelized with the fast low level API.!")
                    axonn_low_level_api_success = True
                    torch.distributed.barrier()
            except AssertionError:
                if rank == 0:
                    print(f"Model {model_id} unavailable in the low level API. Try running evaluate with axonn_low_level_api=False ..")
                    exit(1)

        if not axonn_low_level_api_success:
            if rank == 0:
                print("Attempting to parallelize with the slow easy API.")
            with auto_parallelize():
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True).to(device)
            if rank == 0:
                print("Parallelized with the slower auto parallelize API.")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, force_download=True, trust_remote_code=True).to(device)

    return model

def evaluate_ppl(model_id="facebook/opt-350m", 
            dataset="wikitext",
            sequence_length=2048,
            device="cuda",
            dtype=torch.float16,
            use_axonn=False, 
            past_key_values=None,
            axonn_low_level_api=True,
            return_model=False):
    model = get_model(model_id=model_id, device=device, dtype=dtype, use_axonn=use_axonn, axonn_low_level_api=axonn_low_level_api)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print (f"Using {dataset}")
    if dataset == "wikitext-test":
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    elif dataset == "wikitext-valid":
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    elif dataset == "c4":
        test = load_from_disk('/pscratch/sd/p/prajwal/c4-sample')
    elif dataset == "bookcorpus":
        test = load_from_disk('/pscratch/sd/p/prajwal/bookcorpus-sample')
    else:
        raise ValueError("Dataset not supported")

    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    print (f"Total number of tokens (excluding pad token) = {len(encodings.input_ids[0])}")


    max_length = sequence_length
    stride = sequence_length 
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    model = model.to(device)

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                if past_key_values is not None:
                    outputs = model(input_ids.cuda(), past_key_values=copy.deepcopy(past_key_values), labels=target_ids.cuda())
                else:
                    outputs = model(input_ids.cuda(), labels=target_ids.cuda())
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl

if __name__ == "__main__":
    ppl = evaluate(model_id="facebook/opt-13b",use_axonn=True)
    print(ppl)
