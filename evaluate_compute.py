from methods.pca_topk.attention_benchmark import benchmark_attention
import json
import torch


# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if __name__ == "__main__":
    with torch.no_grad():
        prompt_length = 3500
        for num_gen_steps in [512]:
            for topk in [2, 4, 8]:
                for topr in [2, 4, 8]:
                    print(f"prompt length = {prompt_length}, gen length = {num_gen_steps}, batch_size={16}, topk={topk} and topr={topr}")
                    times_pca_topk, _ = benchmark_attention(prompt_length=prompt_length, num_gen_steps=num_gen_steps, batch_size=16, topk=prompt_length // topk, topr=128 // topr, vanilla=False)
                    #with open(f"prompt_{prompt_length}_gen_{num_gen_steps}_pca_topk_opt_first_matmul.json", "w") as f:
                    with open(f"compute_files/prompt_{prompt_length}_gen_{num_gen_steps}_topk_{topk}_topr_{topr}.json", "w") as f:
                        json.dump(times_pca_topk, f, indent=2)

            _, times_vanilla = benchmark_attention(prompt_length=prompt_length, num_gen_steps=num_gen_steps, batch_size=16, topk=prompt_length // topk, topr=128 // topr, pcatopk=False)
            with open(f"compute_files/prompt_{prompt_length}_gen_{num_gen_steps}_vanilla.json", "w") as f:
                json.dump(times_vanilla, f, indent=2)

    
