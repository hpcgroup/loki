from methods.pca_topk.attention_benchmark_apex import benchmark_attention
import json
import torch
import os
import gc
import sys
import traceback

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

os.environ["OMP_NUM_THREADS"] = "1"  # Set OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # Set MKL threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Set numexpr threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Set OpenBLAS threads

def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f} GB used, "
          f"{torch.cuda.memory_reserved() / (1024**3):.2f} GB reserved")

if __name__ == "__main__":
    os.makedirs("compute_files", exist_ok=True)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    with torch.no_grad():
        for prompt_length in [512, 1024, 2048]:
            for num_gen_steps in [64, 128, 256]:
                # 1. Vanilla Attention (without Loki)
                print(f"\nRunning Vanilla Attention benchmark...")
                print(f"prompt length = {prompt_length}, gen length = {num_gen_steps}, batch_size=16")
                try:
                    free_gpu_memory()
                    _, times_vanilla = benchmark_attention(
                        prompt_length=prompt_length, 
                        num_gen_steps=num_gen_steps, 
                        batch_size=16, 
                        vanilla=True,
                        pcatopk=False,
                        dtype=torch.float16
                    )
                    vanilla_filename = f"compute_files/vanilla_prompt_{prompt_length}_gen_{num_gen_steps}.json"
                    print(f"Saving to {vanilla_filename}")
                    with open(vanilla_filename, "w") as f:
                        json.dump(times_vanilla, f, indent=2)
                    print("Vanilla Attention Times:")
                    for key, value in times_vanilla.items():
                        print(f"  {key}: {value:.6f} s")
                    print(f"Net time (minus cache updates): {times_vanilla.get('total', 0) - times_vanilla.get('cache-update', 0):.6f} s")
                except Exception as e:
                    print(f"Error in Vanilla benchmark: {str(e)}")
                    traceback.print_exc()

                for topk in [4, 8]:
                    for topr in [4]:
                        # 2. Loki without sparsity
                        print(f"\nRunning Loki without Sparsity benchmark...")
                        print(f"prompt length = {prompt_length}, gen length = {num_gen_steps}, batch_size=16, topk={topk} and topr={topr}")
                        try:
                            free_gpu_memory()
                            times_loki, _ = benchmark_attention(
                                prompt_length=prompt_length, 
                                num_gen_steps=num_gen_steps, 
                                batch_size=16, 
                                topk=prompt_length // topk, 
                                topr=128 // topr, 
                                vanilla=False,
                                pcatopk=True,
                                # No sparsity_type parameter
                                dtype=torch.float16
                            )
                            loki_filename = f"compute_files/loki_prompt_{prompt_length}_gen_{num_gen_steps}_topk_{topk}_topr_{topr}.json"
                            print(f"Saving to {loki_filename}")
                            with open(loki_filename, "w") as f:
                                json.dump(times_loki, f, indent=2)
                            print("Loki without Sparsity Times:")
                            for key, value in times_loki.items():
                                print(f"  {key}: {value:.6f} s")
                            print(f"Net time (minus cache updates): {times_loki.get('total', 0) - times_loki.get('cache-update', 0):.6f} s")
                        except Exception as e:
                            print(f"Error in Loki benchmark: {str(e)}")
                            traceback.print_exc()
                            
                        # 3. Loki with attention-query-key sparsity
                        print(f"\nRunning Loki with Attention-Query-Key Sparsity benchmark...")
                        print(f"prompt length = {prompt_length}, gen length = {num_gen_steps}, batch_size=16, topk={topk} and topr={topr}")
                        try:
                            free_gpu_memory()
                            times_sparsity, _ = benchmark_attention(
                                prompt_length=prompt_length, 
                                num_gen_steps=num_gen_steps, 
                                batch_size=16, 
                                topk=prompt_length // topk, 
                                topr=128 // topr, 
                                vanilla=False,
                                pcatopk=True,
                                sparsity_type="attention-query-key",
                                dtype=torch.float16
                            )
                            sparsity_filename = f"compute_files/attention_query_key_prompt_{prompt_length}_gen_{num_gen_steps}_topk_{topk}_topr_{topr}.json"
                            print(f"Saving to {sparsity_filename}")
                            with open(sparsity_filename, "w") as f:
                                json.dump(times_sparsity, f, indent=2)
                            print("Loki with Attention-Query-Key Sparsity Times:")
                            for key, value in times_sparsity.items():
                                print(f"  {key}: {value:.6f} s")
                            print(f"Net time (minus cache updates): {times_sparsity.get('total', 0) - times_sparsity.get('cache-update', 0):.6f} s")
                        except Exception as e:
                            print(f"Error in Attention-Query-Key Sparsity benchmark: {str(e)}")
                            traceback.print_exc()
                            
    print("\nAll benchmarks completed!")
    free_gpu_memory()


