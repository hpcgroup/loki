#!/usr/bin/env python3
"""Interactive test script for attention_benchmark_apex.py
This script allows testing a single case of the benchmark with specified parameters.
"""

from methods.pca_topk.attention_benchmark_apex import benchmark_attention
import argparse
import json
import torch
import os
import gc
import sys
import traceback

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Set thread limits to avoid OpenBLAS thread creation errors
os.environ["OMP_NUM_THREADS"] = "1"  # Set OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # Set MKL threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Set numexpr threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Set OpenBLAS threads

def free_gpu_memory():
    """Force garbage collection and empty CUDA cache to free up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f} GB used, "
          f"{torch.cuda.memory_reserved() / (1024**3):.2f} GB reserved")

def parse_args():
    parser = argparse.ArgumentParser(description="Test PCA TopK Attention Benchmark")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--orig-pca-dir", type=str, required=True, 
                        help="Directory containing original PCA components")
    parser.add_argument("--num-gen-steps", type=int, default=100, 
                        help="Number of generation steps")
    parser.add_argument("--cache-seq-len", type=int, default=2048, 
                        help="Cache sequence length (prompt length)")
    parser.add_argument("--top-d", type=int, default=64, 
                        help="Top-d parameter (related to topk)")
    parser.add_argument("--topk", type=int, default=None,
                        help="Direct topk parameter (overrides calculation from top-d if provided)")
    parser.add_argument("--topr", type=int, default=32,
                        help="Direct topr parameter")
    parser.add_argument("--output-csv", type=str, default=None, 
                        help="Output CSV file path")
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Batch size")
    parser.add_argument("--num-heads", type=int, default=32, 
                        help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=32, 
                        help="Number of transformer layers")
    parser.add_argument("--run-loki-without-sparsity", action="store_true", default=True,
                        help="Whether to run Loki without sparsity benchmark")
    parser.add_argument("--run-attention-query-key-sparsity", action="store_true", default=True,
                        help="Whether to run Loki with attention-query-key sparsity benchmark")
    parser.add_argument("--run-vanilla", action="store_true", default=True,
                        help="Whether to run vanilla attention benchmark (without Loki)")
    parser.add_argument("--sparsity-type", type=str, default="attention-query-key", 
                        help="Sparsity type to use: attention-query-key")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    
    # Print GPU information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Create compute_files directory for JSON outputs
    os.makedirs("compute_files", exist_ok=True)
    
    # Ensure output directory exists if output_csv is provided
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # Free memory before benchmark
    free_gpu_memory()
    
    # Calculate topk and topr based on top-d or use direct values if provided
    if args.topk is None:
        topk = args.cache_seq_len // args.top_d
    else:
        topk = args.topk
    
    topr = args.topr  # Use the provided topr value
    
    print(f"Running benchmark with parameters:")
    print(f"  Prompt length: {args.cache_seq_len}")
    print(f"  Gen steps: {args.num_gen_steps}")
    print(f"  Batch size: {args.batch_size}")
    if args.topk is None:
        print(f"  Top-d: {args.top_d} (resulting in topk={topk})")
    else:
        print(f"  Topk: {topk} (directly specified)")
    print(f"  Topr: {topr}")
    
    with torch.no_grad():
        try:
            # Create file paths for results
            sparse_filename = f"compute_files/prompt_{args.cache_seq_len}_gen_{args.num_gen_steps}_topk_{topk}_topr_{topr}_sparse.json"
            loki_without_sparsity_filename = f"compute_files/prompt_{args.cache_seq_len}_gen_{args.num_gen_steps}_loki_without_sparsity_topk_{topk}_topr_{topr}.json"
            attention_query_key_filename = f"compute_files/prompt_{args.cache_seq_len}_gen_{args.num_gen_steps}_attention_query_key_topk_{topk}_topr_{topr}.json"
            vanilla_filename = f"compute_files/prompt_{args.cache_seq_len}_gen_{args.num_gen_steps}_vanilla.json"
            
            # Variable to track if we have sparse results
            times_sparse = None
            
            # First run: Loki + Sparsity (nvidia-apex-2-4) 
            # print(f"\nRunning Loki + Sparsity (nvidia-apex-2-4) benchmark...")
            # try:
            #     times_sparse, _ = benchmark_attention(
            #         prompt_length=args.cache_seq_len,
            #         num_gen_steps=args.num_gen_steps,
            #         batch_size=args.batch_size,
            #         num_heads=args.num_heads,
            #         num_layers=args.num_layers,
            #         topk=topk,
            #         topr=topr,
            #         vanilla=False,
            #         pcatopk=True,
            #         sparsity_type="nvidia-apex-2-4",
            #         dtype=torch.float16
            #     )
            #     
            #     # Save sparse results
            #     print(f"Saving to {sparse_filename}")
            #     with open(sparse_filename, "w") as f:
            #         json.dump(times_sparse, f, indent=2)
            #     
            #     # Print sparse results
            #     print("\nLoki + Sparsity Times:")
            #     for key, value in times_sparse.items():
            #         print(f"  {key}: {value:.6f} s")
            #     print(f"Net time (minus cache updates): {times_sparse.get('total', 0) - times_sparse.get('cache-update', 0):.6f} s")
            # except Exception as e:
            #     print(f"Error in Loki + Sparsity benchmark: {str(e)}")
            #     traceback.print_exc()
            
            # Set times_sparse to None since we're not running this benchmark
            times_sparse = None
            
            # Free memory before vanilla benchmark
            free_gpu_memory()
            
            # Variable to track if we have vanilla results
            times_loki_without_sparsity = None
            
            # Second run: Loki without Sparsity
            if args.run_loki_without_sparsity:
                print(f"\nRunning Loki without Sparsity benchmark...")
                try:
                    times_loki_without_sparsity, _ = benchmark_attention(  # Note: we're getting the first return value now
                        prompt_length=args.cache_seq_len,
                        num_gen_steps=args.num_gen_steps,
                        batch_size=args.batch_size,
                        num_heads=args.num_heads,
                        num_layers=args.num_layers,
                        topk=topk,
                        topr=topr,
                        vanilla=False,  # Use PCA TopK, not vanilla attention
                        pcatopk=True,   # Use PCA TopK
                        # No sparsity_type parameter, so no 2:4 sparsity
                        dtype=torch.float16
                    )
                    
                    # Add missing attributes to vanilla results for consistent comparison
                    # These operations don't exist in vanilla but are in sparse
                    if times_loki_without_sparsity:
                        missing_attrs = ['project', 'top-k', 'reshape-0', 'reshape-1', 'qk-matmul-2']
                        for attr in missing_attrs:
                            if attr not in times_loki_without_sparsity:
                                times_loki_without_sparsity[attr] = 0.0
                    else:
                        print("Error: No timing results returned for Loki without Sparsity benchmark")
                    
                    # Save vanilla results
                    print(f"Saving to {loki_without_sparsity_filename}")
                    with open(loki_without_sparsity_filename, "w") as f:
                        json.dump(times_loki_without_sparsity, f, indent=2)
                    
                    # Print vanilla results
                    print("\nLoki without Sparsity Times:")
                    for key, value in times_loki_without_sparsity.items():
                        print(f"  {key}: {value:.6f} s")
                    print(f"Net time (minus cache updates): {times_loki_without_sparsity.get('total', 0) - times_loki_without_sparsity.get('cache-update', 0):.6f} s")
                except Exception as e:
                    print(f"Error in Loki without Sparsity benchmark: {str(e)}")
                    traceback.print_exc()
            
            # Third run: Loki + Attention Query/Key Sparsity
            if args.run_attention_query_key_sparsity:
                print(f"\nRunning Loki + Attention Query/Key Sparsity benchmark...")
                try:
                    times_attention_query_key, _ = benchmark_attention(
                        prompt_length=args.cache_seq_len,
                        num_gen_steps=args.num_gen_steps,
                        batch_size=args.batch_size,
                        num_heads=args.num_heads,
                        num_layers=args.num_layers,
                        topk=topk,
                        topr=topr,
                        vanilla=False,
                        pcatopk=True,
                        sparsity_type="attention-query-key",
                        dtype=torch.float16
                    )
                    
                    # Add missing attributes if needed for consistent comparison
                    if times_attention_query_key:
                        # Check if query-key-sparsity is in the times
                        if 'query-key-sparsity' not in times_attention_query_key:
                            times_attention_query_key['query-key-sparsity'] = 0.0
                    else:
                        print("Error: No timing results returned for Loki + Attention Query/Key Sparsity benchmark")
                    
                    # Save attention query/key results
                    print(f"Saving to {attention_query_key_filename}")
                    with open(attention_query_key_filename, "w") as f:
                        json.dump(times_attention_query_key, f, indent=2)
                    
                    # Print attention query/key results
                    print("\nLoki + Attention Query/Key Sparsity Times:")
                    for key, value in times_attention_query_key.items():
                        print(f"  {key}: {value:.6f} s")
                    print(f"Net time (minus cache updates): {times_attention_query_key.get('total', 0) - times_attention_query_key.get('cache-update', 0):.6f} s")
                except Exception as e:
                    print(f"Error in Loki + Attention Query/Key Sparsity benchmark: {str(e)}")
                    traceback.print_exc()
            
            # Fourth run: Vanilla Attention (without Loki)
            times_vanilla = None
            if args.run_vanilla:
                print(f"\nRunning Vanilla Attention benchmark (without Loki)...")
                try:
                    # Free memory before vanilla benchmark
                    free_gpu_memory()
                    
                    _, times_vanilla = benchmark_attention(
                        prompt_length=args.cache_seq_len,
                        num_gen_steps=args.num_gen_steps,
                        batch_size=args.batch_size,
                        num_heads=args.num_heads,
                        num_layers=args.num_layers,
                        topk=topk,
                        topr=topr,
                        vanilla=True,   # Use vanilla attention
                        pcatopk=False,  # Don't use PCA TopK
                        dtype=torch.float16
                    )
                    
                    # Add missing attributes for consistent comparison with other methods
                    if times_vanilla:
                        # These operations don't exist in vanilla but are in other methods
                        missing_attrs = ['project', 'top-k', 'reshape-0', 'reshape-1', 'qk-matmul-2', 'query-key-sparsity']
                        for attr in missing_attrs:
                            if attr not in times_vanilla:
                                times_vanilla[attr] = 0.0
                    else:
                        print("Error: No timing results returned for Vanilla Attention benchmark")
                    
                    # Save vanilla results
                    print(f"Saving to {vanilla_filename}")
                    with open(vanilla_filename, "w") as f:
                        json.dump(times_vanilla, f, indent=2)
                    
                    # Print vanilla results
                    print("\nVanilla Attention Times:")
                    for key, value in times_vanilla.items():
                        print(f"  {key}: {value:.6f} s")
                    print(f"Net time (minus cache updates): {times_vanilla.get('total', 0) - times_vanilla.get('cache-update', 0):.6f} s")
                except Exception as e:
                    print(f"Error in Vanilla Attention benchmark: {str(e)}")
                    traceback.print_exc()
            
            # Save to CSV if requested
            if args.output_csv:
                import csv
                with open(args.output_csv, 'w', newline='') as csvfile:
                    fieldnames = ['method', 'prompt_length', 'gen_steps', 'top_d', 'total_time', 'cache_update_time', 'net_time']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Track if we have any results to write
                    wrote_results = False
                    
                    # Try to write Loki + Sparsity row if available 
                    # if times_sparse:
                    #     writer.writerow({
                    #         'method': 'loki_sparsity',
                    #         'prompt_length': args.cache_seq_len,
                    #         'gen_steps': args.num_gen_steps,
                    #         'top_d': args.top_d,
                    #         'total_time': times_sparse.get('total', 0),
                    #         'cache_update_time': times_sparse.get('cache-update', 0),
                    #         'net_time': times_sparse.get('total', 0) - times_sparse.get('cache-update', 0)
                    #     })
                    #     wrote_results = True
                    
                    # Vanilla Loki row if available
                    if args.run_loki_without_sparsity and times_loki_without_sparsity:
                        writer.writerow({
                            'method': 'loki_without_sparsity',
                            'prompt_length': args.cache_seq_len,
                            'gen_steps': args.num_gen_steps,
                            'top_d': args.top_d,
                            'total_time': times_loki_without_sparsity.get('total', 0),
                            'cache_update_time': times_loki_without_sparsity.get('cache-update', 0),
                            'net_time': times_loki_without_sparsity.get('total', 0) - times_loki_without_sparsity.get('cache-update', 0)
                        })
                        wrote_results = True
                    
                    # Attention Query/Key Sparsity row if available
                    if args.run_attention_query_key_sparsity and times_attention_query_key:
                        writer.writerow({
                            'method': 'loki_attention_query_key_sparsity',
                            'prompt_length': args.cache_seq_len,
                            'gen_steps': args.num_gen_steps,
                            'top_d': args.top_d,
                            'total_time': times_attention_query_key.get('total', 0),
                            'cache_update_time': times_attention_query_key.get('cache-update', 0),
                            'net_time': times_attention_query_key.get('total', 0) - times_attention_query_key.get('cache-update', 0)
                        })
                        wrote_results = True
                        
                    # Vanilla Attention row if available
                    if args.run_vanilla and times_vanilla:
                        writer.writerow({
                            'method': 'vanilla_attention',
                            'prompt_length': args.cache_seq_len,
                            'gen_steps': args.num_gen_steps,
                            'top_d': args.top_d,
                            'total_time': times_vanilla.get('total', 0),
                            'cache_update_time': times_vanilla.get('cache-update', 0),
                            'net_time': times_vanilla.get('total', 0) - times_vanilla.get('cache-update', 0)
                        })
                        wrote_results = True
                
                if wrote_results:
                    print(f"Results saved to CSV: {args.output_csv}")
                else:
                    print(f"Warning: No results were written to CSV file.")
            
        except Exception as e:
            print(f"Error in benchmark: {str(e)}")
            traceback.print_exc()
        
        # Final memory cleanup
        free_gpu_memory()
    
    print("All benchmarks completed!")
