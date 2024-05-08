import torch
import triton
import numpy as np
from pca_topk import gather_outer_bmv_optimized, gather_inner_matrix_only_bmv_optimized
from sparq import gather_outer_bmv, gather_inner_matrix_only_bmv

B = 4
NH = 32
S = 800
D = 128
dtype = torch.float16



configs = [
    triton.testing.Benchmark(
        x_names=["sparsity"],  # Argument names to use as an x-axis for the plot
        x_vals=[0.125, 0.25, 0.5, 0.75, 1.0],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=["torch", "triton-optimized"],  # Label name for the lines
        line_names=["torch (full keys and values)", "Triton (Optimized)"],  # Line styles
        styles=[("black", "-"), ("blue", "-")],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance-" + ("fp16 (time in ms)" ),  # Name for the plot, used also as a file name for saving the plot.
        args = {"B": B, "NH" : NH, "S": S, "D": D}
    )
    ]


@triton.testing.perf_report(configs)
def benchmark_bmm1(sparsity, B, NH, S, D, provider):
    q = torch.randn((B*NH, 1, D), device='cuda', dtype=dtype)
    k = torch.randn((B*NH, S, D), device='cuda', dtype=dtype)
    choice = np.concatenate([np.sort(np.random.choice(S, size=(int(S*sparsity)), replace=False)) for _ in range(B*NH)]).reshape(B*NH, -1)
    token_mask = torch.tensor(choice, device="cuda")

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(q, k.transpose(1,2)), quantiles=quantiles)
    if provider == 'triton-optimized':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gather_outer_bmv_optimized(q, k.transpose(1, 2), token_mask), quantiles=quantiles)
    if provider == 'triton-sparq':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gather_outer_bmv(q, k.transpose(1, 2), token_mask, chunk=256), quantiles=quantiles)
    #perf = lambda ms: 2 * (B*NH) * S * D * 1e-12 / (ms * 1e-3)
    #return perf(ms), perf(max_ms), perf(min_ms)
    return ms, max_ms, min_ms

@triton.testing.perf_report(configs)
def benchmark_bmm2(sparsity, B, NH, S, D, provider):
    k_seq_len = S
    scores_sampled = torch.randn( (B*NH, 1, int(k_seq_len*sparsity)), device='cuda', dtype=dtype)
    scores = torch.randn( (B*NH, 1, k_seq_len), device='cuda', dtype=dtype)

    v = torch.randn((B*NH, k_seq_len, D), device='cuda', dtype=dtype)
    choice = np.concatenate([np.sort(np.random.choice(S, size=(int(k_seq_len*sparsity)), replace=False)) for _ in range(B*NH)]).reshape(B*NH, -1)
    token_mask = torch.tensor(choice, device="cuda")

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(scores, v), quantiles=quantiles)
    if provider == 'triton-optimized':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gather_inner_matrix_only_bmv_optimized(scores_sampled, v, token_mask), quantiles=quantiles)
    if provider == 'triton-sparq':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gather_inner_matrix_only_bmv(scores_sampled, v, token_mask, chunk=256), quantiles=quantiles)

    return ms, max_ms, min_ms



print("===== BENCHMARKING q@k.t() with various sparsities =======")
print("Batch Size : ", B)
print("Number of Heads : ", NH)
print("Number of Key Tokens (or sequence length) : ", S)
print("Hidden dimension per head : ", D)
result = benchmark_bmm1.run(print_data=True)



print("===== BENCHMARKING s@v with various sparsities =======")
print("Batch Size : ", B)
print("Number of Heads : ", NH)
print("Number of Key Tokens (or sequence length) : ", S)
print("Hidden dimension per head : ", D)
result = benchmark_bmm2.run(print_data=True)


