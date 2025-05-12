# Loki (Naive Sparsity)


## Attention Query-Key Sparsity Mode

When running with `sparsity_type = "attention-query-key"` (enabled via `--run-attention-query-key-sparsity` flag), the implementation:

1. Applies magnitude-based pruning to keep only the top 50% of weights by magnitude in linear projection layers
2. Optimizes the attention mechanism's query-key operations using torch's native sparse matrix multiplication (more below)
3. The sparse representations are computed once and reused across forward passes


## Modified Files
- `test_attention_benchmark_fixed.py`: Main benchmark script
- `./methods/pca_topk/attention_benchmark_apex.py`: Implementation of attention mechanisms
- `./methods/pca_topk/sparsity_utils.py`: Utilities for sparse operations

## Usage Example

Benchmark for vanilla vs loki vs sparse naive
```bash
python evaluate_compute.py

```

Run the benchmark with sparse attention query-key optimization:


```bash
python test_attention_benchmark_fixed.py --orig-pca-dir /cmlscratch/sukriti5/pca_stuff/pca_components --cache-seq-len 3500 --num-gen-steps 10 --top-d 32 --num-heads 16 --run-loki-without-sparsity --run-attention-query-key-sparsity --output-csv ./attention_query_key_results.csv
```

### Sparse Matrix Multiplication
- For 2D tensors: Using `torch.sparse.mm` for efficient sparse matrix multiplication
- For 3D tensors (batched attention operations): Using `torch.bmm` for batched multiplication
- Precision handling: Converting half-precision tensors to float32 temporarily for sparse operations, as PyTorch's sparse operations don't directly support half-precision

### Sparsity Configuration
- Default sparsity level: 50% (ie keeping top 50% of weights by magnitude for now)
- Implementation: Magnitude-based pruning applied to linear projection layers
- Format: Using `torch.sparse_coo_tensor` to create COO format sparse representations
- One-time sparse conversion: In the `SparseLinear` class, sparse representation is created only once via the `make_sparse()` method. For caching, rhe sparse weight (`self._sparse_weight`) is stored as a class attribute and reused across forward passes

I've followed the same benchmark/attributes as in yours and Connor's benchmark files.
