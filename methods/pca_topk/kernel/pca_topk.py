import math
import warnings

import torch
import triton
import triton.language as tl
from torch import Tensor

def get_autotune_config():
  return [
    triton.Config({"n_chunk": 4}),
    triton.Config({"n_chunk": 8}),
    triton.Config({"n_chunk": 16}),
    triton.Config({"n_chunk": 32}),
    triton.Config({"n_chunk": 64}),
    triton.Config({"n_chunk": 128}),
    triton.Config({"n_chunk": 256}),
    triton.Config({"n_chunk": 512}),
    triton.Config({"n_chunk": 1024})
  ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['b', 'n', 'k'],
)
@triton.jit
def _kernel_gather_outer_bmv(
    A_ptr,
    B_ptr,
    I_ptr,
    Y_ptr,
    k: tl.constexpr,
    b: int,
    n: int,
    n_chunk: tl.constexpr,
    A_s0: int,
    A_s2: int,
    B_s0: int,
    B_s1: int,
    B_s2: int,
    I_s0: int,
    I_s1: int,
):
    pid_b = tl.program_id(axis=0).to(tl.int64)
    pid_n = tl.program_id(axis=1).to(tl.int64)
    a = tl.load(A_ptr + pid_b * A_s0 + tl.arange(0, k) * A_s2)  # (k)
    chunk_idx = pid_n * n_chunk + tl.arange(0, n_chunk)
    i = tl.load(I_ptr + pid_b * I_s0 + chunk_idx * I_s1, mask=(chunk_idx < n))# (n_chunk)
    b = tl.load(  # (k x n_chunk)
        B_ptr
        + pid_b * B_s0
        + (tl.arange(0, k) * B_s1)[:, None]
        + (i * B_s2)[None, :],
        mask=(chunk_idx < n)[None, :]
    )
    # # As tl.dot() is unavailable for matrix-vector
    y = tl.sum((a[:, None] * b).to(tl.float32), 0).to(a.dtype)  # (n_chunk)
    tl.store(Y_ptr + pid_b * n + chunk_idx, y, mask=(chunk_idx < n))

def gather_outer_bmv_optimized(A: Tensor, B: Tensor, I: Tensor) -> Tensor:
    """Batched vector-matrix multiplication, with a gather on the matrix outer dimension.

    Dimensions:
       b -- batch
       k -- inner dimension, must be a power of two
       n* -- (pre-gather) outer dimension
       n -- (post-gather) outer dimension (n <= n*)

    A -- (b, 1, k)         batch of vectors
    B -- (b, k, n*)        batch of matrices
    I -- int(b, n)         indices, in [0, n*)
    chunk -- int           size of chunks of `B` (along dimension `n`) to be processed at a time

    returns -- (b, 1, n)   the inner product of `A` and `B`, after gathering the outer dimension
                           according to `I`
    """
    if A.ndim > 3:
        assert B.ndim == A.ndim and I.ndim == A.ndim - 1
        return gather_outer_bmv_custom(
            A.flatten(end_dim=-3),
            B.flatten(end_dim=-3),
            I.flatten(end_dim=-2),
        ).unflatten(0, A.shape[:-2])
    assert A.ndim == 3 and B.ndim == 3 and A.shape[1] == 1 and A.shape[2] == B.shape[1]
    assert I.ndim == 2 and I.shape[0] == A.shape[0]

    b, k, n = A.shape[0], A.shape[2], I.shape[1]
    Y = torch.empty((b, 1, n), dtype=A.dtype, device=A.device)
    assert Y.stride(0) == n and Y.stride(2) == 1

    grid = lambda META: (b, triton.cdiv(n, META["n_chunk"]))
    _kernel_gather_outer_bmv[grid](
        A_ptr=A,
        B_ptr=B,
        I_ptr=I,
        Y_ptr=Y,
        b=b,
        k=k,
        n=n,
        A_s0=A.stride(0),
        A_s2=A.stride(2),
        B_s0=B.stride(0),
        B_s1=B.stride(1),
        B_s2=B.stride(2),
        I_s0=I.stride(0),
        I_s1=I.stride(1),
    )


    return Y
