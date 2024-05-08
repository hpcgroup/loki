from pca_topk import gather_outer_bmv_optimized
import torch
import numpy as np
import pytest

@pytest.mark.parametrize("B", [2, 4, 8, 16])
@pytest.mark.parametrize("NH", [8, 16, 32])
@pytest.mark.parametrize("S", [32, 33, 37, 64, 69, 73, 128, 255, 259, 1024, 1028, 2048, 2500])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("sparsity", [0.125, 0.25, 0.5])
def test_first_bmm(B, NH, S, D, sparsity, dtype=torch.float32):
    """
    Test the correctness of the first bmm (q @ k.t())
    B - batch size
    NH - number of heads
    S - key sequence length
    D - hidden dimension per head
    sparsity - topk in [0,1)
    """

    k_seq_len = S
    q = torch.randn((B*NH, 1, D), device='cuda', dtype=dtype)
    k = torch.randn((B*NH, k_seq_len, D), device='cuda', dtype=dtype)

    choice = np.concatenate([np.sort(np.random.choice(S, size=(int(k_seq_len*sparsity)), replace=False)) for _ in range(B*NH)]).reshape(B*NH, -1)
    token_mask = torch.tensor(choice, device="cuda")

    y_optimized = gather_outer_bmv_optimized(q, k.transpose(1,2), token_mask)

    for i in range(B*NH):
        token_mask[i] += k_seq_len*i
    
    k_reshaped = k.view(-1, D)
    k_sampled = torch.index_select(k_reshaped, dim=0, index=token_mask.view(-1)).reshape(B*NH, -1, D)
    y_torch = torch.bmm(q, k_sampled.transpose(1,2))

    assert torch.allclose(y_optimized, y_torch, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    test_first_bmm(
        B=2,
        NH=32,
        S=1024,
        D=128,
        sparsity=0.25
    )
