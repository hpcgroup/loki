import torch

def mask_attn_top_k(attn_weights, top_k, dim=2):
    # Find the indices of the top k elements along the specified dimension
    if attn_weights.shape[dim] <= top_k:
        return attn_weights

    _, indices = attn_weights.topk(top_k, dim=dim, largest=True)

    topk_attn_weights = torch.full_like(attn_weights, fill_value=float('-inf'))

    topk_attn_weights.scatter_(dim, indices, attn_weights.gather(dim, indices))

    return topk_attn_weights


def test_top_k():
    test_tensor = torch.rand(1, 1, 5, 5)
    print (test_tensor)

    ones = torch.ones_like(test_tensor, dtype=torch.bool)
    tril_mask = torch.tril(ones, diagonal=0)
    test_tensor[~tril_mask] = float('-inf')
    print (test_tensor)

    test_tensor = mask_attn_top_k(test_tensor, 2, dim=3)
    print (test_tensor)

if __name__ == "__main__":
    test_top_k()
