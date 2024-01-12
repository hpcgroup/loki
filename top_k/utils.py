import torch
import math


def mask_top_k_elements_sparq(full_attention_scores, attention_mask, query, key, r, k):
    
    dh = query.shape[3]
    l = k / 2

    abs_top_q_r, indices = torch.abs(query).topk(r, dim=3, largest=True)
    query_top_r = torch.gather(query, 3, indices)

    s = key.shape[2]

    scaling_factor = dh * (torch.sum(abs_top_q_r, 3, keepdim=True) / torch.sum(torch.abs(query), 3, keepdim = True))

    key_top_r = torch.gather(key, 3, indices)

    score_estimates = torch.matmul(query_top_r, key_top_r.transpose(2, 3)) / torch.sqrt(scaling_factor)

    score_estimates = score_estimates + attention_mask

    estimate_weights = torch.nn.functional.softmax(score_estimates, dim=-1, dtype=torch.float32).to(query.dtype)

    ones = torch.ones_like(estimate_weights)
    zeros = torch.zeros_like(estimate_weights)

    mask_recent = torch.triu(ones, diagonal=-int(l-1))
    mask_recent = torch.tril(mask_recent, diagonal=0)

    estimate_weights = estimate_weights + mask_recent

    top_estimate_scores, topk_indices = estimate_weights.topk(k, dim=3, largest=True)

    alpha = torch.sum(top_estimate_scores, 3, True)

    mask = torch.full_like(full_attention_scores, fill_value=float('-inf'))

    mask.scatter_(3, topk_indices, full_attention_scores.gather(3, topk_indices))
    return mask, alpha



def mask_top_k_elements_3d(tensor, k, dim=2):
    # Find the indices of the top k elements along the specified dimension
    if tensor.shape[dim] <= k:
        return tensor
    _, indices = tensor.topk(k, dim=dim, largest=True)

    # Create a mask with zeros and ones
    mask = torch.full_like(tensor, fill_value=float('-inf'))
    # mask.scatter_(dim, indices, 1)
    mask.scatter_(dim, indices, tensor.gather(dim, indices))

    # Apply the mask to the original tensor
    # masked_tensor = tensor * mask

    return mask


def test_sparq_mask():
    q_test = torch.rand(2, 2, 5, 10)
    k_test = torch.rand(2, 2, 5, 10)
    attention = torch.rand(2, 2, 5, 5)

    #print (test_tensor)
    ones = torch.ones_like(attention, dtype=torch.bool)
    tril_mask = torch.tril(ones, diagonal=0)
    attention[~tril_mask] = float('-inf')

    #print (test_tensor)

    mask_top_k_elements_sparq(attention, tril_mask, q_test, k_test, 2, 2)

def test_mask():
    test_tensor = torch.rand(1, 1, 5, 5)
    print (test_tensor)
    ones = torch.ones_like(test_tensor, dtype=torch.bool)
    tril_mask = torch.tril(ones, diagonal=0)
    test_tensor[~tril_mask] = float('-inf')
    print (test_tensor)

    test_tensor = mask_top_k_elements_3d(test_tensor, 2, dim=3)
    print (test_tensor)

if __name__ == "__main__":
    test_sparq_mask()
