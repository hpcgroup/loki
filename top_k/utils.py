import torch
import math



def mask_top_k_elements_sparq(attn_weights, attention_mask, query_states, key_states, r, k, l = -1):
    dh = key_states.shape[-1]
    s = key_states.shape[-2]

    # Default recent history = k / 4
    if l == -1:
        l = k / 4

    # Get the top-r absolute values of the keys along the dh dimension
    i1 = torch.topk(torch.abs(key_states), r, -1).indices

    # Zero out all indices other than the top-r (TODO: Make this an actual sparse matrix)
    key_states_sparse = torch.full_like(key_states, fill_value=0)
    key_states_sparse.scatter_(-1, i1, key_states.gather(-1, i1))

    # Scaling factor based on the SPAR-Q paper. Edited it to work with keys
    scaling_factor = dh * (torch.abs(key_states_hat).sum(-1 , keepdim=True) / torch.abs(key_states).sum(-1, keepdim = True))

    # Compute attention with the query_states and key_states_sparse
    s_hat = torch.matmul(query_states, key_states_sparse.transpose(-1, -2)) / torch.sqrt(scaling_factor)
    s_hat = s_hat + attention_mask
    s_hat = torch.nn.functional.softmax(s_hat, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Get the recency mask with 1s for the recent l tokens and 0 otherwise
    ones = torch.ones_like(s_hat)
    mask_recent = torch.triu(ones, diagonal=-int(l-1))
    mask_recent = torch.tril(mask_recent, diagonal=0)

    # Adding 1 to the recent token scores makes sure they are in the top-k
    s_hat_recent = s_hat + mask_recent

    if (k >= key_states.shape[2]):
        k = key_states.shape[2]

    # Get top-k keys based on the s_hat_recent score matrix
    i2 = torch.topk(s_hat_recent, k, dim=-1).indices

    # Based on the top-k indices, change the original attn_weights to zero out the non-top-k indices
    mask = torch.full_like(attn_weights, fill_value=float('-inf'))
    mask.scatter_(-1, i2, attn_weights.gather(-1, i2))

    # Caclulate alpha which is the sum of the probabilities of the top-k scores
    alpha = torch.sum(torch.gather(s_hat, -1, i2), -1, True)

    return mask, alpha

def mask_top_k_elements_sparq(full_attention_scores, attention_mask, query, key, value, r, k):

    
    dh = query.shape[-1]
    l = k / 4

    i1 = torch.topk(torch.abs(query), r, -1).indices
    #i1,_ = torch.sort(i1, dim=-1 )
    #print (f"I1:\n{i1}")

    query_hat = torch.gather(query, -1, i1)
    key_hat = torch.gather(key, -1, i1)

    #print (f"Qhat:\n{query_hat}")
    #print (f"Khat:\n{key_hat}")

    q_sparse = torch.full_like(query, fill_value=0)
    q_sparse.scatter_(-1, i1, query.gather(-1, i1))

    s = key.shape[2]

    scaling_factor = query.shape[-1] * (torch.abs(key_hat).sum(-1 , keepdim=True) / torch.abs(key).sum(-1, keepdim = True))


    s_hat = torch.matmul(q_sparse, key.transpose(-1, -2)) / torch.sqrt(scaling_factor)
    s_hat = s_hat + attention_mask
    #s_hat[~attention_mask] = float('-inf')

    s_hat = torch.nn.functional.softmax(s_hat, dim=-1, dtype=torch.float32).to(query.dtype)
    #print (f"Shat:\n{s_hat}")

    ones = torch.ones_like(s_hat)
    mask_recent = torch.triu(ones, diagonal=-int(l-1))
    mask_recent = torch.tril(mask_recent, diagonal=0)

    s_hat = s_hat + mask_recent
    #print (f"Shat:\n{s_hat}")

    if (k >= key.shape[2]):
        k = key.shape[2]

    i2 = torch.topk(s_hat, k, dim=-1).indices
    #i2,_ = torch.sort(i2, dim=-1)

    s_hat = s_hat - mask_recent

    mask = torch.full_like(full_attention_scores, fill_value=float('-inf'))
    mask.scatter_(-1, i2, full_attention_scores.gather(-1, i2))

    alpha = torch.sum(torch.gather(s_hat, -1, i2), -1, True)
    #print (f"Mask:\n{mask}")
    return mask, alpha
#def mask_top_k_elements_sparq(full_attention_scores, attention_mask, query, key, value, r, k):
#    
#    dh = query.shape[3]
#    l = k / 4
#
#    i1 = torch.topk(torch.abs(query), r, -1).indices
#
#    #abs_top_q_r, indices = torch.abs(query).topk(r, dim=3, largest=True)
#    query_top_r = torch.gather(query, 3, indices)
#
#    s = key.shape[2]
#
#    scaling_factor = dh * (torch.sum(abs_top_q_r, 3, keepdim=True) / torch.sum(torch.abs(query), 3, keepdim = True))
#
#    key_top_r = torch.gather(key, 3, indices)
#
#    score_estimates = torch.matmul(query_top_r, key_top_r.transpose(2, 3)) / torch.sqrt(scaling_factor)
#
#    score_estimates = score_estimates + attention_mask
#
#    estimate_weights = torch.nn.functional.softmax(score_estimates, dim=-1, dtype=torch.float32).to(query.dtype)
#
#    ones = torch.ones_like(estimate_weights)
#    zeros = torch.zeros_like(estimate_weights)
#
#    mask_recent = torch.triu(ones, diagonal=-int(l-1))
#    mask_recent = torch.tril(mask_recent, diagonal=0)
#
#    modified_weights = estimate_weights + mask_recent
#
#    top_estimate_scores, topk_indices = modified_weights.topk(k, dim=3, largest=True)
#
#    alpha = torch.sum(torch.gather(estimate_weights, 3, topk_indices), 3, True)
#
#    mask = torch.full_like(full_attention_scores, fill_value=float('-inf'))
#
#    mask.scatter_(3, topk_indices, full_attention_scores.gather(3, topk_indices))
#    return mask, alpha
#    #print (mask)
#    #print (alpha)
#
#    attn_weights = torch.nn.functional.softmax(mask, dim=-1, dtype=torch.float32).to(query.dtype)
#    #print (attn_weights)
#
#    attn_output = torch.matmul(attn_weights, value)
#    #print (attn_output)
#
#    #print (torch.mean(value, 2, True))
#    #print (torch.mean(value, 2, True).shape)
#
#    #print (1-alpha)
#    #print (torch.matmul(1-alpha, torch.mean(value, 2, True)))
#



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
    q_test = torch.rand(1, 1, 5, 6)
    k_test = torch.rand(1, 1, 5, 6)
    v_test = torch.rand(1, 1, 5, 6)
    #print (f"Q:\n{q_test}")
    #print (f"K:\n{k_test}")
    #print (f"V:\n{v_test}")
    attention = torch.matmul(q_test, k_test.transpose(2,3))/math.sqrt(10)

    ##print (test_tensor)
    ones = torch.ones_like(attention, dtype=torch.bool)
    tril_mask = torch.tril(ones, diagonal=0)
    attention[~tril_mask] = float('-inf')

    #print (f"FullAttention:\n{attention}")

    ##print (test_tensor)

    mask_top_k_elements_sparq(attention, tril_mask, q_test, k_test, v_test, 2, 2)

def test_mask():
    test_tensor = torch.rand(1, 1, 5, 5)
    #print (test_tensor)
    ones = torch.ones_like(test_tensor, dtype=torch.bool)
    tril_mask = torch.tril(ones, diagonal=0)
    test_tensor[~tril_mask] = float('-inf')
    #print (test_tensor)

    test_tensor = mask_top_k_elements_3d(test_tensor, 2, dim=3)
    #print (test_tensor)

if __name__ == "__main__":
    test_sparq_mask()
