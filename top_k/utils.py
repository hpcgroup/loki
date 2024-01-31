import torch
import math

def get_top_r(t, r, dim = -1):
    # Get the top-r absolute values of the keys along the dh dimension
    i1 = torch.topk(torch.abs(t), r, dim).indices

    # Zero out all indices other than the top-r 
    # TODO: Make this an actual sparse matrix
    t_sparse = torch.full_like(t, fill_value=0)
    t_sparse.scatter_(dim, i1, t.gather(dim, i1))

    return t_sparse

def mask_elements_spar_k(attn_weights, attention_mask, query_states, key_states, r, k, l = -1, return_shat = False):
    dh = key_states.shape[-1]

    if r == -1:
        r = dh

    # Default recent history = k / 4
    if l == -1:
        l = k / 4

    # Get the top-r absolute values of the keys along the dh dimension
    i1 = torch.topk(torch.abs(key_states), r, -1).indices

    # Zero out all indices other than the top-r (TODO: Make this an actual sparse matrix)
    key_states_sparse = torch.full_like(key_states, fill_value=0)
    key_states_sparse.scatter_(-1, i1, key_states.gather(-1, i1))

    # Scaling factor based on the SPAR-Q paper. Edited it to work with keys
    scaling_factor = dh * (torch.abs(key_states_sparse).sum(-1 , keepdim=True) / torch.abs(key_states).sum(-1, keepdim = True))
    scaling_factor = scaling_factor.transpose(-1, -2)

    # Compute attention with the query_states and key_states_sparse
    attn_weights_s_hat = torch.matmul(query_states, key_states_sparse.transpose(-1, -2)) / torch.sqrt(scaling_factor)
    #attn_weights_s_hat = attn_weights_s_hat + attention_mask

    #attn_weights_s_hat[~attention_mask] =  float('-inf')
    #test_tensor[~tril_mask] = float('-inf')
    if return_shat:
        return attn_weights_s_hat, 1 

    s_hat = torch.nn.functional.softmax(attn_weights_s_hat, dim=-1, dtype=torch.float32).to(query_states.dtype)

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

def mask_elements_spar_q(attn_weights, attention_mask, query_states, key_states, r, k, l = -1, return_shat = False):
    dh = key_states.shape[-1]

    if r == -1:
        r = dh

    # Default recent history = k / 4
    if l == -1:
        l = k / 4

    # Get the top-r absolute values of the query along the dh dimension
    i1 = torch.topk(torch.abs(query_states), r, -1).indices
    #i1,_ = torch.sort(i1, dim=-1 )
    #print (f"I1:\n{i1}")

    # Zero out all indices other than the top-r (TODO: Make this an actual sparse matrix)
    query_states_sparse = torch.full_like(query_states, fill_value=0)
    query_states_sparse.scatter_(-1, i1, query_states.gather(-1, i1))

    # Scaling factor based on the SPAR-Q paper. Edited it to work with keys
    scaling_factor = dh * (torch.abs(query_states_sparse).sum(-1 , keepdim=True) / torch.abs(query_states).sum(-1, keepdim = True))

    # Compute attention with the query_states and key_states_sparse
    attn_weights_s_hat = torch.matmul(query_states_sparse, key_states.transpose(-1, -2)) / torch.sqrt(scaling_factor)
    attn_weights_s_hat = attn_weights_s_hat + attention_mask
    #attn_weights_s_hat[~attention_mask] =  float('-inf')
    if return_shat:
        return attn_weights_s_hat, 1 

    s_hat = torch.nn.functional.softmax(attn_weights_s_hat, dim=-1, dtype=torch.float32).to(query_states.dtype)

    #print (f"s_hat:\n{s_hat}")

    # Get the recency mask with 1s for the recent l tokens and 0 otherwise
    ones = torch.ones_like(s_hat)
    mask_recent = torch.triu(ones, diagonal=-int(l-1))
    mask_recent = torch.tril(mask_recent, diagonal=0)

    # Adding 1 to the recent token scores makes sure they are in the top-k
    s_hat_recent = s_hat + mask_recent
    #print (f"s_hat_recent:\n{s_hat}")

    if (k >= key_states.shape[2]):
        k = key_states.shape[2]

    # Get top-k keys based on the s_hat_recent score matrix
    i2 = torch.topk(s_hat_recent, k, dim=-1).indices

    # Based on the top-k indices, change the original attn_weights to zero out the non-top-k indices
    mask = torch.full_like(attn_weights, fill_value=float('-inf'))
    mask.scatter_(-1, i2, attn_weights.gather(-1, i2))

    # Caclulate alpha which is the sum of the probabilities of the top-k scores
    alpha = torch.sum(torch.gather(s_hat, -1, i2), -1, True)

    #print (f"Mask:\n{mask}")
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


def test_spar_mask():
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

    print (f"FullAttention:\n{attention}")

    ##print (test_tensor)

    #mask_elements_spar_q(attention, tril_mask, q_test, k_test, v_test, 2, 2)
    spar_attn, alpha = mask_elements_spar_q(attention, tril_mask, q_test, k_test, 2, 2, -1, True)
    print (spar_attn)

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
    test_spar_mask()
