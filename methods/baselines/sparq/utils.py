import torch
import math

def mask_attn_sparq(attn_weights, attention_mask, query_states, key_states, top_r, top_k, l=-1):
    head_dim = query_states.shape[-1]
    if top_r == -1:
        top_r = head_dim

    # Default recent history = k / 4
    if l == -1:
        l = top_k / 4

    # Get the top-r absolute values of the query along the dh dimension
    i1 = torch.topk(torch.abs(query_states), top_r, -1).indices

    # Zero out all indices other than the top-r (TODO: Make this an actual sparse matrix)
    query_states_sparse = torch.full_like(query_states, fill_value=0)
    query_states_sparse.scatter_(-1, i1, query_states.gather(-1, i1))

    # Scaling factor based on the SPAR-Q paper. Edited it to work with keys
    scaling_factor = head_dim * (torch.abs(query_states_sparse).sum(-1 , keepdim=True) / torch.abs(query_states).sum(-1, keepdim = True))

    # Compute attention with the query_states and key_states_sparse
    attn_weights_s_hat = torch.matmul(query_states_sparse, key_states.transpose(-1, -2)) / torch.sqrt(scaling_factor)
    attn_weights_s_hat = attn_weights_s_hat + attention_mask

    s_hat = torch.nn.functional.softmax(attn_weights_s_hat, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Get the recency mask with 1s for the recent l tokens and 0 otherwise
    ones = torch.ones_like(s_hat)
    mask_recent = torch.triu(ones, diagonal=-int(l-1))
    mask_recent = torch.tril(mask_recent, diagonal=0)

    # Adding 1 to the recent token scores makes sure they are in the top-k
    s_hat_recent = s_hat + mask_recent

    if (top_k >= key_states.shape[2]):
        top_k = key_states.shape[2]

    # Get top-k keys based on the s_hat_recent score matrix
    i2 = torch.topk(s_hat_recent, top_k, dim=-1).indices

    # Based on the top-k indices, change the original attn_weights to zero out the non-top-k indices
    mask = torch.full_like(attn_weights, fill_value=float('-inf'))
    mask.scatter_(-1, i2, attn_weights.gather(-1, i2))

    # Caclulate alpha which is the sum of the probabilities of the top-k scores
    alpha = torch.sum(torch.gather(s_hat, -1, i2), -1, True)

    return mask, alpha

def test_spar_mask():
    q_test = torch.rand(1, 1, 5, 6)
    k_test = torch.rand(1, 1, 5, 6)
    v_test = torch.rand(1, 1, 5, 6)
    print (f"Q:\n{q_test}")
    print (f"K:\n{k_test}")
    print (f"V:\n{v_test}")
    attention = torch.matmul(q_test, k_test.transpose(2,3))/math.sqrt(10)

    ##print (test_tensor)
    ones = torch.ones_like(attention, dtype=torch.bool)
    tril_mask = torch.tril(ones, diagonal=0)
    attention[~tril_mask] = float('-inf')

    # Attention mask
    attention_mask = torch.zeros_like(attention, dtype=torch.bool)
    tril_mask = torch.tril(ones, diagonal=0)
    attention_mask[~tril_mask] = float('-inf')

    print (f"FullAttention:\n{attention}")

    ##print (test_tensor)

    spar_attn, alpha = mask_attn_sparq(attention, attention_mask, q_test, k_test, 2, 2, -1)
    print (spar_attn)

if __name__ == "__main__":
    test_spar_mask()