from networkx import intersection
import torch
import math

def mask_attn_pca_topk(layer_idx, attn_weights, attention_mask, query_states, key_states, pca_comps_full, pca_comps, top_r, top_k, l=-1):
    head_dim = key_states.shape[-1]
    if top_r == -1:
        top_r = head_dim

    # Default recent history = k / 4
    if l == -1:
        l = top_k / 4
        #l = 0

    # Transform key_states and query_states to PCA space
    #key_states_pca = torch.matmul(key_states, pca_comps_full).to(query_states.dtype)
    #key_states_full = key_states.to(pca_comps.dtype)
    #query_states_full = query_states.to(pca_comps.dtype)

    key_states_pca = torch.matmul(key_states, pca_comps_full).to(query_states.dtype)
    key_states_sparse = torch.matmul(key_states, pca_comps).to(query_states.dtype)
    query_states_sparse = torch.matmul(query_states, pca_comps).to(query_states.dtype)

    #key_states_reconstructed = torch.matmul(key_states_sparse, pca_comps.transpose(-1, -2)).to(query_states.dtype)
    #query_states_reconstructed = torch.matmul(query_states_sparse, pca_comps.transpose(-1, -2)).to(query_states.dtype)

    scaling_factor = head_dim * torch.sqrt((torch.square(key_states_sparse).sum(-1 , keepdim=True) / torch.square(key_states_pca).sum(-1, keepdim = True)))
    scaling_factor = scaling_factor.transpose(-1, -2)

    # Compute attention with the query_states and key_states_sparse
    attn_weights_s_hat = torch.matmul(query_states_sparse, key_states_sparse.transpose(-1, -2)) / torch.sqrt(scaling_factor)
    attn_weights_s_hat = attn_weights_s_hat + attention_mask

    s_hat = torch.nn.functional.softmax(attn_weights_s_hat, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Get the recency mask with 1s for the recent l tokens and 0 otherwise
    #ones = torch.ones_like(s_hat)
    #mask_recent = torch.triu(ones, diagonal=-int(l-1))
    #mask_recent = torch.tril(mask_recent, diagonal=0)

    # Adding 1 to the recent token scores makes sure they are in the top-k
    #s_hat_recent = s_hat + mask_recent

    if (top_k >= key_states.shape[2]):
        top_k = key_states.shape[2]

    # Get top-k keys based on the s_hat_recent score matrix
    i2 = torch.topk(attn_weights_s_hat, top_k, dim=-1).indices

    ## Get top-k keys based on the exact attention weights
    #i2_ground = torch.topk(attn_weights, top_k, dim=-1).indices

    #zeros = torch.zeros_like(attn_weights)

    #mask_predicted = zeros.scatter(-1, i2, 1)
    #mask_ground = zeros.scatter(-1, i2_ground, 1)

    #mask_predicted = torch.tril(mask_predicted)
    #mask_ground = torch.tril(mask_ground)

    #intersection = torch.logical_and(mask_predicted, mask_ground).sum(dim=-1)
    #union = torch.logical_or(mask_ground, mask_ground).sum(dim=-1)

    #jaccard_sim = intersection / union

    #jaccard_sim = jaccard_sim[:,:,top_k:].mean().item()

    #print (f"LayerId:{layer_idx}|Jaccard:{jaccard_sim}")

    # Based on the top-k indices, change the original attn_weights to zero out the non-top-k indices
    mask = torch.full_like(attn_weights, fill_value=float('-inf'))
    mask.scatter_(-1, i2, attn_weights.gather(-1, i2))

    # Caclulate alpha which is the sum of the probabilities of the top-k scores
    alpha = torch.sum(torch.gather(s_hat, -1, i2), -1, True)


    return mask, alpha
