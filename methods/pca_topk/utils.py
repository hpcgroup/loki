from networkx import intersection
import torch
import math
import os
import methods
try:
    from axonn import axonn as ax
    from axonn.intra_layer import drop
    AXONN_AVAILABLE=True
except ImportError:
    AXONN_AVAILABLE=False

#PCA_DATA_PATH = "/pscratch/sd/p/prajwal/InferenceData"
PCA_DATA_PATH = "/global/cfs/cdirs/m4641/ApproxAttn/"

def get_pca_components(args, layer_idx, head_dim, top_r, num_key_value_groups, repeat_kv):
    model_folder_name = args.model_id.split("/")[-1] + "-PCA"
    rotary_type = args.rotary_type

    components_file_path = os.path.join(PCA_DATA_PATH, f"{model_folder_name}/wikitext/{rotary_type}/key/pca_components/pca_components_layer_{layer_idx}.pt")
    mean_file_path = os.path.join(PCA_DATA_PATH, f"{model_folder_name}/wikitext/{rotary_type}/key/pca_means/pca_means_layer_{layer_idx}.pt")
    explained_variance_file_path = os.path.join(PCA_DATA_PATH, f"{model_folder_name}/wikitext/{rotary_type}/key/pca_explained_variance/pca_explained_variance_layer_{layer_idx}.pt")

    #methods.LOGGER.update_config({"components_file_path": os.path.dirname(components_file_path)})

    # PCA Components with the shape (num_heads, head_dim, top_r)
    pca_components = torch.load(components_file_path).to("cuda")

    # PCA Means with the shape (num_heads, head_dim)
    pca_means = torch.load(mean_file_path).to("cuda")

    # Explained Variance with the shape (num_heads, head_dim)
    pca_explained_variance = torch.load(explained_variance_file_path).to("cuda")

    # Reshaping the components and taking a transpose to have components along the column dimension and means to be easily broadcastable over the keys
    pca_components = pca_components.reshape(1, -1, head_dim, head_dim).transpose(2, 3)
    pca_means = pca_means.reshape(1, -1, 1, head_dim)

    # Get the point where the explained variance is 95% per head
    explained_variance_cumsum = pca_explained_variance.cumsum(-1)


    if top_r < 1:
        # Find the maximum index where the explained variance is 95% across all heads - Uncomment this line adaptively set the top_r:w
        top_correct_r = (explained_variance_cumsum < top_r).sum(-1).max().item()

    #    # Instead of sum, we use the median index 
    #    #top_r = (explained_variance_cumsum < 0.95).sum(-1).median().item()
    else:
        top_correct_r = int(top_r)

    # Only keep the top_r components of the pca_components
    pca_components_r_key = pca_components[:, :, :, :top_correct_r]
    
    if repeat_kv is not None:
        pca_components_r_key = repeat_kv(pca_components_r_key, num_key_value_groups)
        pca_components = repeat_kv(pca_components, num_key_value_groups)


    print ("{}: PCA Components Shape: {}".format(layer_idx, pca_components_r_key.shape))
    print ("{}: PCA Means Shape: {}".format(layer_idx, pca_means.shape))
    print ("Compression Ratio: {}".format(top_correct_r / head_dim))

    #methods.LOGGER.update_config({"Compression Ratio": top_correct_r / head_dim})

    if AXONN_AVAILABLE and ax.is_initialized:
        print ("Dropping PCA Components and PCA Means")
        ## only keep pca data for the heads on the GPU
        pca_components = drop(pca_components, transpose=True, skip_batch=True, dim=1)
        pca_means = drop(pca_means, transpose=True, skip_batch=True, dim=1)
        pca_components_r_key = drop(pca_components_r_key, transpose=True, skip_batch=True, dim=1)

    return pca_means, pca_components, pca_components_r_key


def mask_attn_pca_topk(args, layer_idx, attn_weights, attention_mask, query_states, key_states, pca_comps_full, pca_comps, top_r, top_k, l=-1):
    head_dim = key_states.shape[-1]
    if top_r == -1:
        top_r = head_dim

    # Default recent history = k / 4
    if args.recent_ratio == -1:
      l = 0
    else:
      l = int(args.recent_ratio * key_states.shape[-2])

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
    methods.LOGGER.update_config({"scaling_factor": "ratio-based"})

    # Compute attention with the query_states and key_states_sparse
    attn_weights_s_hat = torch.matmul(query_states_sparse, key_states_sparse.transpose(-1, -2)) / torch.sqrt(scaling_factor)
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights_s_hat = attn_weights_s_hat + causal_mask

    s_hat = torch.nn.functional.softmax(attn_weights_s_hat, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Get the recency mask with 1s for the recent l tokens and 0 otherwise
    #ones = torch.ones_like(s_hat)
    #mask_recent = torch.triu(ones, diagonal=-int(l-1))
    #mask_recent = torch.tril(mask_recent, diagonal=0)

    # Adding 1 to the recent token scores makes sure they are in the top-k
    #s_hat_recent = s_hat + mask_recent

    if (top_k >= key_states.shape[-2]):
        top_k = key_states.shape[-2]

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
