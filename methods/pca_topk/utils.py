from networkx import intersection
import torch
import numpy as np
import math
import os
import methods
import math
import random
from scipy.optimize import curve_fit
try:
    from axonn import axonn as ax
    from axonn.intra_layer import drop
    AXONN_AVAILABLE=True
except ImportError:
    AXONN_AVAILABLE=False
    
import json

# Attention score sampling for plotting
# Head/ layer to collect full attention for (to plot a single query)
COLLECT_LAYER = 8
COLLECT_HEAD = 8 

MAX_SAMPLES = 100000 # Max # of samples
SAMPLES_PER_CALL = 512 # # samples collected per call of function

# Accumulators for plotting attention scores
collected_attention_data = []
collected_final_query_attention = []
collected_final_query = False

# Attention collectors for aggregate statistics
n_samples = 0

# Pre-softmax statistics
sum_pre_softmax_diff_abs = sum_pre_softmax_diff_squared = 0.0
mean_pre_softmax_full = mean_pre_softmax_approx = 0.0
M2_pre_softmax_full = M2_pre_softmax_approx = 0.0
C_pre_softmax = 0.0

# Post-softmax statistics
sum_post_softmax_diff_abs = sum_post_softmax_diff_squared = 0.0
mean_post_softmax_full = mean_post_softmax_approx = 0.0
M2_post_softmax_full = M2_post_softmax_approx = 0.0
C_post_softmax = 0.0

# Cross entropy collectors 
total_cross_entropy = 0.0
total_cross_entropy_samples = 0

# For percentile data
collected_query_percentile_data = []
collected_layer_percentile_data = []
MAX_PERCENTILE_SAMPLES = 8
percentile_samples_collected = 0
batch_num = 0
visited_percentiles = set()
collect_flag = False

# Keeping track of compression ratio
# set to 1 to avoid div by 0 (given the # of scores starting at 1 should be negligible) 
num_scores = kept_scores = 1
a_accumulator = [1]
b_accumulator = [1]

# Constants for thresholding  
WARMUP_QUERIES = 16
THRESHOLD_PERCENTILE = 0.5

PCA_DATA_PATH = "/pscratch/sd/n/nkoley/BhateleLab/approximate-attention/transform"
#PCA_DATA_PATH = "/global/cfs/cdirs/m4641/ApproxAttn/"

def get_pca_components(args, layer_idx, head_dim, top_r, num_key_value_groups, repeat_kv, device = None):
    # print (f"Getting pca components - {args.model_id}")
    model_folder_name = args.model_id.split("/")[-1] + "-PCA"
    rotary_type = args.rotary_type
    transform_dataset = args.transform_dataset

    components_file_path = os.path.join(PCA_DATA_PATH, f"{model_folder_name}/{transform_dataset}/{rotary_type}/key/pca_components/pca_components_layer_{layer_idx}.pt")
    mean_file_path = os.path.join(PCA_DATA_PATH, f"{model_folder_name}/{transform_dataset}/{rotary_type}/key/pca_means/pca_means_layer_{layer_idx}.pt")
    explained_variance_file_path = os.path.join(PCA_DATA_PATH, f"{model_folder_name}/{transform_dataset}/{rotary_type}/key/pca_explained_variance/pca_explained_variance_layer_{layer_idx}.pt")

    #methods.LOGGER.update_config({"components_file_path": os.path.dirname(components_file_path)})

    # PCA Components with the shape (num_heads, head_dim, top_r)
    if device is None:
        device = "cuda"

    pca_components = torch.load(components_file_path).to(device)

    # PCA Means with the shape (num_heads, head_dim)
    pca_means = torch.load(mean_file_path).to(device)

    # Explained Variance with the shape (num_heads, head_dim)
    pca_explained_variance = torch.load(explained_variance_file_path).to(device)

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


    # print ("{}: PCA Components Shape: {}".format(layer_idx, pca_components_r_key.shape))
    # print ("{}: PCA Means Shape: {}".format(layer_idx, pca_means.shape))
    # print ("Compression Ratio: {}".format(top_correct_r / head_dim))

    if methods.LOGGER is not None:
        methods.LOGGER.log({"compression_ratio": top_correct_r / head_dim})

    if AXONN_AVAILABLE and ax.is_initialized:
        # print ("Dropping PCA Components and PCA Means")
        ## only keep pca data for the heads on the GPU
        pca_components = drop(pca_components, transpose=True, skip_batch=True, dim=1)
        pca_means = drop(pca_means, transpose=True, skip_batch=True, dim=1)
        pca_components_r_key = drop(pca_components_r_key, transpose=True, skip_batch=True, dim=1)

    return pca_means, pca_components, pca_components_r_key

def threshold_func(x, a, b):
    return a / np.power(x+1, b)

count = 0
def mask_attn_pca_topk(args, layer_idx, attn_weights, attention_mask, query_states, key_states, pca_comps_full, pca_comps, top_r, top_k, l=-1, q_len=None, sequence_length=None):    
    global count
    global collected_attention_data, collected_final_query_attention, collected_final_query
    global n_samples
    global sum_pre_softmax_diff_abs, sum_pre_softmax_diff_squared
    global sum_post_softmax_diff_abs, sum_post_softmax_diff_squared
    global mean_pre_softmax_full, mean_pre_softmax_approx, M2_pre_softmax_full
    global M2_pre_softmax_approx, C_pre_softmax
    global mean_post_softmax_full, mean_post_softmax_approx, M2_post_softmax_full
    global M2_post_softmax_approx, C_post_softmax
    global total_cross_entropy, total_cross_entropy_samples
    global collected_query_percentile_data, collected_layer_percentile_data
    global num_scores, kept_scores
    global percentile_samples_collected, MAX_PERCENTILE_SAMPLES
    global weights_accumulator
    
    head_dim = key_states.shape[-1]
    if top_r == -1:
        top_r = head_dim

    # Default recent history = k / 4
    
    if hasattr(args, "recent_ratio"):
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

    # scaling_factor = head_dim * torch.sqrt((torch.square(key_states_sparse).sum(-1 , keepdim=True) / torch.square(key_states_pca).sum(-1, keepdim = True)))
    # scaling_factor = scaling_factor.transpose(-1, -2)
    
    # # SPARQ approach to temperature 
    # epsilon = 1e-8

    # sum_abs_query_sparse = torch.sum(torch.abs(query_states_sparse))
    # sum_abs_query_full = torch.sum(torch.abs(query_states))

    # softmaxTemp = head_dim * (sum_abs_query_sparse / (sum_abs_query_full + epsilon))
    softmaxTemp = args.temp * top_r if args.temp != -1 else head_dim
    
    # Compute attention with the query_states and key_states_sparse
    attn_weights_s_hat = torch.matmul(query_states_sparse, key_states_sparse.transpose(-1, -2)) / math.sqrt(softmaxTemp)
    if methods.LOGGER is not None:
        methods.LOGGER.update_config({"scaling_factor": "fixed"})
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights_s_hat = attn_weights_s_hat + causal_mask
        attn_weights = attn_weights + causal_mask

    s_hat = torch.nn.functional.softmax(attn_weights_s_hat, dim=-1, dtype=torch.float32).to(query_states.dtype) # Approx
    s_full = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) # Full    
    
    # ---- START OF DATA COLLECTION ----
    
    mask = causal_mask == 0
    
    # Flatten the masks & scores for aggregate analysis
    mask_flat = mask.flatten()
    attn_weights_s_hat_flat = attn_weights_s_hat.flatten()
    attn_weights_full_flat = attn_weights.flatten()
    s_hat_flat = s_hat.flatten()
    s_full_flat = s_full.flatten()
    
    # Filter out masked positions
    valid_indices = mask_flat.nonzero(as_tuple=False).squeeze()
    total_valid_elements = valid_indices.shape[0]

    global batch_num
    global collect_flag
    global visited_percentiles
    
    if layer_idx == 0:
        
        batch_num += 1
        p = 0.15 # ~MAX_PERCENTILE_SAMPLES / Batch len
        
        if percentile_samples_collected < MAX_PERCENTILE_SAMPLES and random.random() < p:
            collect_flag = True
    
    elif layer_idx == 31:
        collect_flag = False
        
    if collect_flag:
        _, heads, seq_len_X, seq_len_Y = s_hat.shape
        
        # Collect all layers @ head 0,8,16,24,31, query 4096
        for head_idx in [0,8,16,24,31]:
            if (batch_num, layer_idx, head_idx, 4095) in visited_percentiles:
                continue
            
            # Extract row scores up to the current query index to mimic causality
            row_scores = s_hat[0, head_idx, 4095, :]
            
            # Calculate the 25th, 50th (median), and 75th percentiles
            p25 = torch.quantile(row_scores.float(), 0.25).item()
            p50 = torch.quantile(row_scores.float(), 0.50).item()
            p75 = torch.quantile(row_scores.float(), 0.75).item()
            
            collected_query_percentile_data.append({
                'layer_idx': layer_idx,
                'head_idx': head_idx,
                'query_idx': 4095,
                'p25': p25,
                'p50': p50,
                'p75': p75
            })
            visited_percentiles.add((batch_num, layer_idx, head_idx, 4095))
    
        if layer_idx in [0,8,16,24,31]:
            # Collect entire query-key matrix @ head 0,8,16,24,31, layer 0,8,16,24,31
            for head_idx in [0,8,16,24,31]: 
                for query_idx in range(seq_len_Y):
                    if (batch_num, layer_idx, head_idx, query_idx) in visited_percentiles:
                        continue
                    # Extract row scores up to the current query index to mimic causality
                    row_scores = s_hat[0, head_idx, query_idx, :query_idx + 1]
                    
                    # Calculate the 25th, 50th (median), and 75th percentiles
                    p25 = torch.quantile(row_scores.float(), 0.25).item()
                    p50 = torch.quantile(row_scores.float(), 0.50).item()
                    p75 = torch.quantile(row_scores.float(), 0.75).item()
                    
                    collected_query_percentile_data.append({
                        'layer_idx': layer_idx,
                        'head_idx': head_idx,
                        'query_idx': query_idx,
                        'p25': p25,
                        'p50': p50,
                        'p75': p75
                    })
                    visited_percentiles.add((batch_num, layer_idx, head_idx, query_idx))
                    
            # Collect all heads @ query 4096, layer 0,8,16,24,31 
            for head_idx in range(heads):
                if (batch_num, layer_idx, head_idx, 4095) in visited_percentiles:
                    continue
                # Extract row scores up to the current query index to mimic causality
                row_scores = s_hat[0, head_idx, 4095, :]
                
                # Calculate the 25th, 50th (median), and 75th percentiles
                p25 = torch.quantile(row_scores.float(), 0.25).item()
                p50 = torch.quantile(row_scores.float(), 0.50).item()
                p75 = torch.quantile(row_scores.float(), 0.75).item()
                
                collected_query_percentile_data.append({
                    'layer_idx': layer_idx,
                    'head_idx': head_idx,
                    'query_idx': 4095,
                    'p25': p25,
                    'p50': p50,
                    'p75': p75
                })
                visited_percentiles.add((batch_num, layer_idx, head_idx, 4095))
                        

    # Calculate aggregate statistics on the fly using Welfords Algorithm
    if total_valid_elements > 0:
        # Convert tensors to double precision for better accuracy
        pre_softmax_full = attn_weights_full_flat[valid_indices].double()
        pre_softmax_approx = attn_weights_s_hat_flat[valid_indices].double()
        post_softmax_full = s_full_flat[valid_indices].double()
        post_softmax_approx = s_hat_flat[valid_indices].double()

        diff_pre_softmax = pre_softmax_approx - pre_softmax_full
        diff_post_softmax = post_softmax_approx - post_softmax_full

        sum_pre_softmax_diff_abs += torch.abs(diff_pre_softmax).sum().item()
        sum_pre_softmax_diff_squared += (diff_pre_softmax ** 2).sum().item()

        sum_post_softmax_diff_abs += torch.abs(diff_post_softmax).sum().item()
        sum_post_softmax_diff_squared += (diff_post_softmax ** 2).sum().item()

        batch_size = total_valid_elements

        batch_mean_pre_softmax_full = pre_softmax_full.mean().item()
        batch_mean_pre_softmax_approx = pre_softmax_approx.mean().item()

        delta_pre_full = batch_mean_pre_softmax_full - mean_pre_softmax_full
        delta_pre_approx = batch_mean_pre_softmax_approx - mean_pre_softmax_approx

        n_samples += batch_size
        mean_pre_softmax_full += delta_pre_full * batch_size / n_samples
        mean_pre_softmax_approx += delta_pre_approx * batch_size / n_samples

        M2_pre_softmax_full += ((pre_softmax_full - batch_mean_pre_softmax_full) ** 2).sum().item() + \
        delta_pre_full ** 2 * (n_samples - batch_size) * batch_size / n_samples
        
        M2_pre_softmax_approx += ((pre_softmax_approx - batch_mean_pre_softmax_approx) ** 2).sum().item() + \
        delta_pre_approx ** 2 * (n_samples - batch_size) * batch_size / n_samples

        C_pre_softmax += \
            ((pre_softmax_full - batch_mean_pre_softmax_full) * \
            (pre_softmax_approx - batch_mean_pre_softmax_approx)).sum().item() + \
            delta_pre_full * delta_pre_approx * (n_samples - batch_size) * batch_size / n_samples

        batch_mean_post_softmax_full = post_softmax_full.mean().item()
        batch_mean_post_softmax_approx = post_softmax_approx.mean().item()

        delta_post_full = batch_mean_post_softmax_full - mean_post_softmax_full
        delta_post_approx = batch_mean_post_softmax_approx - mean_post_softmax_approx

        mean_post_softmax_full += delta_post_full * batch_size / n_samples
        mean_post_softmax_approx += delta_post_approx * batch_size / n_samples

        M2_post_softmax_full += ((post_softmax_full - batch_mean_post_softmax_full) ** 2).sum().item() + \
            delta_post_full ** 2 * (n_samples - batch_size) * batch_size / n_samples
        
        M2_post_softmax_approx += ((post_softmax_approx - batch_mean_post_softmax_approx) ** 2).sum().item() + \
            delta_post_approx ** 2 * (n_samples - batch_size) * batch_size / n_samples

        C_post_softmax += \
            ((post_softmax_full - batch_mean_post_softmax_full) * \
            (post_softmax_approx - batch_mean_post_softmax_approx)).sum().item() + \
            delta_post_full * delta_post_approx * (n_samples - batch_size) * batch_size / n_samples


    # Calculate cross entropy of final query
    if q_len == sequence_length:
        epsilon = 1e-8  # To avoid log(0)
        s_hat_last = s_hat[:, :, -1, :]  # Shape: [1, layer, key, query]
        s_full_last = s_full[:, :, -1, :]  # Same shape

        # Clip s_hat_last and s_full_last to prevent log(0)
        s_hat_last_clipped = torch.clamp(s_hat_last, min=epsilon)
        s_full_last_clipped = torch.clamp(s_full_last, min=epsilon)

        # Element-wise cross-entropy
        elementwise_cross_entropy = -s_full_last_clipped * torch.log(s_hat_last_clipped)

        # Exclude NaNs and infs
        valid_mask = torch.isfinite(elementwise_cross_entropy)

        # Sum cross-entropy over valid positions
        valid_cross_entropy = elementwise_cross_entropy[valid_mask]
        valid_count = valid_cross_entropy.numel()

        if valid_count > 0:
            cross_entropy_sum = valid_cross_entropy.sum().item()
            total_cross_entropy += cross_entropy_sum
            total_cross_entropy_samples += valid_count
            
            
    # Sample attention scores to plot (exported into json, too many total scores to export all)
    if len(collected_attention_data) < MAX_SAMPLES and total_valid_elements > 0:
        num_samples = min(SAMPLES_PER_CALL, MAX_SAMPLES - len(collected_attention_data), total_valid_elements)

        # Randomly sample indices from valid positions
        sampled_indices = valid_indices[torch.randint(0, total_valid_elements, (num_samples,))]

        # Get the sampled attention scores
        pre_softmax_approx = attn_weights_s_hat_flat[sampled_indices].cpu().tolist()
        pre_softmax_full = attn_weights_full_flat[sampled_indices].cpu().tolist()
        post_softmax_approx = s_hat_flat[sampled_indices].cpu().tolist()
        post_softmax_full = s_full_flat[sampled_indices].cpu().tolist()

        # Append to collected_attention_data
        for i in range(num_samples):
            collected_attention_data.append({
                'layer_idx': layer_idx,
                'pre_softmax_approx': pre_softmax_approx[i],
                'pre_softmax_full': pre_softmax_full[i],
                'post_softmax_approx': post_softmax_approx[i],
                'post_softmax_full': post_softmax_full[i]
            })

        # Collect final query for plotting 
        if not collected_final_query:
            if layer_idx == COLLECT_LAYER:
                s_hat_last = s_hat[0, COLLECT_HEAD, -1, :] 
                s_full_last = s_full[0, COLLECT_HEAD, -1, :]
                s_hat_last_np = s_hat_last.cpu().numpy()
                s_full_last_np = s_full_last.cpu().numpy()
                if hasattr(args, 'tokens'):
                    tokens = args.tokens
                else:
                    tokens = None
                collected_final_query_attention.append({
                    'layer_idx': layer_idx,
                    's_hat_last': s_hat_last_np.tolist(),
                    's_full_last': s_full_last_np.tolist(),
                    'tokens': tokens
                })
                collected_final_query = True
    # ---- END OF DATA COLLECTION ----
    # ## Get top-k keys based on the exact attention weights
    # #i2_ground = torch.topk(attn_weights, top_k, dim=-1).indices

    # #zeros = torch.zeros_like(attn_weights)

    # #mask_predicted = zeros.scatter(-1, i2, 1)
    # #mask_ground = zeros.scatter(-1, i2_ground, 1)

    # #mask_predicted = torch.tril(mask_predicted)
    # #mask_ground = torch.tril(mask_ground)

    # #intersection = torch.logical_and(mask_predicted, mask_ground).sum(dim=-1)
    # #union = torch.logical_or(mask_ground, mask_ground).sum(dim=-1)

    # #jaccard_sim = intersection / union

    # #jaccard_sim = jaccard_sim[:,:,top_k:].mean().item()

    # #print (f"LayerId:{layer_idx}|Jaccard:{jaccard_sim}")
    
    
    # THRESHOLDING
    global WARMUP_QUERIES
    global THRESHOLD_PERCENTILE
    global a_accumulator, b_accumulator
    
    masked_attn = torch.full_like(attn_weights, float('-inf'))
    
    B, H, QX, QY = s_hat.shape
    # Increment # scores for compression calculation
    num_scores += B * H * QX * QY // 2 # Divide by 2 because of causality mask
    
    x_vals = []
    y_vals = []

    for i in range(WARMUP_QUERIES):
        row_scores = s_hat[0, 0, i, : i+1]
        quantile_value = torch.quantile(row_scores.float(), THRESHOLD_PERCENTILE).item()
        x_vals.append(i+1)
        y_vals.append(quantile_value)
    #     x_vals.append(math.log(i+1))
    #     y_vals.append(math.log(quantile_value))
    
    # if len(x_vals) >= 2:
    #     slope, intercept = np.polyfit(x_vals, y_vals, 1)
        
    #     a = math.exp(intercept)
    #     b = -slope
    # else:
    #     a = b = 1.0
    
    if len(x_vals) >= 2:
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        
        popt, pcov = curve_fit(threshold_func, x_vals, y_vals, p0=[1.0,1.0])
        a, b = popt[0], popt[1] 
    
    a_accumulator.append(a)
    b_accumulator.append(b)
    
    # Generate threshold matrix (constant)
    thresholds = torch.arange(1, QX + 1, device=s_hat.device, dtype=s_hat.dtype)  
    thresholds = a / (thresholds ** b)
    thresholds = thresholds.view(1, 1, QX, 1) # Cast size

    # Generate mask based on threshold
    keep_mask = (s_hat >= thresholds) 

    # Create and add identity matrix to mask
    # keeps most recent key/value (also prevents situation where entire row is masked)
    diag_mask = torch.eye(QX, device=s_hat.device, dtype=torch.bool)
    diag_mask = diag_mask.view(1, 1, QX, QX) 
    keep_mask = keep_mask | diag_mask

    # Increment # kept scores for compression calculation
    keep_count = keep_mask.sum().item()
    kept_scores += keep_count

    # Mask
    masked_attn[keep_mask] = attn_weights[keep_mask]

    # Calculate alpha
    is_inf = (masked_attn == float('-inf'))
    alpha_sums = s_hat.clone() 
    alpha_sums[is_inf] = 0.0
    alpha = alpha_sums.sum(dim=-1, keepdim=True)  

    return masked_attn, alpha


    # TOP-K
    # # Get the recency mask with 1s for the recent l tokens and 0 otherwise
    # ones = torch.ones_like(s_hat)
    # mask_recent = torch.triu(ones, diagonal=-int(l-1))
    # mask_recent = torch.tril(mask_recent, diagonal=0)

    # # Adding 1 to the recent token scores makes sure they are in the top-k
    # #s_hat_recent = s_hat + mask_recent
    # if (top_k >= key_states.shape[-2]):
    #     top_k = key_states.shape[-2]

    # # Get top-k keys based on the s_hat_recent score matrix
    # i2 = torch.topk(s_hat, top_k, dim=-1).indices

    # # Based on the top-k indices, change the original attn_weights to zero out the non-top-k indices
    # mask = torch.full_like(attn_weights, fill_value=float('-inf'))
    # mask.scatter_(-1, i2, attn_weights.gather(-1, i2))

    # # Caclulate alpha which is the sum of the probabilities of the top-k scores
    # alpha = torch.sum(torch.gather(s_hat, -1, i2), -1, True)
    
    # return mask, alpha


def compute_and_save_statistics(args, ppl):
    global n_samples
    global sum_pre_softmax_diff_abs, sum_pre_softmax_diff_squared
    global sum_post_softmax_diff_abs, sum_post_softmax_diff_squared
    global mean_pre_softmax_full, mean_pre_softmax_approx, M2_pre_softmax_full 
    global M2_pre_softmax_approx, C_pre_softmax
    global mean_post_softmax_full, mean_post_softmax_approx, M2_post_softmax_full
    global M2_post_softmax_approx, C_post_softmax
    global total_cross_entropy, total_cross_entropy_samples
    global num_scores, kept_scores
    global a_accumulator, b_accumulator
    global THRESHOLD_PERCENTILE

    # filename = f"{args.model_id.split('/')[-1]}_{args.rotary_type}_topr{args.top_r}_temp{args.temp}_layer{COLLECT_LAYER}_head{COLLECT_HEAD}.txt"
    filename = f"{args.model_id.split('/')[-1]}_{args.rotary_type}_topr{args.top_r}_temp{args.temp}_THRESH{THRESHOLD_PERCENTILE}.txt"
    
    with open(filename, 'w') as f:
        f.write('Statistics\n')
        f.write('==========\n')
        f.write(f'Total samples: {n_samples}\n\n')

        # Pre-softmax statistics
        mae_pre_softmax = sum_pre_softmax_diff_abs / n_samples
        mse_pre_softmax = sum_pre_softmax_diff_squared / n_samples

        variance_pre_softmax_full = M2_pre_softmax_full / (n_samples - 1) if n_samples > 1 else 0.0
        variance_pre_softmax_approx = M2_pre_softmax_approx / (n_samples - 1) if n_samples > 1 else 0.0

        covariance_pre_softmax = C_pre_softmax / (n_samples - 1) if n_samples > 1 else 0.0

        std_pre_softmax_full = math.sqrt(variance_pre_softmax_full)
        std_pre_softmax_approx = math.sqrt(variance_pre_softmax_approx)

        if std_pre_softmax_full > 0 and std_pre_softmax_approx > 0:
            r_pre_softmax = covariance_pre_softmax / (std_pre_softmax_full * std_pre_softmax_approx)
        else:
            r_pre_softmax = 0.0

        # Write to file
        f.write('Pre-softmax statistics:\n')
        f.write(f'MAE: {mae_pre_softmax:.5f}\n')
        f.write(f'MSE: {mse_pre_softmax:.5f}\n')
        f.write(f'Variance (full): {variance_pre_softmax_full:.5f}\n')
        f.write(f'Variance (approx): {variance_pre_softmax_approx:.5f}\n')
        f.write(f'Correlation coefficient (r): {r_pre_softmax:.5f}\n\n')

        # Post-softmax statistics
        mae_post_softmax = sum_post_softmax_diff_abs / n_samples
        mse_post_softmax = sum_post_softmax_diff_squared / n_samples

        variance_post_softmax_full = M2_post_softmax_full / (n_samples - 1) if n_samples > 1 else 0.0
        variance_post_softmax_approx = M2_post_softmax_approx / (n_samples - 1) if n_samples > 1 else 0.0

        covariance_post_softmax = C_post_softmax / (n_samples - 1) if n_samples > 1 else 0.0

        std_post_softmax_full = math.sqrt(variance_post_softmax_full)
        std_post_softmax_approx = math.sqrt(variance_post_softmax_approx)

        if std_post_softmax_full > 0 and std_post_softmax_approx > 0:
            r_post_softmax = covariance_post_softmax / (std_post_softmax_full * std_post_softmax_approx)
        else:
            r_post_softmax = 0.0

        # Write to file
        f.write('Post-softmax statistics:\n')
        f.write(f'MAE: {mae_post_softmax:.5f}\n')
        f.write(f'MSE: {mse_post_softmax:.5f}\n')
        f.write(f'Variance (full): {variance_post_softmax_full:.5f}\n')
        f.write(f'Variance (approx): {variance_post_softmax_approx:.5f}\n')
        f.write(f'Correlation coefficient (r): {r_post_softmax:.5f}\n\n')
        
        # Cross-entropy
        average_cross_entropy = total_cross_entropy / total_cross_entropy_samples if total_cross_entropy_samples > 0 else 0.0
        f.write(f'Average cross-entropy per final query: {average_cross_entropy:.5f}\n\n')
        
        a_accumulator = np.array(a_accumulator)
        b_accumulator = np.array(b_accumulator)
        f.write(f"Perplexity: {ppl.float():.5f}\n") 
        f.write(f"Compression Ratio: {(kept_scores/num_scores):.3f}\n")
        f.write(f"Average a weight: {a_accumulator.mean()}\n")
        f.write(f"Stdv a weight: {a_accumulator.std()}\n")
        f.write(f"Average b weight: {b_accumulator.mean()}\n")
        f.write(f"Stdv b weight: {b_accumulator.std()}\n")
        


def save_collected_attention_data(args):
    global collected_attention_data, collected_query_percentile_data
    global collected_layer_percentile_data, collected_final_query_attention
    data_to_save = {
        'rotary_type': args.rotary_type,
        'top_r': args.top_r,
        'top_k': args.top_k,
        'temp': args.temp,
        'layer': COLLECT_LAYER,
        'head': COLLECT_HEAD,
        'data': collected_attention_data,
        'query_percentile_data': collected_query_percentile_data,
        'layer_percentile_data': collected_layer_percentile_data,
        'final_query_data': collected_final_query_attention
    }
    
    # Construct filename with metadata
    # filename = f"{args.model_id.split('/')[-1]}_{args.rotary_type}_topr{args.top_r}_temp{args.temp}_layer{COLLECT_LAYER}_head{COLLECT_HEAD}.json"
    filename = f"{args.model_id.split('/')[-1]}_{args.rotary_type}_topr{args.top_r}_temp{args.temp}_THRESH{THRESHOLD_PERCENTILE}.json"
    
    
    # Dump as json
    with open(filename, 'w') as f:
        json.dump(data_to_save, f)
    print(f"Attention data saved to {filename}")