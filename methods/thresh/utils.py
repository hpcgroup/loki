from typing import List, Optional, Tuple, Union
from networkx import intersection
import torch
from torch import nn
import numpy as np
import math
import os
import methods
import math
import random
import json
from scipy.optimize import curve_fit
try:
    from axonn import axonn as ax
    from axonn.intra_layer import drop
    AXONN_AVAILABLE=True
except ImportError:
    AXONN_AVAILABLE=False

# Experiment collectors
percentile_data = []
percentile_sampling_rate = 0.001

query_data = []
query_sampling_rate = 0.003

# Thresh % accumulator (set to 1 to avoid div 0 errors)
total_scores = 1
kept_scores = 1

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def thresh_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    layer_idx: int = None,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights_softmax = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
    # Shape [batch, num_heads, query_length, key_length]
    B, H, Q, K = attn_weights.shape
    
    # Collect percentile data 
    for head_idx in range(H):
        # Random sample
        if random.random() < percentile_sampling_rate:
    #        print("percentile sampled")
            for query_idx in range(Q):
                    # Because of causality, valid keys for query index i are first (i+1) elements
                    row = attn_weights_softmax[0, head_idx, query_idx, :query_idx+1]
                    p25 = torch.quantile(row.float(), 0.25).item()
                    p50 = torch.quantile(row.float(), 0.50).item()
                    p75 = torch.quantile(row.float(), 0.75).item()
                    percentile_data.append({
                        'layer_idx': layer_idx,
                        'head_idx': head_idx,
                        'query_idx': query_idx,
                        'p25': p25,
                        'p50': p50,
                        'p75': p75,
                    })

        if random.random() < query_sampling_rate: 
     #       print("query sampled")
            #Collect per query data
            sample_query_idx = random.randint(0,Q-1)
            full_row = attn_weights_softmax[0, head_idx, sample_query_idx, :sample_query_idx+1].detach().cpu().tolist()
            query_data.append({
                'layer_idx': getattr(module, 'layer_idx', -1),
                'head_idx': head_idx,
                'query_idx': sample_query_idx,
                'row_length': sample_query_idx+1,
                'attention_row': full_row,
            })
    
    # -- THRESH START --
    global total_scores, kept_scores

    # TODO: Add command args for these two
    PERCENTILE = 0.5
    WARMUP_QUERIES = 32
    
    # Collect warmup data
    x_vals = []
    y_vals = []
    for i in range(WARMUP_QUERIES):
        # Valid keys for query row i: first (i+1) elements
        row_scores = attn_weights_softmax[0, 0, i, : i+1]
        quantile_value = torch.quantile(row_scores.float(), PERCENTILE).item()
        x_vals.append(i + 1) # Use (i+1) so that x starts at 1
        y_vals.append(quantile_value)
    
    epsilon = 1e-8 # avoid log(0)
    x_vals_np = np.array(x_vals, dtype=float)
    y_vals_np = np.array(y_vals, dtype=float)
    X = np.log(x_vals_np) # x_vals already have 1 added
    Y = np.log(y_vals_np + epsilon) # add epsilon
    slope, intercept = np.polyfit(X, Y, 1)
    a = np.exp(intercept)
    b = slope

    # Generate threshold matrix
    thresholds = torch.arange(1, Q + 1, device=attn_weights_softmax.device, dtype=attn_weights_softmax.dtype)
    thresholds = a * (thresholds ** b)
    thresholds = thresholds.view(1, 1, Q, 1)  # cast size

    # Generate mask based on threshold
    keep_mask = (attn_weights_softmax >= thresholds) 
    
    # Also keep the diagonal (most recent key/value) to ensure at least one value remains.
    diag_mask = torch.eye(Q, device=attn_weights_softmax.device, dtype=torch.bool).view(1, 1, Q, Q)
    keep_mask = keep_mask | diag_mask

    # Increment num_scores and kept_scores
    total_scores += B * H * (((Q+1) * K)//2)    
    kept_scores += keep_mask.sum().item()
    
    # Create masked attention: entries not kept are set to -inf.
    attn_weights_masked = torch.full_like(attn_weights, float('-inf'))
    attn_weights_masked[keep_mask] = attn_weights[keep_mask]
    
    # -- THRESH END --

    # softmax, dropout, etc.
    attn_weights = nn.functional.softmax(attn_weights_masked, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

def save_experiment_data(args, ppl):
    global percentile_data, query_data

    filename = f"thresh_{args.model_id.split('/')[-1]}_exp"
    
    exp_data = {
        'percentile_data': percentile_data,
        'query_data': query_data,
        'model_id': args.model_id,
    }

    with open(filename + ".json", 'w') as f:
        json.dump(exp_data, f, indent=2)
    
    print(f"Experiment data saved to {filename}.json")


    global kept_scores, total_scores

    with open(filename + ".txt", 'w') as f:
        f.write('Statistics\n')
        f.write('==========\n')
        f.write(f"Compression Ratio: {(kept_scores/total_scores):.3f}\n")
        f.write(f"Perplexity: {ppl.float():.5f}\n")
    
    print(f"Experiment data saved to {filename}.txt")

