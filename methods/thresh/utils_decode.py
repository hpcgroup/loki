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
try:
    from axonn import axonn as ax
    from axonn.intra_layer import drop
    AXONN_AVAILABLE=True
except ImportError:
    AXONN_AVAILABLE=False

# Powerlaw experiment collectors
powerlaw_percentile_acc = {}
powerlaw_query_acc = {}
POWERLAW_SAMPLING_RATE = 0.075

# Retain % experiment collectors
retain_acc = []
RETAIN_SAMPLING_RATE = 0.05

# Global retain % accumulator (set to 1 to avoid div 0 errors)
total_scores = 1
kept_scores = 1

# Could probably make this into an object
global_thresh_data = {}
generation_step = 0

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

def fit_powerlaw_linreg(x: np.ndarray, y: np.ndarray):
    epsilon = 1e-8
    
    X = np.log(x + 1)
    Y = np.log(y + epsilon)
    
    slope, intercept = np.polyfit(X, Y, 1)
    
    y_pred = intercept + slope * X
    
    ss_res = np.sum((Y - y_pred)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    
    r2 = 1.0 - ss_res/ss_tot if ss_tot != 0 else 1.0
    a = np.exp(intercept)
    b = slope
        
    return a, b, r2

def collect_powerlaw_stats(attn_weights_softmax, layer_idx):
    global powerlaw_percentile_acc, powerlaw_query_acc, POWERLAW_SAMPLING_RATE

    B, H, Q, K = attn_weights_softmax.shape
    
    # Across heads
    for head_idx in range(H):
        # Random sample
        if random.random() < POWERLAW_SAMPLING_RATE:
            pvals = { "p25": [], "p50": [], "p75": [] }
            query_acc = { "a": [], "b": [], "r2": [] }

            for query_idx in range(Q):
                # Because of causality, valid keys for query index i are first (i+1) elements
                row = attn_weights_softmax[0, head_idx, query_idx, :query_idx+1]
                sorted_row, _ = torch.sort(row.float())
                
                # Calculate and get percentile data
                n = sorted_row.numel()
                
                idx25 = int(0.25 * (n - 1))
                idx50 = int(0.50 * (n - 1))
                idx75 = int(0.75 * (n - 1))
                
                p25 = sorted_row[idx25].item()
                p50 = sorted_row[idx50].item()
                p75 = sorted_row[idx75].item()

                pvals["p25"].append(p25)
                pvals["p50"].append(p50)
                pvals["p75"].append(p75)
        

                # Calculate and get per query data
                if query_idx > 1 and query_idx % 128 == 0: # Simulating start & step size 
                    x_np = np.arange(sorted_row.numel(), dtype=float)
                    y_np = sorted_row.cpu().numpy().astype(float)

                    a, b, r2 = fit_powerlaw_linreg(x_np, y_np)
                    
                    query_acc["a"].append(a) 
                    query_acc["b"].append(b)
                    query_acc["r2"].append(r2)


            # Set default dictionary path for percentile if it doesn't exist
            layer_head_key = (layer_idx, head_idx)
            if layer_head_key not in powerlaw_percentile_acc:
                powerlaw_percentile_acc[layer_head_key] = {"p25": [], "p50": [], "p75": []}
            
            # Fit each percentile and add a, b, r2 to experiment accumulator
            for perc_key in ["p25", "p50", "p75"]:
                x_np = np.arange(Q, dtype=float)
                y_np = np.array(pvals[perc_key], dtype=float)
                a, b, r2 = fit_powerlaw_linreg(x_np, y_np)
                powerlaw_percentile_acc[layer_head_key][perc_key].append((a, b, r2))
            

            # Caclulate per query data 
            a_mean = float(np.mean(np.array(query_acc["a"])))
            a_var = float(np.var(np.array(query_acc["a"])))
            b_mean = float(np.mean(np.array(query_acc["b"])))
            b_var = float(np.var(np.array(query_acc["b"])))
            r2_mean = float(np.mean(np.array(query_acc["r2"])))
            r2_var = float(np.var(np.array(query_acc["r2"])))
            
            # Set default dictionary path
            layer_head_key = (layer_idx, head_idx)
            if layer_head_key not in powerlaw_query_acc:
                powerlaw_query_acc[layer_head_key] = []
            
            # Add experiment statistics to accumulator
            powerlaw_query_acc[layer_head_key].append((a_mean, a_var, b_mean, b_var, r2_mean, r2_var))

def thresh_attention_forward(
    module: nn.Module,
    args,
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
    
    # Shape [batch, num_heads, query_length, key_length]
    B, H, Q, K = attn_weights.shape
    
    # Fix for torch bugged attention mask
    if attention_mask is None:
        attention_mask = torch.triu(
        torch.ones(Q, K, dtype=torch.float) * float('-inf'),
        diagonal=1
        ).to(attn_weights.device)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(B, H, Q, K)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    # Change commented line to set pre/post softmax
    attn_weights_thresh = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # attn_weights_thresh = attn_weights
        
    # Collect percentile and query data for powerlaw fitting
    if not args.no_json:
        collect_powerlaw_stats(attn_weights_thresh, layer_idx)
        
    # -- THRESH START --
    global total_scores, kept_scores, global_thresh_data, generation_step

    # Increment generation step (essentially will start at 1)
    if layer_idx == 0:
        generation_step += 1

    # In prefill (regular attention)
    if Q > 1:
        global_thresh_data = {}
        generation_step = 0

        # regular attention
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights
  
    # Generative decoding
    elif Q == 1: 
        # Check if we can calculate warmup
        if not global_thresh_data[layer_idx]["warmup_complete"] and generation_step > args.init_warmup:
            x_np = np.array(global_thresh_data[layer_idx]["x_vals"], dtype=float)
            y_np = np.array(global_thresh_data[layer_idx]["y_vals"], dtype=float)

            a, b, _ = fit_powerlaw_linreg(x_np, y_np)

            global_thresh_data[layer_idx]["a"] = a
            global_thresh_data[layer_idx]["b"] = b         

            global_thresh_data[layer_idx]["warmup_complete"] = True    

        # Warmup incomplete (normal_attn & collect warmup)
        elif not global_thresh_data[layer_idx]["warmup_complete"]:
            # Create layer idx specific datapoint if doesn't exist
            if layer_idx not in global_thresh_data:
                global_thresh_data[layer_idx] = {
                    "x_vals": [],
                    "y_vals": [],
                    "a": 1.0,
                    "b": -1.0,
                    "warmup_complete": False
                }

            row_scores = attn_weights_thresh[0, 0, 0, :1]
            quantile_value = torch.quantile(row_scores.float(), args.percentile).item()

            global_thresh_data[layer_idx]["x_vals"].append(generation_step)
            global_thresh_data[layer_idx]["y_vals"].append(quantile_value)

            # regular attention
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
            
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()

            return attn_output, attn_weights

        # Warmup complete (thresh_attn)
        if global_thresh_data[layer_idx]["warmup_complete"]:
            # Generate threshold matrix
            thresholds = torch.arange(1, Q + 1, device=attn_weights_thresh.device, dtype=attn_weights_thresh.dtype)
            thresholds = a * (thresholds ** b)
            thresholds = thresholds.view(1, 1, Q, 1)  # cast size

            # Generate mask based on threshold
            keep_mask = (attn_weights_thresh >= thresholds) 
            
            # Also keep the diagonal (most recent key/value) to ensure at least one value remains.
            diag_mask = torch.zeros(Q, K, device=attn_weights_thresh.device, dtype=torch.bool).view(1, 1, Q, K)
            diag_mask[0, generation_step-1] = True
            keep_mask = keep_mask | diag_mask

            # Collect keys retained data
            if not args.no_json:
                global retain_acc, RETAIN_SAMPLING_RATE
                
                for head_idx in range(H):
                    if random.random() < RETAIN_SAMPLING_RATE: # Sample across layers and heads
                        for query_idx in range(Q):
                            query_total_scores = query_idx + 1
                            query_kept_scores = keep_mask[0, head_idx, query_idx, :query_idx+1].sum().item()
                            
                            query_retain = query_kept_scores / query_total_scores

                            retain_acc.append({
                                "layer_idx": layer_idx,
                                "head_idx": head_idx,
                                "query_idx": query_idx,
                                "retain": query_retain
                            })
                
            # Increment num_scores and kept_scores
            total_scores += B * H * (((Q+1) * K)//2)    
            kept_scores += keep_mask.sum().item()

            # Create masked attention: entries not kept are set to -inf.
            attn_weights_masked = torch.full_like(attn_weights, float('-inf'))
            attn_weights_masked[keep_mask] = attn_weights[keep_mask]

            # softmax, dropout, etc.
            attn_weights = nn.functional.softmax(attn_weights_masked, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
            
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights
    # -- THRESH END --

def save_experiment_data(args, ppl):
    filename = f"thresh_{args.model_id.split('/')[-1]}_pctl{args.percentile}_iwmp{args.init_warmup}"
    global powerlaw_percentile_acc, powerlaw_query_acc, retain_acc, max_layer, max_head
    
    if not args.no_json:
        print("Calculating aggregate statistics")
                
        powerlaw_agg = {}

        def aggregate_powerlaw_fits(fit_list):
            fit_array = np.array(fit_list, dtype=float)  # shape: (N, 3)
            mean_a = float(np.mean(fit_array[:, 0]))
            var_a  = float(np.var(fit_array[:, 0]))
            mean_b = float(np.mean(fit_array[:, 1]))
            var_b  = float(np.var(fit_array[:, 1]))
            mean_r2 = float(np.mean(fit_array[:, 2]))
            var_r2  = float(np.var(fit_array[:, 2]))
            return {"mean_a": mean_a, "var_a": var_a,
                    "mean_b": mean_b, "var_b": var_b,
                    "mean_r2": mean_r2, "var_r2": var_r2}
        
        def aggregate_query_fits(fit_list):
            fit_array = np.array(fit_list, dtype=float)  # shape: (N, 6)
            mean_a = float(np.mean(fit_array[:, 0]))
            var_a  = float(np.mean(fit_array[:, 1]))  # average variance
            mean_b = float(np.mean(fit_array[:, 2]))
            var_b  = float(np.mean(fit_array[:, 3]))
            mean_r2 = float(np.mean(fit_array[:, 4]))
            var_r2  = float(np.mean(fit_array[:, 5]))
            return {"mean_a": mean_a, "var_a": var_a,
                    "mean_b": mean_b, "var_b": var_b,
                    "mean_r2": mean_r2, "var_r2": var_r2}


        # Process percentile fits from powerlaw_percentile_acc.
        for (layer_idx, head_idx), perc_dict in powerlaw_percentile_acc.items():
            key_str = f"layer_{layer_idx}_head_{head_idx}"
            
            if key_str not in powerlaw_agg:
                powerlaw_agg[key_str] = {"percentile": {}, "query": {}}
            for perc in ["p25", "p50", "p75"]:
                fits = perc_dict.get(perc, [])
                if fits:
                    powerlaw_agg[key_str]["percentile"][perc] = aggregate_powerlaw_fits(fits)

        # Process per-query fits from powerlaw_query_acc.
        for (layer_idx, head_idx), fit_list in powerlaw_query_acc.items():
            key_str = f"layer_{layer_idx}_head_{head_idx}"
            
            if key_str not in powerlaw_agg:
                powerlaw_agg[key_str] = {"percentile": {}, "query": {}}
            if fit_list:
                powerlaw_agg[key_str]["query"] = aggregate_query_fits(fit_list)
        
        # Process retain stats
        retain_agg_temp = {}
        for item in retain_acc:
            q_idx = item["query_idx"]
            if q_idx not in retain_agg_temp:
                retain_agg_temp[q_idx] = []
            retain_agg_temp[q_idx].append(item["retain"])
        
        retain_agg = {}
        for q_idx, values in retain_agg_temp.items():
            retain_agg[str(q_idx)] = {
                "mean_retain": float(np.mean(values)),
                "var_retain": float(np.var(values))
            }


        exp_data = {
            'powerlaw_data': powerlaw_agg,
            'retain_data': retain_agg,
            'model_id': args.model_id,
            'percentile': args.percentile,
            'init_warmup': args.init_warmup,
        }

        with open(filename + ".json", 'w') as f:
            json.dump(exp_data, f, indent=2)
        
        print(f"Experiment data saved to {filename}.json")

    
    if not args.no_txt:
        global kept_scores, total_scores

        with open(filename + ".txt", 'w') as f:
            f.write('Statistics\n')
            f.write(f"Model: {args.model_id}\n")
            f.write(f"Percentile: {args.percentile}\n")
            f.write(f"Warmup size: {args.init_warmup}\n")
            f.write('==========\n')
            f.write(f"Compression Ratio: {(kept_scores/total_scores):.3f}\n")
            f.write(f"Perplexity: {ppl.float():.5f}\n")
        
        print(f"Experiment data saved to {filename}.txt")

#thresh_attention_forward = torch.compile(thresh_attention_forward)