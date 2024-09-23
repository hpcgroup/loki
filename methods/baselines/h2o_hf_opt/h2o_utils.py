# The function has been MODIFIED from the H2O github repo which is under the MIT Liscense
# The original file can be found at https://github.com/FMInference/H2O/blob/main/h2o_hf/utils_lm_eval/modify_llama.py 

# The original H2O monkey patching code used a lot of memory and was slow for large models. 
# This is a more efficient version of the code that is used for larger models
import os
import pdb
import copy
import math
import numpy as np

import torch
from torch import nn
import time

def local_heavy_hitter_mask(attn_weights, heavy_budget):
    # attn_weights (BS, head, query, keys)
    dtype_attn_weights = attn_weights.dtype
    seq_length = attn_weights.shape[-1]

    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    accumulated_attention_score = torch.sum(tmp_attn[:,:,:heavy_budget,:], dim=-2) #(head, keys)
    
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

    mask_bottom[:,:, :heavy_budget, :heavy_budget] = True

    zeros_index = torch.zeros_like(accumulated_attention_score, dtype=torch.bool)

    for token_index in range(heavy_budget, seq_length):
        tmp_topk_index = accumulated_attention_score.topk(k=heavy_budget - 1, dim=-1).indices

        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)

        mask_bottom_index[:,:, token_index] = True 

        mask_bottom[:,:,token_index,:] = mask_bottom_index
        accumulated_attention_score += tmp_attn[:,:,token_index,:]
        accumulated_attention_score *= mask_bottom_index
      
    return mask_bottom
