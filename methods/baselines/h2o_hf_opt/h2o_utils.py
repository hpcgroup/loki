# The function has been MODIFIED from the H2O github repo which is under the MIT Liscense
# The original file can be found at https://github.com/FMInference/H2O/blob/main/h2o_hf/utils_lm_eval/modify_llama.py 

'''
Copyright <YEAR> <COPYRIGHT HOLDER>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''

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
    #alpha = torch.sum(mask_bottom, dim=-1, keepdim=True).to(dtype_attn_weights)

    mask_bottom[:,:, :heavy_budget, :heavy_budget] = True
    #alpha[:, :, :heavy_budget, :] = 1

    zeros_index = torch.zeros_like(accumulated_attention_score, dtype=torch.bool)

    for token_index in range(heavy_budget, seq_length):
        tmp_topk_index = accumulated_attention_score.topk(k=heavy_budget - 1, dim=-1).indices

        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)

        #alpha[:,:,token_index,:] = torch.sum(tmp_attn[:,:,token_index - 1,:] * mask_bottom_index, dim=-1, keepdim=True)

        mask_bottom_index[:,:, token_index] = True 

        mask_bottom[:,:,token_index,:] = mask_bottom_index
        accumulated_attention_score += tmp_attn[:,:,token_index,:]
        accumulated_attention_score *= mask_bottom_index
      
    return mask_bottom
