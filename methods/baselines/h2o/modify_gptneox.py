from typing import List, Optional, Tuple, Union
import math
import warnings
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, apply_rotary_pos_emb
from transformers.cache_utils import Cache
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from .external.h2o_utils import local_heavy_hitter_mask
import methods


def get_h2o_attn(args):
    def modified_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)

        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=self.norm_factor,
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)



        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask
        else:
            # Create the attention mask if it is not provided
            attention_mask = torch.where(causal_mask, torch.tensor(0.0).to(attn_scores.dtype), mask_value)

        ### Heavy + Recent
        heavy_budget = int(args.heavy_ratio * attn_scores.shape[-1])
        recent_budget = int(args.heavy_ratio * attn_scores.shape[-1])

        # Heavy Hitter Mask
        if heavy_budget > 0:
            mask_bottom = local_heavy_hitter_mask(attn_scores, heavy_budget) # Default: No padding applied to input
        else:
            mask_bottom = torch.zeros_like(attn_scores, dtype=torch.bool)

        ones = torch.ones_like(attn_scores, dtype=torch.bool)
        ones = torch.triu(ones, diagonal=-recent_budget)
        mask_bottom = torch.logical_or(mask_bottom, ones)

        mask_bottom = torch.tril(mask_bottom, diagonal=0)

        # mask_bottom = ones
        attn_scores[~mask_bottom] = torch.min(attention_mask)
        
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights
    return modified_attn

def make_gptneox_attention_h2o(args):
    print ("Modifying GPT NeoX Attention -> H2O")
    print (f"Heavy and Recent Ratio:{args.heavy_ratio}")
    GPTNeoXAttention._attn = get_h2o_attn(args)
