from typing import List, Optional, Tuple, Union
import math
import warnings
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, apply_rotary_pos_emb
from transformers.cache_utils import Cache
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from methods.common.utils import mask_attn_top_k
import methods

def get_topk_init(top_k):
    def modified_attention_init(self, config):
        super(GPTNeoXAttention, self).__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
            )
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self._init_bias(config.max_position_embeddings)

        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)
        self._init_rope()

        self.norm_factor = self.head_size**-0.5
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.is_causal = True

        self.layer_idx = methods.G_TENSOR_SAVER.get_layer_idx()
    return modified_attention_init



def get_top_k_forward(top_k, use_percentage=False):
    def modified_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        if methods.G_TENSOR_SAVER is not None:
            methods.G_TENSOR_SAVER.save("key", key, self.layer_idx, "prerotary")
            methods.G_TENSOR_SAVER.save("query", query, self.layer_idx, "prerotary")
            methods.G_TENSOR_SAVER.save("value", value, self.layer_idx, "prerotary")

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # TODO: Implement top-k scheme

        # Reshape outputs
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        if methods.G_TENSOR_SAVER is not None:
            methods.G_TENSOR_SAVER.save("key", key, self.layer_idx, "postrotary")
            methods.G_TENSOR_SAVER.save("query", query, self.layer_idx, "postrotary")

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    return modified_forward

def make_gptneox_attention_top_k(top_k, use_percentage=False):
    print ("Modifying GPT Neo X Attention -> TopK Attention")
    if not use_percentage:
        print (f"TopK - {top_k}")
    else:
        print (f"TopK% - {top_k}")

    GPTNeoXAttention.forward = get_top_k_forward(top_k, use_percentage)
    GPTNeoXAttention.__init__ = get_topk_init(top_k)
