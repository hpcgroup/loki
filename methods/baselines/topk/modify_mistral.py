
from typing import List, Optional, Tuple, Union
import math
import warnings
from transformers.models.mistral.modeling_mistral import MistralAttention, repeat_kv, apply_rotary_pos_emb
from transformers.models.mixtral.modeling_mixtral import MixtralAttention
from transformers.cache_utils import Cache
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from methods.common.utils import mask_attn_top_k
import methods

try:
    from axonn import axonn as ax
    from axonn.intra_layer import gather
    AXONN_AVAILABLE=True
except ImportError:
    AXONN_AVAILABLE=False

def get_top_k_forward(args):
    def modified_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if methods.G_TENSOR_SAVER is not None:
            if AXONN_AVAILABLE and ax.is_initialized:
                key_tensor_to_save = gather(key_states, transpose=True, dim=1, skip_batch=True)
                query_tensor_to_save = gather(query_states, transpose=True, dim=1, skip_batch=True) 
                value_tensor_to_save = gather(value_states, transpose=True, dim=1, skip_batch=True)
                if torch.distributed.get_rank() == 0:
                    methods.G_TENSOR_SAVER.save("key", key_tensor_to_save, self.layer_idx, "prerotary")
                    methods.G_TENSOR_SAVER.save("query", query_tensor_to_save, self.layer_idx, "prerotary")
                    methods.G_TENSOR_SAVER.save("value", value_tensor_to_save, self.layer_idx, "prerotary")
                del key_tensor_to_save
            else:
                methods.G_TENSOR_SAVER.save("key", key_states, self.layer_idx, "prerotary")
                methods.G_TENSOR_SAVER.save("query", query_states, self.layer_idx, "prerotary")
                methods.G_TENSOR_SAVER.save("value", value_states, self.layer_idx, "prerotary")

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if methods.G_TENSOR_SAVER is not None:
            if AXONN_AVAILABLE and ax.is_initialized:
                key_tensor_to_save = gather(key_states, transpose=True, dim=1, skip_batch=True)
                query_tensor_to_save = gather(query_states, transpose=True, dim=1, skip_batch=True)
                if torch.distributed.get_rank() == 0:
                    methods.G_TENSOR_SAVER.save("key", key_tensor_to_save, self.layer_idx, "postrotary")
                    methods.G_TENSOR_SAVER.save("query", query_tensor_to_save, self.layer_idx, "postrotary")
                del key_tensor_to_save
            else:
                methods.G_TENSOR_SAVER.save("key", key_states, self.layer_idx, "postrotary")
                methods.G_TENSOR_SAVER.save("query", query_states, self.layer_idx, "postrotary")

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # Get top-k attention weights
        if args.top_k <= 1:
            topk = int(args.top_k * attn_weights.shape[-1])
        else:
            topk = int(args.top_k)
        attn_weights = mask_attn_top_k(attn_weights, topk, dim=-1)
        
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    return modified_forward

# TODO: Remove use_percentage
def make_mistral_attention_top_k(args):
    print ("Modifying Mistral and Mixtral Attention -> TopK Attention")
    if args.top_k <= 1:
        print (f"TopK% - {args.top_k}")
    else:
        print (f"TopK - {args.top_k}")

    MistralAttention.forward = get_top_k_forward(args)
    MixtralAttention.forward = get_top_k_forward(args)