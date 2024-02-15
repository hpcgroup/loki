from typing import List, Optional, Tuple, Union
import math
import warnings
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb
from transformers.cache_utils import Cache
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial


base_tensor_file_path = "/pscratch/sd/p/prajwal/InferenceData/tensor_iteration_{}_{}.pt"
iteration_count = 0

#def get_h2o_forward(heavy_ratio):
#    def modified_forward(
#        self,
#        hidden_states: torch.Tensor,
#        attention_mask: Optional[torch.Tensor] = None,
#        position_ids: Optional[torch.LongTensor] = None,
#        past_key_value: Optional[Cache] = None,
#        output_attentions: bool = False,
#        use_cache: bool = False,
#        **kwargs,
#    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#        if "padding_mask" in kwargs:
#            warnings.warn(
#                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
#            )
#        bsz, q_len, _ = hidden_states.size()
#
#        if self.config.pretraining_tp > 1:
#            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
#            query_slices = self.q_proj.weight.split(
#                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
#            )
#            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
#            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
#
#            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
#            query_states = torch.cat(query_states, dim=-1)
#
#            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
#            key_states = torch.cat(key_states, dim=-1)
#
#            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
#            value_states = torch.cat(value_states, dim=-1)
#
#        else:
#            query_states = self.q_proj(hidden_states)
#            key_states = self.k_proj(hidden_states)
#            value_states = self.v_proj(hidden_states)
#
#        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#
#        kv_seq_len = key_states.shape[-2]
#        if past_key_value is not None:
#            if self.layer_idx is None:
#                raise ValueError(
#                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
#                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
#                    "with a layer index."
#                )
#            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
#        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
#
#        if past_key_value is not None:
#            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
#            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
#
#        key_states = repeat_kv(key_states, self.num_key_value_groups)
#        value_states = repeat_kv(value_states, self.num_key_value_groups)
#
#        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
#
#        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
#            raise ValueError(
#                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
#                f" {attn_weights.size()}"
#            )
#
#        if attention_mask is not None:
#            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
#                raise ValueError(
#                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
#                )
#            attn_weights = attn_weights + attention_mask
#            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
#
#        ### Heavy + Recent
#        heavy_budget = int(heavy_ratio * attn_weights.shape[-1])
#        recent_budget = int(heavy_ratio * attn_weights.shape[-1])
#
#        # Heavy Hitter Mask
#        if heavy_budget > 0:
#            mask_bottom = local_heavy_hitter_mask(attn_weights, heavy_budget) # Default: No padding applied to input
#        else:
#            mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
#
#        ones = torch.ones_like(attn_weights, dtype=torch.bool)
#        ones = torch.triu(ones, diagonal=-recent_budget)
#        mask_bottom = torch.logical_or(mask_bottom, ones)
#
#        mask_bottom = torch.tril(mask_bottom, diagonal=0)
#
#        # mask_bottom = ones
#        attn_weights[~mask_bottom] = torch.min(attention_mask)
#
#        # upcast attention to fp32
#        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
#        attn_output = torch.matmul(attn_weights, value_states)
#
#        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#            raise ValueError(
#                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                f" {attn_output.size()}"
#            )
#
#        attn_output = attn_output.transpose(1, 2).contiguous()
#
#        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
#
#        if self.config.pretraining_tp > 1:
#            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
#            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
#            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
#        else:
#            attn_output = self.o_proj(attn_output)
#
#        if not output_attentions:
#            attn_weights = None
#
#        return attn_output, attn_weights, past_key_value
#    return modified_forward

def get_s_hat_forward():
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

        global iteration_count

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

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
            key_states, key_scaling_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        key_scaling_states = repeat_kv(key_scaling_states, self.num_key_value_groups)

        zeros_count_per_key_channel = torch.sum(key_states == 0, dim=2)


        #print (zeros_count_per_key_channel)
        current_file_path = base_tensor_file_path.format(self.layer_idx, iteration_count)
        iteration_count += 1
        torch.save(zeros_count_per_key_channel, current_file_path)
        #print (key_states.size())
        #print (value_states.size())
        #print (key_scaling_states.size())

        scaling_factor = self.head_dim * (torch.abs(key_states).sum(-1, keepdim=True) / key_scaling_states)
        #scaling_factor = self.head_dim * (key_scaling_states)
        scaling_factor = scaling_factor.transpose(-1, -2)
        #print (scaling_factor.size())


        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / torch.sqrt(scaling_factor)

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

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # This does not work
        #attn_output = ((0.5) * torch.mean(value_states, 2, True)) + 0.5 * attn_output

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    return modified_forward

def make_s_hat_attention():
    print ("Modifying Llama S_hat Attention")
    LlamaAttention.forward = get_s_hat_forward()

#def make_h2o_attention(hr):
#    print ("Modifying Llama H2O Attention")
#    print (f"Ratio:{hr}")
#    LlamaAttention.forward = get_h2o_forward(hr)
