from typing import List, Optional, Tuple, Union
import math
import warnings
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, ACT2FN
from transformers.cache_utils import Cache
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from .utils import mask_attn_pca_topk, get_pca_components
import methods.pca_topk.external.gather_matmul as G
import methods.pca_topk.kernel.pca_topk as G
from methods.common.timers import Timers
import methods


try:
    from axonn import axonn as ax
    from axonn.intra_layer import drop
    AXONN_AVAILABLE=True
except ImportError:
    AXONN_AVAILABLE=False

def get_pca_forward(args):
    def modified_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

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
        
        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not hasattr(self, "pca_components"):
            _, self.pca_components, _= get_pca_components(args, self.layer_idx, self.head_dim, args.top_r, self.num_key_value_groups, repeat_kv)
            self.pca_components = self.pca_components.to(query_states.dtype)
            self.pca_components = self.pca_components.view(self.num_key_value_heads, self.head_dim, -1)

            # TODO: Keep it fixed or make it dynamic?
            if args.top_k <= 1:
                self.top_k = int(args.top_k * key_states.shape[-2])
            else:
                self.top_k = int(args.top_k)

            if methods.G_TIMERS is None:
                methods.G_TIMERS = Timers()
                

        #if query_states.shape[-2] == 1:
        #    methods.G_TIMERS.start("pca_topk_gen")
        #    methods.G_TIMERS.start("PCA_MM")

        key_states = key_states.transpose(0, 1).view(self.num_key_value_heads, -1, self.head_dim)
        query_states = query_states.transpose(0, 1).view(self.num_key_value_heads, -1, self.head_dim)

        key_states = torch.matmul(key_states, self.pca_components)
        query_states = torch.matmul(query_states, self.pca_components)

        key_states = key_states.view(self.num_key_value_heads, bsz, q_len, self.head_dim).transpose(0, 1).contiguous()
        query_states = query_states.view(self.num_key_value_heads, bsz, q_len, self.head_dim).transpose(0, 1)

        #if query_states.shape[-2] == 1:
        #    methods.G_TIMERS.stop("PCA_MM")
        #    methods.G_TIMERS.start("cache")

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        #if query_states.shape[-2] == 1:
        #    methods.G_TIMERS.stop("cache")

        # Generation Step
        if query_states.shape[-2] == 1:
            # Compute Approximate Attention Weights
            # We do not need a causal mask here since this is the generation step
            #methods.G_TIMERS.start("main_code")
            #methods.G_TIMERS.start("ApproxMM")
            attn_weights = torch.matmul(query_states[:,:,:,:args.top_r], key_states[:,:,:,:args.top_r].transpose(2, 3))
            #methods.G_TIMERS.stop("ApproxMM")
            #/ math.sqrt(self.head_dim)

            #methods.G_TIMERS.start("TopKFind")
            key_states_topk_indices = torch.topk(attn_weights, self.top_k, dim=-1, sorted=False).indices.to("cuda")
            #key_states_topk_indices , _ = torch.sort(key_states_topk_indices, dim=-1)
            key_states_topk_indices = key_states_topk_indices.reshape(-1, key_states_topk_indices.shape[-1])
            #methods.G_TIMERS.stop("TopKFind")

            #methods.G_TIMERS.start("OptKernel1")
            key_states = key_states.view(-1, key_states.shape[-2], key_states.shape[-1])
            query_states = query_states.reshape(-1, query_states.shape[-2], query_states.shape[-1])

            attn_weights = G.gather_outer_bmv_optimized(
                query_states,
                key_states.transpose(-1, -2),
                key_states_topk_indices,
            ) / math.sqrt(self.head_dim)
            #methods.G_TIMERS.stop("OptKernel1")
            

            #methods.G_TIMERS.start("OptKernel2")
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

            value_states = value_states.reshape(-1, value_states.shape[-2], value_states.shape[-1])
            attn_output = G.gather_inner_matrix_only_bmv_optimized(
              attn_weights, 
              value_states, 
              key_states_topk_indices, 
            )
            attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)
            #methods.G_TIMERS.stop("OptKernel2")
            #methods.G_TIMERS.stop("main_code")
        else:
            # Compute Standard Attention
            attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3))) / math.sqrt(self.head_dim)
        
            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

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

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        #if query_states.shape[-2] == 1:
        #    methods.G_TIMERS.stop("pca_topk_gen")

        return attn_output, attn_weights, past_key_value
    return modified_forward

def make_llama_attention_pca_topk(args):
    print ("Modifying Llama Attention -> PCA TopK Attention")
    print ("Top R:", args.top_r)
    print ("Top K:", args.top_k)
    #if args.optimised:
    print ("Optimised PCA TopK Attention")
    #LlamaAttention.__init__ = get_pca_init(top_r, top_k)
    LlamaAttention.forward = get_pca_forward(args)
