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
import numpy as np

import methods

#def rotate_half(x):
#    """Rotates half the hidden dims of the input."""
#    x1 = x[..., : x.shape[-1] // 2]
#    x2 = x[..., x.shape[-1] // 2 :]
#    return torch.cat((-x2, x1), dim=-1)
#
#def rotate_other_half(x):
#    """Rotates half the hidden dims of the input."""
#    x1 = x[..., : x.shape[-1] // 2]
#    x2 = x[..., x.shape[-1] // 2 :]
#    return torch.cat((x2, x1), dim=-1)
#
#def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
#    """Applies Rotary Position Embedding to the query and key tensors.
#
#    Args:
#        q (`torch.Tensor`): The query tensor.
#        k (`torch.Tensor`): The key tensor.
#        cos (`torch.Tensor`): The cosine part of the rotary embedding.
#        sin (`torch.Tensor`): The sine part of the rotary embedding.
#        position_ids (`torch.Tensor`):
#            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
#            used to pass offsetted position ids when working with a KV-cache.
#        unsqueeze_dim (`int`, *optional*, defaults to 1):
#            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#    Returns:
#        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#    """
#    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
#    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
#
#    q_t = q.transpose(-2, -3)
#    cost = cos.transpose(-2, -3)
#    cost = cost.squeeze(-2)
#    cost = cost.unsqueeze(-1) * torch.eye(128).to(cost.device)
#
#    sint = sin.transpose(-2, -3)
#    sint = sint.squeeze(-2)
#
#    sint_1 = torch.diag_embed(sint[:,:,:64], offset=64)
#    sint_2 = torch.diag_embed(sint[:,:,64:], offset=-64)
#    sint = sint_1 - sint_2
#
#    R = cost + sint
#    R = R.to(torch.float64)
#
#    Rrt = torch.matmul(R, R.transpose(-2, -1))
#    q_t = q_t.to(torch.float64)
#
#    q_embed_new = torch.matmul(q_t, Rrt)
#    q_embed_new = q_embed_new.transpose(-2, -3)
#    q_embed_new = q_embed_new.to(q.dtype)
#
#    q_embed = (q * cos) + (rotate_half(q) * sin)
#
#    #torch.testing.assert_allclose(q_embed, q_embed_new)
#
#    #q_embed = (q_embed * cos) + (rotate_other_half(q_embed) * sin)
#    
#    #q_embed_new = (q_embed * cos) - (rotate_half(q_embed) * rotate_other_half(sin))
#    k_embed = (k * cos) + (rotate_half(k) * sin)
#    return q_embed, q_embed_new, k_embed, k

def get_pca_init(top_r):
    def modified_attention_init(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super(LlamaAttention, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
          
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

        # Initialise PCA transforms
        components_file_path = "/pscratch/sd/p/prajwal/InferenceData/Llama2-7B-PCA/wikitext/postrotary/key/pca_components/pca_components_layer_{}.pt".format(layer_idx)
        mean_file_path = "/pscratch/sd/p/prajwal/InferenceData/Llama2-7B-PCA/wikitext/postrotary/key/pca_means/pca_means_layer_{}.pt".format(layer_idx)
        explained_variance_file_path = "/pscratch/sd/p/prajwal/InferenceData/Llama2-7B-PCA/wikitext/postrotary/key/pca_explained_variance/pca_explained_variance_layer_{}.pt".format(layer_idx)

        # PCA Components with the shape (num_heads, head_dim, top_r)
        self.pca_components = torch.load(components_file_path).to("cuda")

        # PCA Means with the shape (num_heads, head_dim)
        self.pca_means = torch.load(mean_file_path).to("cuda")

        # Explained Variance with the shape (num_heads, head_dim)
        self.pca_explained_variance = torch.load(explained_variance_file_path).to("cuda")

        # Reshaping the components and taking a transpose to have components along the column dimension and means to be easily broadcastable over the keys
        self.pca_components = self.pca_components.reshape(1, self.num_heads, self.head_dim, self.head_dim).transpose(2, 3)
        self.pca_means = self.pca_means.reshape(1, self.num_heads, 1, self.head_dim)

        # Get the point where the explained variance is 95% per head
        explained_variance_cumsum = self.pca_explained_variance.cumsum(-1)


        if top_r <= 1:
            # Find the maximum index where the explained variance is 95% across all heads - Uncomment this line adaptively set the top_r:w
            top_correct_r = (explained_variance_cumsum < top_r).sum(-1).max().item()

        #    # Instead of sum, we use the median index 
        #    #top_r = (explained_variance_cumsum < 0.95).sum(-1).median().item()
        else:
            top_correct_r = int(top_r)

        # Only keep the top_r components of the pca_components
        self.pca_components_r_key = self.pca_components[:, :, :, :top_correct_r]

        print ("{}: PCA Components Shape: {}".format(layer_idx, self.pca_components_r_key.shape))
        print ("{}: PCA Means Shape: {}".format(layer_idx, self.pca_means.shape))
        print ("Compression Ratio: {}".format(top_correct_r / self.head_dim))

    return modified_attention_init



def get_pca_forward(top_r):
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

        #self.pca_means = self.pca_means.to(key_states.dtype)
        #self.pca_components_r_key = self.pca_components_r_key.to(key_states.dtype)

        # Apply PCA
        #key_states_r = torch.matmul(key_states - self.pca_means, self.pca_components_r_key)
        # Reconstruct keys 
        #key_states = torch.matmul(key_states_r, self.pca_components_r_key.transpose(2, 3)) + self.pca_means

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
        #query_states, query_states_emb, key_states, key_states_emb = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        #self.pca_means = self.pca_means.to(key_states.dtype)
        self.pca_components_r_key = self.pca_components_r_key.to(key_states.dtype)

        # Apply PCA
        key_states_r = torch.matmul(key_states, self.pca_components_r_key)
        query_states_r = torch.matmul(query_states, self.pca_components_r_key)

        # Reconstruct keys 
        #key_states = torch.matmul(key_states_r, self.pca_components_r_key.transpose(2, 3)) + self.pca_means

        attn_weights = (torch.matmul(query_states_r, key_states_r.transpose(2, 3))) / math.sqrt(self.head_dim)

        #attn_weights_new = (torch.matmul(query_states_emb, key_states_emb.transpose(2, 3))) / math.sqrt(self.head_dim)

        #torch.testing.assert_allclose(attn_weights, attn_weights_new)

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

def make_llama_attention_pca(top_r):
    print ("Modifying Llama Attention -> PCA Attention")
    print ("Top R:", top_r)
    print ("Fixed TopR")
    LlamaAttention.__init__ = get_pca_init(top_r)
    LlamaAttention.forward = get_pca_forward(top_r)
