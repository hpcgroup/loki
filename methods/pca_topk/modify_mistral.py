
from typing import List, Optional, Tuple, Union
import math
import warnings
from transformers.models.mistral.modeling_mistral import MistralAttention, repeat_kv, apply_rotary_pos_emb, MistralRotaryEmbedding
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.cache_utils import Cache
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from .utils import mask_attn_pca_topk
import methods


import os
pca_data_path = "/global/cfs/cdirs/m4641/ApproxAttn"


try:
    from axonn import axonn as ax
    from axonn.intra_layer import drop
    AXONN_AVAILABLE=True
except ImportError:
    AXONN_AVAILABLE=False

def get_pca_components(layer_idx, head_dim, top_r, num_key_value_groups):
    components_file_path = os.path.join(pca_data_path, "Mistral-7B-PCA/wikitext/postrotary/key/pca_components/pca_components_layer_{}.pt".format(layer_idx))
    mean_file_path = os.path.join(pca_data_path, "Mistral-7B-PCA/wikitext/postrotary/key/pca_means/pca_means_layer_{}.pt".format(layer_idx))
    explained_variance_file_path = os.path.join(pca_data_path, "Mistral-7B-PCA/wikitext/postrotary/key/pca_explained_variance/pca_explained_variance_layer_{}.pt".format(layer_idx))

    # PCA Components with the shape (num_heads, head_dim, top_r)
    pca_components = torch.load(components_file_path).to("cuda")

    # PCA Means with the shape (num_heads, head_dim)
    pca_means = torch.load(mean_file_path).to("cuda")

    # Explained Variance with the shape (num_heads, head_dim)
    pca_explained_variance = torch.load(explained_variance_file_path).to("cuda")

    # Reshaping the components and taking a transpose to have components along the column dimension and means to be easily broadcastable over the keys
    pca_components = pca_components.reshape(1, -1, head_dim, head_dim).transpose(2, 3)
    pca_means = pca_means.reshape(1, -1, 1, head_dim)

    # Get the point where the explained variance is 95% per head
    explained_variance_cumsum = pca_explained_variance.cumsum(-1)


    if top_r < 1:
        # Find the maximum index where the explained variance is 95% across all heads - Uncomment this line adaptively set the top_r:w
        top_correct_r = (explained_variance_cumsum < top_r).sum(-1).max().item()

    #    # Instead of sum, we use the median index 
    #    #top_r = (explained_variance_cumsum < 0.95).sum(-1).median().item()
    else:
        top_correct_r = int(top_r)

    # Only keep the top_r components of the pca_components
    pca_components_r_key = pca_components[:, :, :, :top_correct_r]
    pca_components_r_key = repeat_kv(pca_components_r_key, num_key_value_groups)
    pca_components = repeat_kv(pca_components, num_key_value_groups)


    print ("{}: PCA Components Shape: {}".format(layer_idx, pca_components_r_key.shape))
    print ("{}: PCA Means Shape: {}".format(layer_idx, pca_means.shape))
    print ("Compression Ratio: {}".format(top_correct_r / head_dim))

    if AXONN_AVAILABLE and ax.is_initialized:
        ## only keep pca data for the heads on the GPU
        pca_components = drop(pca_components, transpose=True, skip_batch=True, dim=1)
        pca_means = drop(pca_means, transpose=True, skip_batch=True, dim=1)
        pca_components_r_key = drop(pca_components_r_key, transpose=True, skip_batch=True, dim=1)

    return pca_means, pca_components, pca_components_r_key
def get_pca_forward(top_r, top_k):
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

        if not hasattr(self, "pca_components"):
            self.pca_means, self.pca_components, self.pca_components_r_key = get_pca_components(self.layer_idx, self.head_dim, top_r, self.num_key_value_groups)
        bsz, q_len, _ = hidden_states.size()

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
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        self.pca_means = self.pca_means.to(key_states.dtype)
        self.pca_components_r_key = self.pca_components_r_key.to(key_states.dtype)
        self.pca_components = self.pca_components.to(key_states.dtype)


        key_states_pca  = torch.matmul(key_states, self.pca_components)
        query_states_pca = torch.matmul(query_states, self.pca_components)
        attn_weights = (torch.matmul(query_states_pca, key_states_pca.transpose(2, 3))) / math.sqrt(self.head_dim)

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

        if top_k <= 1:
            topk = int(top_k * attn_weights.shape[-1])
        else:
            topk = int(top_k)
        attn_weights, alpha = mask_attn_pca_topk(self.layer_idx, attn_weights, attention_mask, query_states, key_states, self.pca_components, self.pca_components_r_key, top_r, topk)

        assert alpha is not None, "alpha is None"

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        ## Compute cumulative sum along the desired dimension
        #cumulative_sum = torch.cumsum(value_states, dim=2).cuda()

        ## Compute the cumulative mean along the desired dimension
        #cumulative_mean = cumulative_sum / torch.arange(1, value_states.size(2) + 1).float().unsqueeze(0).unsqueeze(1).unsqueeze(3).cuda()

        #attn_output = ((1 - alpha) * cumulative_mean) + alpha * attn_output
        #attn_output = attn_output.to(query_states.dtype)

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

def make_mistral_attention_pca_topk(top_r, top_k):
    print ("Modifying Mistral Attention -> PCA Attention")
    print ("Top R:", top_r)
    print ("Top K:", top_k)
    MistralAttention.forward = get_pca_forward(top_r, top_k)
