from typing import List, Optional, Tuple, Union
import math
import warnings
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, apply_rotary_pos_emb
from transformers.cache_utils import Cache
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from .utils import mask_attn_pca_topk
import methods

def get_pca_topk_init(top_r, top_k):
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

        # Initialise PCA transforms
        components_file_path = "/pscratch/sd/p/prajwal/InferenceData/Pythia-6.9B-PCA/wikitext/postrotary/key/pca_components/pca_components_layer_{}.pt".format(self.layer_idx)
        mean_file_path = "/pscratch/sd/p/prajwal/InferenceData/Pythia-6.9B-PCA/wikitext/postrotary/key/pca_means/pca_means_layer_{}.pt".format(self.layer_idx)
        explained_variance_file_path = "/pscratch/sd/p/prajwal/InferenceData/Pythia-6.9B-PCA/wikitext/postrotary/key/pca_explained_variance/pca_explained_variance_layer_{}.pt".format(self.layer_idx)

        print ("Components File Path:", components_file_path)

        # PCA Components with the shape (num_heads, head_dim, top_r)
        self.pca_components = torch.load(components_file_path).to("cuda")

        # PCA Means with the shape (num_heads, head_dim)
        self.pca_means = torch.load(mean_file_path).to("cuda")

        # Explained Variance with the shape (num_heads, head_dim)
        self.pca_explained_variance = torch.load(explained_variance_file_path).to("cuda")

        # Reshaping the components and taking a transpose to have components along the column dimension and means to be easily broadcastable over the keys
        self.pca_components = self.pca_components.reshape(1, self.num_attention_heads, self.head_size, self.head_size).transpose(2, 3)
        self.pca_means = self.pca_means.reshape(1, self.num_attention_heads, 1, self.head_size)

        # Get the point where the explained variance is 95% per head
        explained_variance_cumsum = self.pca_explained_variance.cumsum(-1)


        if top_r < 1:
            # Find the maximum index where the explained variance is 95% across all heads - Uncomment this line adaptively set the top_r:w
            top_correct_r = (explained_variance_cumsum < top_r).sum(-1).max().item()

        #    # Instead of sum, we use the median index 
        #    #top_r = (explained_variance_cumsum < 0.95).sum(-1).median().item()
        else:
            top_correct_r = int(top_r)

        # Only keep the top_r components of the pca_components
        self.pca_components_r_key = self.pca_components[:, :, :, :top_correct_r]

        print ("{}: PCA Components Shape: {}".format(self.layer_idx, self.pca_components_r_key.shape))
        print ("{}: PCA Means Shape: {}".format(self.layer_idx, self.pca_means.shape))
        print ("Compression Ratio: {}".format(top_correct_r / self.head_size))
    return modified_attention_init


def get_pca_topk_attn(top_r, top_k):
    def modified_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # dynamically increase the causal mask with the key length, if needed.
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        self.pca_means = self.pca_means.to(key.dtype)
        self.pca_components_r_key = self.pca_components_r_key.to(key.dtype)
        self.pca_components = self.pca_components.to(key.dtype)

        query_pca = torch.matmul(query, self.pca_components)
        key_pca = torch.matmul(key, self.pca_components)

        query_pca = query_pca.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key_pca = key_pca.view(batch_size * num_attention_heads, key_length, attn_head_size)

        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query_pca,
            key_pca.transpose(1, 2),
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
        
        # Get top-k attention weights
        if top_k <= 1:
            topk = int(top_k * attn_scores.shape[-1])
        else:
            topk = int(top_k)
        attn_scores, alpha = mask_attn_pca_topk(self.layer_idx, attn_scores, attention_mask, query, key, self.pca_components, self.pca_components_r_key, top_r, topk)

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights
    return modified_attn

def make_gptneox_attention_pca_topk(top_r, top_k):
    print ("Modifying GPT NeoX Attention -> PCA TopK Attention")
    print ("Top R:", top_r)
    print ("Top K:", top_k)
    print ("Not using alpha")
    GPTNeoXAttention.__init__ = get_pca_topk_init(top_r, top_k)
    GPTNeoXAttention._attn = get_pca_topk_attn(top_r, top_k)
