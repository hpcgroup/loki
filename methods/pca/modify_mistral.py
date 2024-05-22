
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

import methods

def get_pca_init(top_r):
    def modified_attention_init(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super(MistralAttention, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        # Initialise PCA transforms
        components_file_path = "/pscratch/sd/p/Dir/InferenceData/Mistral-7B-PCA/wikitext/postrotary/key/pca_components/pca_components_layer_{}.pt".format(layer_idx)
        mean_file_path = "/pscratch/sd/p/Dir/InferenceData/Mistral-7B-PCA/wikitext/postrotary/key/pca_means/pca_means_layer_{}.pt".format(layer_idx)
        explained_variance_file_path = "/pscratch/sd/p/Dir/InferenceData/Mistral-7B-PCA/wikitext/postrotary/key/pca_explained_variance/pca_explained_variance_layer_{}.pt".format(layer_idx)

        # PCA Components with the shape (num_heads, head_dim, top_r)
        self.pca_components = torch.load(components_file_path).to("cuda")

        # PCA Means with the shape (num_heads, head_dim)
        self.pca_means = torch.load(mean_file_path).to("cuda")

        # Explained Variance with the shape (num_heads, head_dim)
        self.pca_explained_variance = torch.load(explained_variance_file_path).to("cuda")

        # Reshaping the components and taking a transpose to have components along the column dimension and means to be easily broadcastable over the keys
        self.pca_components = self.pca_components.reshape(1, self.num_key_value_heads, self.head_dim, self.head_dim).transpose(2, 3)
        self.pca_means = self.pca_means.reshape(1, self.num_key_value_heads, 1, self.head_dim)

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

        self.pca_means = self.pca_means.to(key_states.dtype)
        self.pca_components_r_key = self.pca_components_r_key.to(key_states.dtype)

        # Apply PCA
        key_states_r = torch.matmul(key_states - self.pca_means, self.pca_components_r_key)
        # Reconstruct keys 
        key_states = torch.matmul(key_states_r, self.pca_components_r_key.transpose(2, 3)) + self.pca_means

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

def make_mistral_attention_pca(top_r):
    print ("Modifying Llama Attention -> PCA Attention")
    print ("Top R:", top_r)
    print ("Fixed TopR")
    MistralAttention.__init__ = get_pca_init(top_r)
    MistralAttention.forward = get_pca_forward(top_r)