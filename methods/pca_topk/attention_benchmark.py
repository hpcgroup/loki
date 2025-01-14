
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache
import math
import time
import torch
import methods.pca_topk.kernel.pca_topk as G
from methods.common.timers import Timers
import json

# This was earlier supposed to be a specific Cache class for PCA TopK mechanism
# But now it just reimplements the Cache class from transformers.cache_utils
# TODO: Remove this class
class PcaTopKCache(Cache): # Not used anymore
    """
    Cache based on PcaTopK mechanism
    Note: This class is now just a wrapper around the Cache class from transformers.cache_utils
    """
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = [] # Stores the reduced keys for each layer
        self.value_cache: List[torch.Tensor] = []

    @torch.no_grad()
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        query_states: torch.Tensor,
        layer_idx: int,
        topk: bool = True,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            
            # This is also the prompt iteration so we need all the keys for attention
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)          

            return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reset(self):
        self.key_cache: List[torch.Tensor] = [] 
        self.value_cache: List[torch.Tensor] = []

def micro_benchmark_pca_topk(cache, prompt_keys, top_r, top_k, num_layers, timers,
                             num_gen_steps=2000, use_optimised_gather=False):
    import time
    torch.set_float32_matmul_precision("highest")

    head_dim = prompt_keys[0].shape[-1]
    bs = prompt_keys[0].shape[0]
    num_heads = prompt_keys[0].shape[1]
    dtype = prompt_keys[0].dtype
    prompt_seq_length = prompt_keys[0].shape[2]

    matmul_time = 0
    top_keys = torch.zeros(bs, num_heads, top_k, head_dim).to("cuda")
    top_vals = torch.zeros(bs, num_heads, top_k, head_dim).to("cuda")
    pca_projection_mat = torch.randn(num_heads, head_dim, head_dim, dtype=dtype, device='cuda')


    assert use_optimised_gather
    if use_optimised_gather:
        timers.start('total')
        for i in range(num_gen_steps):
            for layer in range(num_layers):
                timers.start('qk-gen')
                generative_query = torch.rand(bs, num_heads, 1, head_dim, device='cuda', dtype=dtype)
                generative_key = torch.rand(bs, num_heads, 1, head_dim, device='cuda', dtype=dtype)
                timers.stop('qk-gen')

                timers.start('project')
                generative_key = generative_key.squeeze().transpose(0, 1).bmm(pca_projection_mat).unsqueeze(2)
                generative_query = generative_query.squeeze().transpose(0, 1).bmm(pca_projection_mat).unsqueeze(2)
                timers.stop('project')

                timers.start('cache-update')
                keys, vals = cache.update(generative_key, generative_key, generative_query, layer, False)
                timers.stop('cache-update')

                timers.start('qk-matmul-1')
                nh, bs, s, r = keys.shape
                attn_weights = G.topr_bmv_optimized(A=generative_query.view(nh*bs, 1, r), B=keys.view(nh*bs, s, r).transpose(-1,-2), 
                                                    r=top_r)
                attn_weights = attn_weights.view(nh, bs, 1, s)
                timers.stop('qk-matmul-1')

                # Get top-k keys and top-k values based on the attention scores
                timers.start('top-k')
                key_states_topk_indices = torch.argsort(attn_weights, dim=-1, descending=True)[:,:,:,:top_k]
                timers.stop('top-k')


                timers.start('reshape-0')
                key_states_topk_indices= key_states_topk_indices.reshape(-1, key_states_topk_indices.shape[-1])
                timers.stop('reshape-0')

                timers.start('reshape-1')
                keys = keys.view(-1, keys.shape[-2] , keys.shape[-1])
                vals = vals.view(-1, vals.shape[-2] , vals.shape[-1])
                timers.stop('reshape-1')

                timers.start('qk-matmul-2')
                attn_weights = G.gather_outer_bmv_optimized(
                    generative_query.reshape(-1, 1, head_dim),
                    keys.transpose(-1, -2),
                    key_states_topk_indices,
                    #.squeeze(0).squeeze(-1),
                    #chunk=256
                    #chunk=min(k2, 65536 // Q.shape[-1]),
                ) / math.sqrt(head_dim)
                timers.stop('qk-matmul-2')

                timers.start('softmax')
                attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(dtype)
                timers.stop('softmax')

                timers.start('sv-matmul')
                attn_output = G.gather_inner_matrix_only_bmv_optimized(
                    attn_weights, vals, key_states_topk_indices)
                timers.stop('sv-matmul')

                timers.start('reshape-output')
                attn_output = attn_output.view(num_heads, bs, 1, head_dim).transpose(0,1).transpose(1,2).contiguous()
                timers.stop('reshape-output')
        timers.stop('total')
    else:
      for i in range(num_gen_steps):
            keys, vals = cache.update(generative_key, generative_key, generative_query, 0, False)
            torch.cuda.synchronize()

            start = time.time()
            attn_weights = torch.matmul(generative_query[:,:,:,:top_r], keys.transpose(2, 3)[:,:,:top_r,:]) / math.sqrt(head_dim)
            # Get top-k keys and top-k values based on the attention scores
            key_states_topk_indices = torch.topk(attn_weights, top_k, dim=-1).indices.to("cuda")
            key_states_topk_indices,_ = torch.sort(key_states_topk_indices, dim=-1)
            key_states_topk_indices = key_states_topk_indices.transpose(-1, -2).expand(-1, -1, -1, head_dim)

            torch.gather(keys, -2, key_states_topk_indices, out=top_keys)
            torch.gather(vals, -2, key_states_topk_indices, out=top_vals)

            attn_weights = torch.matmul(generative_query, top_keys.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, top_vals)
            torch.cuda.synchronize()
            end = time.time()


def micro_bench_actual_attention(cache, prompt_keys, num_layers, timers, num_gen_steps=2000):
    import time
    torch.set_float32_matmul_precision("highest")

    head_dim = prompt_keys[0].shape[-1]
    bs = prompt_keys[0].shape[0]
    num_heads = prompt_keys[0].shape[1]
    dtype = prompt_keys[0].dtype

    matmul_time = 0

    timers.start('total')
    for i in range(num_gen_steps):
      for layer in range(num_layers):
          timers.start('qk-gen')
          generative_query = torch.rand(bs, num_heads, 1, head_dim, dtype=dtype, device='cuda')
          generative_key = torch.rand(bs, num_heads, 1, head_dim, dtype=dtype, device='cuda')
          timers.stop('qk-gen')
          
          timers.start('cache-update')
          keys, vals = cache.update(generative_key, generative_key, generative_query, layer, False)
          timers.stop('cache-update')

          timers.start('qk-matmul-1')
          attn_weights = torch.matmul(generative_query, keys.transpose(2, 3)) / math.sqrt(head_dim)
          timers.stop('qk-matmul-1')

          timers.start('softmax')
          attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(dtype)
          timers.stop('softmax')

          timers.start('sv-matmul')
          attn_output = torch.matmul(attn_weights, vals)
          timers.stop('sv-matmul')
            
          timers.start('reshape-output')
          attn_output = attn_output.transpose(1, 2).contiguous()
          timers.stop('reshape-output')
    

    timers.stop('total')

@torch.no_grad()
def benchmark_attention(batch_size=1,
                        num_heads=32,
                        num_gen_steps=128,
                        prompt_length=3072,
                        topk=256,
                        topr=32,
                        num_layers=32,
                        dtype=torch.float16,
                        vanilla=True,
                        pcatopk=True,
                        ):

    head_dim=128
    # Change this to change batch size, etc.
    prompt_keys = [torch.rand(batch_size, num_heads, prompt_length, head_dim, device='cuda', dtype=dtype) for _ in range(num_layers)]


    #print("PCA TOPK Unoptimized")
    #cache1 = [PcaTopKCache() for _ in range(num_layers)]
    #for i in range(num_layers):
    #    cache1[i].update(prompt_keys[i], prompt_keys[i], prompt_keys[i], 0)
    #micro_benchmark_pca_topk(cache1, prompt_keys, 32, topk, num_gen_steps=num_gen_steps)
    #del cache1



    times_pca_topk = None
    if pcatopk:
        print("PCA TOPK Optimized")
        for _ in range(10):
            cache2 = PcaTopKCache()
            for i in range(num_layers):
                cache2.update(prompt_keys[i].transpose(0,1).contiguous(), 
                              prompt_keys[i].transpose(0,1).contiguous(), 
                              prompt_keys[i].transpose(0,1).contiguous(), i)
            timers = Timers()
            micro_benchmark_pca_topk(cache2, prompt_keys, topr, topk, 
                                     num_gen_steps=num_gen_steps, num_layers=num_layers, 
                                     use_optimised_gather=True, timers=timers)
            del cache2
            times = timers.get_times()
            print(times)
    
        print("Average time (minus cache updates) is - ")
        print(times['total'] - times['cache-update'], " s")
        print("==================================")
        times_pca_topk = times    


    times_vanilla = None
    if vanilla:
        print("Actual Attention")
        for _ in range(10):
            cache3= PcaTopKCache()
            for i in range(num_layers):
                cache3.update(prompt_keys[i], prompt_keys[i], prompt_keys[i], i)
            timers = Timers()
            micro_bench_actual_attention(cache3, prompt_keys, num_layers=num_layers, 
                                         num_gen_steps=num_gen_steps, timers=timers)
            del cache3
            times = timers.get_times()
        print("Average time (minus cache updates) is - ")
        print(times['total'] - times['cache-update'], " s")
        print(times)
        print("==================================")
        times_vanilla = times
    return times_pca_topk, times_vanilla

