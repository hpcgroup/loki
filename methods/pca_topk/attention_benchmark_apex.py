from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache
import math
import time
import torch
import methods.pca_topk.kernel.pca_topk as G
from methods.common.timers import Timers
from methods.pca_topk.sparsity_utils import SparsityHandler
import json


class PcaTopKCache(Cache):
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
        is_prompt: bool = False,  # Added is_prompt parameter with default value
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
            is_prompt (`bool`, `optional`):
                Whether this update is for the prompt or for generation.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

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
            # Ensure dimensions match before concatenation
            if key_states.dim() != self.key_cache[layer_idx].dim():
                # Adjust dimensions to match
                if key_states.dim() < self.key_cache[layer_idx].dim():
                    # Add missing dimensions
                    for _ in range(self.key_cache[layer_idx].dim() - key_states.dim()):
                        key_states = key_states.unsqueeze(0)
                    value_states = value_states.unsqueeze(0)
            
            # Check sequence length dimension
            cache_seq_len = self.key_cache[layer_idx].shape[-2]
            new_seq_len = key_states.shape[-2]
            
            # Ensure shapes match in all dimensions except sequence length
            if self.key_cache[layer_idx].shape[:-2] != key_states.shape[:-2]:
                # Adjust shapes to match
                if self.key_cache[layer_idx].shape[0] != key_states.shape[0]:
                    # Reshape to match number of heads
                    key_states = key_states.expand(self.key_cache[layer_idx].shape[0], *key_states.shape[1:])
                    value_states = value_states.expand(self.value_cache[layer_idx].shape[0], *value_states.shape[1:])
            
            # Now concatenate along sequence length dimension
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
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reset(self):
        self.key_cache: List[torch.Tensor] = [] 
        self.value_cache: List[torch.Tensor] = []


class DenseAttentionProj:
    """Implements realistic query/key/value projections for a transformer layer."""
    def __init__(self, hidden_size, num_heads, head_dim, dtype=torch.float16, device='cuda', sparsity_handler=None):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sparsity_handler = sparsity_handler
        
        # Initialize Q/K/V projection matrices as real transformers would have
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, dtype=dtype, device=device)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, dtype=dtype, device=device)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, dtype=dtype, device=device)
        
        # Output projection
        self.out_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, dtype=dtype, device=device)
        
        # Apply sparsity to weight matrices if sparsity handler is provided
        if self.sparsity_handler and self.sparsity_handler.sparsity_type == "attention-query-key":
            self.q_proj = self.sparsity_handler.sparsify_linear_weight(self.q_proj, "q_proj")
            self.k_proj = self.sparsity_handler.sparsify_linear_weight(self.k_proj, "k_proj")
            # We typically don't sparsify the value projection, but you can uncomment if needed
            # self.v_proj = self.sparsity_handler.sparsify_linear_weight(self.v_proj, "v_proj")
        
    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
    @torch.no_grad()
    def forward(self, hidden_states):
        """
        Compute Q, K, V projections from input embeddings as in a real transformer
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            query_states, key_states, value_states of shapes (batch_size, num_heads, seq_len, head_dim)
        """
        bsz, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        
        # Project to Q, K, V using sparse matrix multiplication if available
        if self.sparsity_handler and self.sparsity_handler.sparsity_type == "attention-query-key":
            # The weight matrices are already sparsified during initialization
            # We just need to use them for the projection
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        else:
            # Regular dense projection
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        query_states = self._shape(q, seq_len, bsz)
        key_states = self._shape(k, seq_len, bsz)
        value_states = self._shape(v, seq_len, bsz)
        
        return query_states, key_states, value_states
    
    @torch.no_grad()
    def output_projection(self, attn_output):
        """
        Apply output projection to attention output
        
        Args:
            attn_output: Attention output of shape (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        bsz, seq_len = attn_output.shape[0], attn_output.shape[2]
        
        # Reshape to (batch_size, seq_len, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        
        # Apply output projection
        return self.out_proj(attn_output)


def micro_benchmark_pca_topk(cache, prompt_keys, top_r, top_k, num_layers, timers,
                             num_gen_steps=2000, use_optimised_gather=False, sparsity_type=None):
    import time
    torch.set_float32_matmul_precision("highest")

    head_dim = prompt_keys[0].shape[-1]
    bs = prompt_keys[0].shape[0]
    num_heads = prompt_keys[0].shape[1]
    hidden_size = num_heads * head_dim
    dtype = prompt_keys[0].dtype
    prompt_seq_length = prompt_keys[0].shape[2]

    # Initialize sparsity handler if sparsity type is specified
    sparsity_handler = None
    if sparsity_type:
        sparsity_handler = SparsityHandler(sparsity_type)
        
    # PCA projection matrix
    pca_projection_mat = torch.randn(num_heads, head_dim, head_dim, dtype=dtype, device='cuda')
    
    # Apply sparsity pattern to the projection matrix if sparsity handler is initialized
    if sparsity_handler:
        # We've already verified the SparsityHandler is working correctly during setup
        pca_projection_mat = sparsity_handler.apply_sparsity_pattern(pca_projection_mat)
        # Get sparsity statistics
        stats = sparsity_handler.get_sparsity_stats(pca_projection_mat)
    
    # Initialize dense attention projections for each layer (more realistic)
    attention_projections = [
        DenseAttentionProj(hidden_size, num_heads, head_dim, dtype=dtype, sparsity_handler=sparsity_handler) 
        for _ in range(num_layers)
    ]
    
    # Initial input embedding to start the autoregressive process
    input_embedding = torch.rand(bs, 1, hidden_size, device='cuda', dtype=dtype)

    assert use_optimised_gather
    if use_optimised_gather:
        timers.start('total')
        for i in range(num_gen_steps):
            for layer in range(num_layers):
                # Use the previous output as input (autoregressive) for realistic benchmarking
                layer_input = input_embedding
                
                timers.start('qk-gen')
                # Generate query/key/value from the same input embedding (realistic)
                query_states, key_states, value_states = attention_projections[layer].forward(layer_input)
                timers.stop('qk-gen')

                timers.start('project')
                # Apply PCA projection (if using)
                projected_key = key_states.squeeze(2).transpose(0, 1).bmm(pca_projection_mat).unsqueeze(2)
                projected_query = query_states.squeeze(2).transpose(0, 1).bmm(pca_projection_mat).unsqueeze(2)
                timers.stop('project')
                
                # Note: We no longer apply sparsity here since it's now applied to the weight matrices before projection

                # Ensure projected_key and value_states have the same shape format as the cache
                # The cache expects [num_heads, batch_size, seq_len, head_dim] format
                if projected_key.shape[0] != num_heads:
                    # Transpose to match cache format if needed
                    if projected_key.dim() == 4 and projected_key.shape[1] == bs:
                        # Already in [num_heads, bs, seq_len, head_dim] format
                        pass
                    elif projected_key.dim() == 3:
                        # Add batch dimension if missing
                        projected_key = projected_key.unsqueeze(1)
                    else:
                        # Reshape to match expected format
                        projected_key = projected_key.view(num_heads, bs, -1, head_dim)
                
                # Ensure value_states has the same format as the cache
                if value_states.shape[0] != num_heads:
                    # Transpose to match cache format if needed
                    if value_states.dim() == 4 and value_states.shape[1] == bs:
                        # Already in [num_heads, bs, seq_len, head_dim] format
                        pass
                    elif value_states.dim() == 3:
                        # Add batch dimension if missing
                        value_states = value_states.unsqueeze(1)
                    else:
                        # Reshape to match expected format
                        value_states = value_states.view(num_heads, bs, -1, head_dim)
                
                timers.start('cache-update')
                keys, vals = cache.update(projected_key, value_states, projected_query, layer, False)
                timers.stop('cache-update')

                timers.start('qk-matmul-1')
                nh, bs, s, r = keys.shape
                
                # Use sparse matrix multiplication if needed
                if sparsity_handler and sparsity_type == "attention-query-key":
                    attn_weights = sparsity_handler.sparse_matmul(
                        projected_query.view(nh*bs, 1, r), 
                        keys.view(nh*bs, s, r).transpose(-1,-2)
                    )
                else:
                    attn_weights = G.topr_bmv_optimized(
                        A=projected_query.view(nh*bs, 1, r), 
                        B=keys.view(nh*bs, s, r).transpose(-1,-2), 
                        r=top_r
                    )
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
                # Use sparse matrix multiplication if needed
                if sparsity_handler and sparsity_type == "attention-query-key":
                    # For sparse matmul, we need to gather the keys first
                    gathered_keys = torch.gather(
                        keys, 
                        1, 
                        key_states_topk_indices.unsqueeze(-1).expand(-1, -1, keys.size(-1))
                    )
                    
                    # Then perform sparse matmul
                    attn_weights = sparsity_handler.sparse_matmul(
                        projected_query.reshape(-1, 1, head_dim),
                        gathered_keys.transpose(-1, -2)
                    ) / math.sqrt(head_dim)
                else:
                    attn_weights = G.gather_outer_bmv_optimized(
                        projected_query.reshape(-1, 1, head_dim),
                        keys.transpose(-1, -2),
                        key_states_topk_indices,
                    ) / math.sqrt(head_dim)
                timers.stop('qk-matmul-2')

                timers.start('softmax')
                attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(dtype)
                timers.stop('softmax')

                timers.start('sv-matmul')
                attn_output = G.gather_inner_matrix_only_bmv_optimized(
                    attn_weights, vals, key_states_topk_indices)
                timers.stop('sv-matmul')

                # Reshape is handled along with output projection
                
                # Apply output projection (realistic) - include in reshape-output timing
                timers.start('reshape-output')
                attn_output = attn_output.view(num_heads, bs, 1, head_dim).transpose(0,1)
                output = attention_projections[layer].output_projection(attn_output)
                timers.stop('reshape-output')
                
                # Store this layer's output as input for the next layer or next token
                input_embedding = output
                
        timers.stop('total')
    else:
        pass


def micro_bench_actual_attention(cache, prompt_keys, num_layers, timers, num_gen_steps=2000, sparsity_type=None):
    import time
    torch.set_float32_matmul_precision("highest")

    head_dim = prompt_keys[0].shape[-1]
    bs = prompt_keys[0].shape[0]
    num_heads = prompt_keys[0].shape[1]
    hidden_size = num_heads * head_dim
    dtype = prompt_keys[0].dtype
    
    # Initialize sparsity handler if needed
    sparsity_handler = None
    if sparsity_type:
        from methods.pca_topk.sparsity_utils import SparsityHandler
        sparsity_handler = SparsityHandler(sparsity_type)

    # Initialize dense attention projections for each layer (more realistic)
    attention_projections = [
        DenseAttentionProj(hidden_size, num_heads, head_dim, dtype=dtype, sparsity_handler=sparsity_handler) 
        for _ in range(num_layers)
    ]
    
    # Initial input embedding to start the autoregressive process
    input_embedding = torch.rand(bs, 1, hidden_size, device='cuda', dtype=dtype)

    timers.start('total')
    for i in range(num_gen_steps):
      for layer in range(num_layers):
          # Use the previous output as input (autoregressive)
          layer_input = input_embedding
          
          timers.start('qk-gen')
          # Generate query/key/value from the same input embedding (realistic)
          query_states, key_states, value_states = attention_projections[layer].forward(layer_input)
          timers.stop('qk-gen')
          
          timers.start('cache-update')
          keys, vals = cache.update(key_states, value_states, query_states, layer, False)
          timers.stop('cache-update')

          timers.start('qk-matmul-1')
          attn_weights = torch.matmul(query_states, keys.transpose(2, 3)) / math.sqrt(head_dim)
          timers.stop('qk-matmul-1')

          timers.start('softmax')
          attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(dtype)
          timers.stop('softmax')

          timers.start('sv-matmul')
          attn_output = torch.matmul(attn_weights, vals)
          timers.stop('sv-matmul')
            
          timers.start('reshape-output')
          # Apply output projection (realistic) as part of reshape timing
          output = attention_projections[layer].output_projection(attn_output)
          timers.stop('reshape-output')
          
          # Store this layer's output as input for the next layer or next token
          input_embedding = output

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
                        sparsity_type=None,
                        ):

    # If sparsity is requested, verify that the sparse multiplication is available
    if sparsity_type:
        # Only allow attention-query-key sparsity type
        if sparsity_type != "attention-query-key":
            print(f"Warning: Sparsity type '{sparsity_type}' is not supported. Only 'attention-query-key' is supported.")
            print("Setting sparsity_type to 'attention-query-key'")
            sparsity_type = "attention-query-key"
            
        try:
            # Initialize sparsity handler
            from methods.pca_topk.sparsity_utils import SparsityHandler
            sparsity_handler = SparsityHandler(sparsity_type)
            
            # Create some test tensors
            test_q = torch.rand(2, 1, 128, device='cuda', dtype=dtype)
            test_k = torch.rand(2, 10, 128, device='cuda', dtype=dtype)
            
            # Try the sparse matrix multiplication
            _ = sparsity_handler.sparse_matmul(test_q, test_k.transpose(-1, -2))
            
            # If we get here, sparse matmul is working
            print(f"Verified that sparse matmul for sparsity type '{sparsity_type}' is working.")
        except Exception as e:
            # If sparse matmul fails, abort the benchmark - don't continue with a fallback
            raise RuntimeError(f"Sparse matmul for '{sparsity_type}' failed verification: {e}\n"
                               f"Cannot proceed with benchmark as sparse optimization is required.")

    head_dim=128
    hidden_size = num_heads * head_dim
    
    # Generate initial prompt embeddings (more realistic than pure random keys)
    prompt_embeddings = [torch.rand(batch_size, prompt_length, hidden_size, device='cuda', dtype=dtype) for _ in range(num_layers)]
    
    # Create attention projections
    # Initialize sparsity_handler to None if not defined (for Loki baseline)
    if 'sparsity_handler' not in locals() and sparsity_type is None:
        sparsity_handler = None
        
    attention_projections = [
        DenseAttentionProj(hidden_size, num_heads, head_dim, dtype=dtype, sparsity_handler=sparsity_handler) 
        for _ in range(num_layers)
    ]
    
    # Create prompt keys, values using proper projections
    prompt_keys = []
    prompt_values = []
    
    for i in range(num_layers):
        q, k, v = attention_projections[i].forward(prompt_embeddings[i])
        prompt_keys.append(k)
        prompt_values.append(v)

    times_pca_topk = None
    if pcatopk:
        print("PCA TOPK Optimized")
        for _ in range(100):
            cache2 = PcaTopKCache()
            for i in range(num_layers):
                cache2.update(prompt_keys[i].transpose(0,1).contiguous(), 
                              prompt_values[i].transpose(0,1).contiguous(), 
                              prompt_keys[i].transpose(0,1).contiguous(), i)
            timers = Timers()
            micro_benchmark_pca_topk(cache2, prompt_keys, topr, topk, 
                                     num_gen_steps=num_gen_steps, num_layers=num_layers, 
                                     use_optimised_gather=True, timers=timers,
                                     sparsity_type=sparsity_type)
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
        for _ in range(100):
            cache3= PcaTopKCache()
            for i in range(num_layers):
                cache3.update(prompt_keys[i], prompt_values[i], prompt_keys[i], i)
            timers = Timers()
            micro_bench_actual_attention(cache3, prompt_keys, num_layers=num_layers, 
                                         num_gen_steps=num_gen_steps, timers=timers,
                                         sparsity_type=sparsity_type)
            del cache3
            times = timers.get_times()
        print("Average time (minus cache updates) is - ")
        print(times['total'] - times['cache-update'], " s")
        print(times)
        print("==================================")
        times_vanilla = times
    return times_pca_topk, times_vanilla