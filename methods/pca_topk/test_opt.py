
from typing import Any, Dict, List, Optional, Tuple
from transformers.cache_utils import Cache
import math
import time

import torch
import external.gather_matmul as G



topk_time = 0
iter_num = 0
torch.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(0)




def test_pca_topk_attn(keys, vals, top_r, top_k, num_gen_steps=2000, use_optimised_gather=False):
    import time
    torch.set_float32_matmul_precision("highest")

    head_dim = keys.shape[-1]
    bs = keys.shape[0]
    num_heads = keys.shape[1]
    
    generative_query = torch.rand(bs, num_heads, 1, head_dim).to("cuda")
    print (generative_query)
    generative_key = torch.rand(bs, num_heads, 1, head_dim).to("cuda")

    top_keys = torch.zeros(bs, num_heads, top_k, head_dim).to("cuda")
    top_vals = torch.zeros(bs, num_heads, top_k, head_dim).to("cuda")

    if use_optimised_gather:
        print (generative_query[:,:,:,:top_r])
        print (keys.transpose(2, 3)[:,:,:top_r,:]) 
        print (head_dim)
        attn_weights = torch.matmul(generative_query[:,:,:,:top_r], keys[:,:,:,:top_r].transpose(-1, -2)) / math.sqrt(head_dim)

        print ("Approximate Attention Weights")
        print (attn_weights)

        # Get top-k keys and top-k values based on the attention scores
        key_states_topk_indices = torch.topk(attn_weights, top_k, dim=-1).indices.to("cuda")
        key_states_topk_indices,_ = torch.sort(key_states_topk_indices, dim=-1)
        key_states_topk_indices= key_states_topk_indices.reshape(-1, key_states_topk_indices.shape[-1])

        print ("Topk Indices")
        print (key_states_topk_indices)

        keys = keys.reshape(-1, keys.shape[-2] , keys.shape[-1])
        vals = vals.reshape(-1, vals.shape[-2] , vals.shape[-1])

        attn_weights = G.gather_outer_bmv(
            generative_query.reshape(-1, 1, head_dim),
            keys.transpose(-1, -2),
            key_states_topk_indices,
            #.squeeze(0).squeeze(-1),
            chunk=256
            #chunk=min(k2, 65536 // Q.shape[-1]),
        )
        print ("TopK Attention")
        print (attn_weights)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        print (attn_weights)

        attn_output = G.gather_inner_matrix_only_bmv(
            attn_weights, vals, key_states_topk_indices, chunk=64
        )

        print ("Attn Output")
        print (attn_output)

        torch.cuda.synchronize()
    else:
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



def benchmark_attention(batch_size=1,
                        num_heads=32,
                        num_gen_steps=128,
                        prompt_length=3072,
                        topk=256):

    head_dim=4
    # Change this to change batch size, etc.
    keys = torch.rand(batch_size, num_heads, prompt_length, head_dim).to("cuda")
    print (keys)

    test_pca_topk_attn(keys, keys, 2, topk, 1, True)

if __name__ == "__main__":
    #test_pcatopk_cache()
    with torch.no_grad():
        benchmark_attention(prompt_length=4, num_heads=1, num_gen_steps=1, batch_size=1, topk=2)
    

