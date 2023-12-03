import faiss
import faiss.contrib.torch_utils
import torch

# Supported Types of indexes:
# GpuIndexFlatIP
INDEX_TYPE = 'GpuIndexFlatIP'


def get_index(index_type, vector_size):
    index = None
    if index_type == 'GpuIndexFlatIP':
        index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), vector_size)
    return index


def faiss_attention(query_states, key_states, k):
    """
    ToDo(Siddhant): This method assumes shape of these tensors as num_heads, seq_len, hidden_size which is not the case
    for all LMs. Fix this.
    """
    num_heads, seq_len, hidden_size = query_states.shape
    topk = min(k, seq_len)
    indexes = [get_index(INDEX_TYPE, hidden_size) for _ in range(num_heads)]
    attention = torch.full((num_heads, seq_len, hidden_size), float('-inf')).cuda(query_states.device)

    for head_index, index in enumerate(indexes):
        index.add(key_states[head_index])
        attn_scores, attn_indexes = index.search(query_states[head_index], topk)
        curr_attention = torch.full((seq_len, hidden_size), float('-inf')).cuda(query_states.device)
        curr_attention.scatter_(-1, attn_indexes, attn_scores)
        attention[head_index] = curr_attention
    # todo(Siddhant): how to delete the entire index?
    for _index in indexes:
        _index.reset()
    return attention


def mask_top_k_elements_3d(tensor, k, dim=2):
    # For when k is more than the context length
    topk = min(k, tensor.shape[1])
    # Find the indices of the top k elements along the specified dimension
    _, indices = tensor.topk(topk, dim=dim, largest=True)

    # Create a mask with zeros and ones
    mask = torch.full_like(tensor, fill_value=float('-inf'))
    # mask.scatter_(dim, indices, 1)
    mask.scatter_(dim, indices, tensor.gather(dim, indices))

    # Apply the mask to the original tensor
    # masked_tensor = tensor * mask

    return mask
