import faiss
import faiss.contrib.torch_utils
import torch

# Supported Types of indexes:
# GpuIndexFlatIP
INDEX_TYPE = 'GpuIndexFlatIP'
STANDARD_GPU_RESOURCE = faiss.StandardGpuResources()


def get_index(index_type, vector_size):
    index = None
    if index_type == 'GpuIndexFlatIP':
        index = faiss.GpuIndexFlatIP(STANDARD_GPU_RESOURCE, vector_size)
    return index


def faiss_attention(query_states, key_states, k):
    """
    ToDo(Siddhant): This method assumes shape of these tensors as num_heads, seq_len, hidden_size which is not the case
    for all LMs. Fix this.
    """
    num_heads, seq_len, hidden_size = query_states.shape
    topk = min(k, seq_len)
    print(num_heads, seq_len, hidden_size, topk)
    indexes = [get_index(INDEX_TYPE, hidden_size) for _ in range(num_heads)]
    attention = torch.full((num_heads, seq_len, seq_len), float('-inf')).cuda(query_states.device)

    for head_index, index in enumerate(indexes):
        index.add(key_states[head_index].float())
        attn_scores, attn_indexes = index.search(query_states[head_index].float(), topk)
        curr_attention = torch.full((seq_len, seq_len), float('-inf')).cuda(query_states.device)
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


if __name__ == '__main__':
    num_heads = 3
    seq_len = 4
    hidden_size = 5
    topk = 2
    print(f"Assuming num_heads={num_heads}, seq_len={seq_len}, hidden_size={hidden_size}, topk={topk}")
    torch.set_printoptions(precision=10)
    for i in range(5):
        query_states = torch.randn(num_heads, seq_len, hidden_size, dtype=torch.float16).cuda()
        key_states = torch.randn(num_heads, seq_len, hidden_size, dtype=torch.float16).cuda()
        print(f">>>>>>> faiss attention using {INDEX_TYPE}")
        print(faiss_attention(query_states, key_states, topk))
        print(f">>>>>>> using old method (torch.bmm followed by masking topk)")
        print(mask_top_k_elements_3d(torch.bmm(query_states, key_states.transpose(1, 2)), topk))
