import faiss
import faiss.contrib.torch_utils
import torch

# Supported Types of indexes:
# GpuIndexFlatIP|GpuIndexIVFFlat|GpuIndexIVFPQ
INDEX_TYPE = 'GpuIndexFlatIP'
STANDARD_GPU_RESOURCE = faiss.StandardGpuResources()
NLIST = 180
NSUBQUANTIZER = 16
NBITS_PER_QUANTIZER = 8  #GPU version only support this value :shrug:


def get_index(index_type, vector_size):
    index = None
    if index_type == 'GpuIndexFlatIP':
        index = faiss.GpuIndexFlatIP(STANDARD_GPU_RESOURCE, vector_size)
    if index_type == 'GpuIndexIVFFlat':
        index = faiss.GpuIndexIVFFlat(STANDARD_GPU_RESOURCE, vector_size, NLIST, faiss.METRIC_INNER_PRODUCT)
    if index_type == 'GpuIndexIVFPQ':
        index = faiss.GpuIndexIVFPQ(STANDARD_GPU_RESOURCE, vector_size, NLIST, NSUBQUANTIZER, NBITS_PER_QUANTIZER, faiss.METRIC_INNER_PRODUCT)
    return index


def faiss_attention(query_states, key_states, k):
    """
    ToDo(Siddhant): This method assumes shape of these tensors as num_heads, seq_len, hidden_size which is not the case
    for all LMs. Fix this.
    """
    num_heads, seq_len, hidden_size = query_states.shape
    topk = min(k, seq_len)
    indexes = [get_index(INDEX_TYPE, hidden_size) for _ in range(num_heads)]
    attention = torch.full((num_heads, seq_len, seq_len), float('-inf')).cuda(query_states.device)
    #filter_ids = [[0 for _y in range(seq_len)] for _x in range(seq_len)]
    #id_selector = faiss.IDSelectorArray(filter_ids)

    for head_index, index in enumerate(indexes):
        list_tensors = []
        for idx in range(seq_len):
            index.add(key_states[head_index][idx].view(1, hidden_size).float())
            attn_scores, attn_indexes = index.search(query_states[head_index][idx].view(1, hidden_size).float(), topk)
            #print(attn_indexes[:, :idx+1])
            #print(attn_scores[:, :idx+1])
            temp_tensor = torch.full((1, seq_len), float('-inf')).cuda(query_states.device)
            temp_tensor.scatter_(-1, attn_indexes[:, :idx+1], attn_scores[:, :idx+1])
            #print(temp_tensor)
            list_tensors.append(temp_tensor)
        curr_attention = torch.cat(list_tensors)
        #print(curr_attention)
        
        
        #index.add(key_states[head_index].float())
        #attn_scores, attn_indexes = index.search(query_states[head_index].float(), topk)
        #print(attn_indexes)
        #print(attn_scores)
        #curr_attention = torch.full((seq_len, seq_len), float('-inf')).cuda(query_states.device)
        #curr_attention.scatter_(-1, attn_indexes, attn_scores)
        
        
        attention[head_index] = curr_attention
    # todo(Siddhant): how to delete the entire index?
    for _index in indexes:
        _index.reset()
    return attention


def faiss_attention_v2(query_states, key_states, k, use_faiss_scores=False):
    num_heads, seq_len, hidden_size = query_states.shape
    topk = min(k, seq_len)
    indexes = [get_index(INDEX_TYPE, hidden_size) for _ in range(num_heads)]
    attention = torch.bmm(query_states, key_states.transpose(1, 2))

    for head_index, index in enumerate(indexes):
        index.add(key_states[head_index].float())
        scoped_query = query_states[head_index][seq_len - 1].view(1, hidden_size)
        attn_scores, attn_indexes = index.search(scoped_query.float(), topk)
        print(torch.min(attn_indexes), attn_indexes.shape)
        if use_faiss_scores:
            curr_attention = torch.full((1, seq_len), float('-inf')).cuda()
            curr_attention.scatter_(-1, attn_indexes, attn_scores)
        else:
            # todo(Siddhant): implement False case
            pass

        attention[head_index][seq_len - 1] = curr_attention
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
    for i in range(1):
        query_states = torch.randn(num_heads, seq_len, hidden_size, dtype=torch.float16).cuda()
        key_states = torch.randn(num_heads, seq_len, hidden_size, dtype=torch.float16).cuda()
        print(f">>>>>>> faiss attention using {INDEX_TYPE}")
        print(faiss_attention_v2(query_states, key_states, topk, True))
        print(f">>>>>>> using old method (torch.bmm followed by masking topk)")
        print(mask_top_k_elements_3d(torch.bmm(query_states, key_states.transpose(1, 2)), 4))
