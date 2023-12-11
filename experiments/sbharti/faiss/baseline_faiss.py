import time

import faiss
import faiss.contrib.torch_utils
import torch


torch.set_printoptions(precision=10)

# Constants
DIMENTION = 768
SEQ_LEN = 2048
TOP_K = 30
STANDARD_GPU_RESOURCE = faiss.StandardGpuResources()

# Index
INDEX_TYPE = 'GpuIndexFlatIP'


class TimeIt:
    def __init__(self, *args):
        # save args as attributes
        self.message = args[0]

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        print(f"Executed {self.message} in {elapsed_time:.4f} seconds")


def get_index(index_type, vector_size):
    index = None
    if index_type == 'GpuIndexFlatIP':
        index = faiss.GpuIndexFlatIP(STANDARD_GPU_RESOURCE, vector_size)
    return index


def run(index_type):
    keys = torch.rand(SEQ_LEN, DIMENTION).cuda()
    queries = torch.rand(SEQ_LEN, DIMENTION).cuda()
    print("Please note that we are only doing following operation on two matrices of shape (1 head):")
    print("keys: ", keys.shape)
    print("queries: ", keys.shape)

    with TimeIt("creating index"):
        index = get_index(index_type, DIMENTION)

    with TimeIt("adding to index"):
        index.add(keys)

    with TimeIt("getting attn_scores and attn_indexes for all queries"):
        attn_scores, attn_indexes = index.search(queries, TOP_K)

    with TimeIt("getting top_k Q.K_t using faiss"):
        curr_attention = torch.full((SEQ_LEN, SEQ_LEN), float('-inf')).cuda()
        curr_attention.scatter_(-1, attn_indexes, attn_scores)

    with TimeIt("sanity check on scores and indexes (considering only first query)"):
        scoped_query = queries[0].view(1, DIMENTION)
        scoped_q_dot_k = torch.mm(scoped_query, keys.transpose(0, 1))
        scores, indices = scoped_q_dot_k.topk(TOP_K, dim=-1, largest=True)
        set_index_exact = set()
        print_tensor = torch.cat((indices.transpose(0, 1), scores.transpose(0, 1)), -1)
        print("Exact result is:")
        print(print_tensor)
        for idx in indices[0]:
            set_index_exact.add(idx.cpu().numpy())

        attn_scores_1, attn_indexes_1 = index.search(scoped_query, TOP_K)
        set_index_faiss = set()
        print_tensor_1 = torch.cat((attn_indexes_1.transpose(0, 1), attn_scores_1.transpose(0, 1)), -1)
        print("Faiss result is:")
        print(print_tensor_1)
        for idx in attn_indexes_1[0]:
            print(idx.cpu())
            #set_index_faiss.add(idx.cpu().numpy())
        print("set_index_faiss - set_index_exact: ", set_index_faiss - set_index_exact)
        print("set_index_exact - set_index_faiss: ", set_index_exact - set_index_faiss)



    with TimeIt("torch.mm on keys and queries"):
        q_dot_k = torch.mm(queries, keys.transpose(0, 1))
        _, torch_mm_indices = q_dot_k.topk(TOP_K, dim=-1, largest=True)
        mask = torch.full_like(q_dot_k, fill_value=float('-inf'))
        mask.scatter_(-1, torch_mm_indices, q_dot_k.gather(-1, torch_mm_indices))


if __name__ == '__main__':
    run(INDEX_TYPE)
