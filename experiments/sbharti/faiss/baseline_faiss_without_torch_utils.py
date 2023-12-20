import time

import faiss
import torch


torch.set_printoptions(precision=10)

# Constants
DIMENTION = 768
SEQ_LEN = 2048
TOP_K = 30
STANDARD_GPU_RESOURCE = faiss.StandardGpuResources()

# Index (one of IndexFlatIP|IndexIVFFlat|IndexPQ|IndexHNSWFlat)
INDEX_TYPE = 'IndexHNSWFlat'
NLIST = 180
NSUBQUANTIZER = 16
NBITS_PER_QUANTIZER = 8
M_HNSW = 10


class TimeIt:
    def __init__(self, *args):
        # save args as attributes
        self.message = args[0]

    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        print(f"Executed {self.message} in {elapsed_time:.4f} seconds")


def get_index(index_type, vector_size):
    index = None
    if index_type == 'IndexFlatIP':
        index = faiss.IndexFlatIP(vector_size)
    if index_type == 'IndexIVFFlat':
        quantizer_index = faiss.IndexFlatIP(vector_size)
        index = faiss.IndexIVFFlat(quantizer_index, DIMENTION, NLIST, faiss.METRIC_INNER_PRODUCT)
    if index_type == 'IndexPQ':
        index = faiss.IndexPQ(DIMENTION, NSUBQUANTIZER, NBITS_PER_QUANTIZER, faiss.METRIC_INNER_PRODUCT)
    if index_type == 'IndexHNSWFlat':
        index = faiss.IndexHNSWFlat(DIMENTION, M_HNSW, faiss.METRIC_INNER_PRODUCT)
    t1 = time.perf_counter()
    gpu_index = faiss.index_cpu_to_gpu(STANDARD_GPU_RESOURCE, 0, index)
    t2 = time.perf_counter()
    print(f"Converting index to gpu took {t2-t1:.4f} seconds")
    return gpu_index


def run(index_type):
    keys = torch.rand(SEQ_LEN, DIMENTION).cuda()
    queries = torch.rand(SEQ_LEN, DIMENTION).cuda()
    print("Please note that we are only doing following operation on two matrices of shape (1 head):")
    print("keys: ", keys.shape)
    print("queries: ", keys.shape)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with TimeIt("creating index"):
        index = get_index(index_type, DIMENTION)

    if not index.is_trained:
        with TimeIt("training the index with keys"):
            index.train(keys.cpu())

    with TimeIt("adding to index"):
        index.add(keys.cpu())

    with TimeIt("getting attn_scores and attn_indexes for all queries"):
        attn_scores, attn_indexes = index.search(queries.cpu(), TOP_K)
        attn_scores = torch.from_numpy(attn_scores).cuda()
        attn_indexes = torch.from_numpy(attn_indexes).cuda()

    attn_indexes[attn_indexes == -1] = 0
    with TimeIt("getting top_k Q.K_t using faiss"):
        curr_attention = torch.full((SEQ_LEN, SEQ_LEN), float('-inf')).cuda()
        curr_attention.scatter_(-1, attn_indexes, attn_scores)

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(f"Executed faiss based Q.K_t with topk in {t2-t1:.4f} seconds")
    with TimeIt("sanity check on scores and indexes (considering only first query)"):
        scoped_query = queries[0].view(1, DIMENTION)
        scoped_q_dot_k = torch.mm(scoped_query, keys.transpose(0, 1))
        scores, indices = scoped_q_dot_k.topk(TOP_K, dim=-1, largest=True)
        set_index_exact = set()
        print_tensor = torch.cat((indices.transpose(0, 1), scores.transpose(0, 1)), -1)
        for idx in indices[0]:
            _idx = int(idx.cpu().numpy())
            set_index_exact.add(_idx)

        attn_scores_1, attn_indexes_1 = index.search(scoped_query.cpu(), TOP_K)
        attn_scores_1 = torch.from_numpy(attn_scores_1).cuda()
        attn_indexes_1 = torch.from_numpy(attn_indexes_1).cuda()
        set_index_faiss = set()
        print_tensor_1 = torch.cat((attn_indexes_1.transpose(0, 1), attn_scores_1.transpose(0, 1)), -1)
        print("exact_index, exact_score, faiss_index, faiss_score")
        print_tensor_2 = torch.cat((print_tensor, print_tensor_1), -1)
        print(print_tensor_2)
        for idx in attn_indexes_1[0]:
            _idx = int(idx.cpu().numpy())
            set_index_faiss.add(_idx)
        print("set_index_faiss - set_index_exact: ", set_index_faiss - set_index_exact)
        print("set_index_exact - set_index_faiss: ", set_index_exact - set_index_faiss)

    with TimeIt("torch.mm based Q.K_t with topk"):
        q_dot_k = torch.mm(queries, keys.transpose(0, 1))
        _, torch_mm_indices = q_dot_k.topk(TOP_K, dim=-1, largest=True)
        mask = torch.full_like(q_dot_k, fill_value=float('-inf'))
        mask.scatter_(-1, torch_mm_indices, q_dot_k.gather(-1, torch_mm_indices))

    mask[mask == float('-inf')] = float('-1e10')
    curr_attention[curr_attention == float('-inf')] = float('-1e10')

    is_close = torch.all(torch.abs(curr_attention - mask) < 0.01)
    if is_close:
        print("The Q.K_t from two methods are close.")
    else:
        print("The Q.K_t from two methods are not close.")


if __name__ == '__main__':
    run(INDEX_TYPE)
