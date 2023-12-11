import time

import torch

DIM = 500
NUM_SAMPLES = 1

times = []
for e in range(NUM_SAMPLES):
    mat1 = torch.rand(DIM, DIM).cuda()
    mat2 = torch.rand(DIM, DIM).cuda()
    torch.cuda.synchronize()
    start_epoch = time.perf_counter()
    torch.mm(mat1, mat2)
    torch.cuda.synchronize()
    end_epoch = time.perf_counter()
    elapsed = end_epoch - start_epoch
    times.append(elapsed)
    print(f"Executed matmul on 2 {DIM}x{DIM} matrix in {elapsed:.4f} seconds")

avg_time = sum(times)/NUM_SAMPLES
print("Average time taken: ", avg_time)
