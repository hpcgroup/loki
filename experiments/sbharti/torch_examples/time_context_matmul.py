import time

import torch


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


DIM = 100000
NUM_SAMPLES = 30

times = []
for e in range(NUM_SAMPLES):
    mat1 = torch.rand(DIM, DIM).cuda()
    mat2 = torch.rand(DIM, DIM).cuda()
    start_epoch = time.perf_counter()
    with TimeIt(f"matmul on 2 {DIM}x{DIM} matrix"):
        torch.mm(mat1, mat2)
    end_epoch = time.perf_counter()
    elapsed = end_epoch - start_epoch
    print(f"time measurement outside context manager: {elapsed:.4f} seconds")
    times.append(elapsed)

avg_time = sum(times)/NUM_SAMPLES
print("Average time taken: ", avg_time)
