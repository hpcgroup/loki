import torch
import time
import numpy as np

# Try to import APEX for sparse operations
try:
    from apex.contrib.sparsity import ASP
except ImportError:
    print("NVIDIA Apex not found. Please install it using: pip install -v --no-cache-dir --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\" 'git+https://github.com/NVIDIA/apex.git'")
    exit(1)

# Create a simple model with one linear layer
class SimpleModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)

def benchmark_sparsity():
    # Parameters
    in_features = 4096
    out_features = 4096
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA is not available. This benchmark requires an NVIDIA GPU.")
        exit(1)
    
    # Create models
    dense_model = SimpleModel(in_features, out_features).to(device)
    sparse_model = SimpleModel(in_features, out_features).to(device)
    
    # ASP.prune_trained_model needs an optimizer for some reason?
    optimizer = torch.optim.SGD(sparse_model.parameters(), lr=0.01)

    # Apply 2:4 sparsity to the sparse model
    ASP.prune_trained_model(sparse_model, optimizer)
    
    # Create input tensor
    x = torch.randn(batch_size, in_features, device=device)
    
    # Warmup
    for _ in range(10):
        dense_model(x)
        sparse_model(x)
    
    # Benchmark dense model
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        dense_output = dense_model(x)
        torch.cuda.synchronize()
    dense_time = (time.time() - start_time) / 1000
    
#    # Calculate FLOPS for dense
#    # For a linear layer: 2 * batch_size * in_features * out_features
#    dense_flops = 2 * batch_size * in_features * out_features
#    dense_tflops = dense_flops / dense_time / 1e12
    
    # Benchmark sparse model
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(1000):
        sparse_output = sparse_model(x)
        torch.cuda.synchronize()
    sparse_time = (time.time() - start_time) / 1000
    
#    # Calculate FLOPS for sparse (50% of dense for 2:4 sparsity)
#    sparse_flops = dense_flops * 0.5
#    sparse_tflops = sparse_flops / sparse_time / 1e12
    
#    # Check that outputs are close
#    assert torch.allclose(dense_output, sparse_output, rtol=1e-2, atol=1e-2), "Outputs are not close!"
    
    # Print results
    print(f"Input size: {batch_size}x{in_features}, Output size: {out_features}")
    print(f"Dense time: {dense_time*1000:.3f} ms")  #, {dense_tflops:.2f} TFLOPS")
    print(f"Sparse time: {sparse_time*1000:.3f} ms")  #, {sparse_tflops:.2f} TFLOPS")
    print(f"Speedup: {dense_time/sparse_time:.2f}x")
    
#    # Try to get energy measurements
#    try:
#        import pynvml
#        pynvml.nvmlInit()
#        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
#        # Dense energy
#        torch.cuda.synchronize()
#        pynvml.nvmlDeviceResetApplicationsClocks(handle)
#        start_time = time.time()
#        start_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
        
#        for _ in range(100):
#            dense_model(x)
#            torch.cuda.synchronize()
            
#        end_time = time.time()
#        end_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
#        dense_duration = end_time - start_time
#        dense_power = (start_power + end_power) / 2

#        dense_energy = dense_power * dense_duration
        
#        # Sparse energy
#        torch.cuda.synchronize()
#        pynvml.nvmlDeviceResetApplicationsClocks(handle)
#        start_time = time.time()
#        start_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
        
#        for _ in range(100):
#            sparse_model(x)
#            torch.cuda.synchronize()
            
#        end_time = time.time()
#        end_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
#        sparse_duration = end_time - start_time
#        sparse_power = (start_power + end_power) / 2
#        sparse_energy = sparse_power * sparse_duration
        
#        print(f"Dense power: {dense_power:.2f} W, Energy: {dense_energy:.2f} J")
#        print(f"Sparse power: {sparse_power:.2f} W, Energy: {sparse_energy:.2f} J")
#        print(f"Energy savings: {dense_energy/sparse_energy:.2f}x")
        
#        pynvml.nvmlShutdown()
#    except:
#        print("Energy measurements not available. Install pynvml for energy measurements.")

if __name__ == "__main__":
    benchmark_sparsity()
