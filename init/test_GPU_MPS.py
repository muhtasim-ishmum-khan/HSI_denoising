import torch
import time

def create_torch_tensors(device):
    x = torch.rand((10000, 10000), dtype=torch.float32)
    y = torch.rand((10000, 10000), dtype=torch.float32)
    return x.to(device), y.to(device)

def measure_time(device_name, device):
    print(f"\nTesting on: {device_name}")
    x, y = create_torch_tensors(device)

    # Warm-up (important for GPU to get realistic timings)
    for _ in range(10):
        _ = x * y

    # Sync before timing
    if device.type != "cpu":
        torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize()

    start = time.time()
    for _ in range(10):  # Repeat to get averaged time
        _ = x * y
    if device.type != "cpu":
        torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / 10
    print(f"Average time for x * y: {avg_time:.6f} seconds")

if __name__ == "__main__":
    # Test on CPU
    measure_time("CPU", torch.device("cpu"))

    # Test on Apple Silicon GPU (MPS)
    if torch.backends.mps.is_available():
        measure_time("Apple MPS GPU", torch.device("mps"))
    else:
        print("MPS backend is not available on this system.")

    # Optional: Test on CUDA if running on NVIDIA
    if torch.cuda.is_available():
        measure_time("CUDA GPU", torch.device("cuda"))
    else:
        print("CUDA backend not available.")
