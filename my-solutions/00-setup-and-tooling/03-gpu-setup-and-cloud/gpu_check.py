import torch
import time

def benchmark(size=5000, runs=3):
    print(f"{'='*40}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"{'='*40}\n")

    times_cpu = []
    for _ in range(runs):
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        start = time.time()
        _ = a @ b
        times_cpu.append(time.time() - start)
    cpu_avg = sum(times_cpu) / runs
    print(f"CPU avg ({runs} runs): {cpu_avg:.3f}s")

    if not torch.cuda.is_available():
        print("CUDA недоступна")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    times_gpu = []
    a_gpu = a.to("cuda")
    b_gpu = b.to("cuda")
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.time()
        _ = a_gpu @ b_gpu
        torch.cuda.synchronize()
        times_gpu.append(time.time() - start)
    gpu_avg = sum(times_gpu) / runs
    print(f"GPU avg ({runs} runs): {gpu_avg:.3f}s")

    print(f"\nSpeedup: {cpu_avg / gpu_avg:.1f}x")
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"VRAM cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

if __name__ == "__main__":
    benchmark()
