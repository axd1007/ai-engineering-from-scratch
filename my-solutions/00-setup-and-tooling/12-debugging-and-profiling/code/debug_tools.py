import torch
import torch.nn as nn
import time
import tracemalloc
import logging

# ========================
# Part 1: debug_print
# ========================
def debug_print(name, tensor):
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, "
          f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, "
          f"has_nan={tensor.isnan().any().item()}")

print("=== Part 1: debug_print ===")
x = torch.randn(32, 128)
debug_print("input", x)
x_with_nan = x.clone()
x_with_nan[0, 0] = float('nan')
debug_print("with_nan", x_with_nan)

# ========================
# Part 2: conditional breakpoint
# ========================
print("\n=== Part 2: conditional breakpoint ===")

def training_step(model, batch, criterion, optimizer):
    inputs, labels = batch
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    if loss.item() > 100 or torch.isnan(loss):
        print(f"Bad loss detected: {loss.item()} — breakpoint would trigger here")
        # breakpoint()  # раскомментировать для интерактивной отладки

    return loss

model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# нормальный батч
batch = (torch.randn(8, 10), torch.randn(8, 1))
loss = training_step(model, batch, criterion, optimizer)
print(f"Normal loss: {loss.item():.4f}")

# плохой батч с NaN
bad_batch = (torch.full((8, 10), float('nan')), torch.randn(8, 1))
loss = training_step(model, bad_batch, criterion, optimizer)
print(f"Bad loss: {loss.item()}")

# ========================
# Part 3: logging
# ========================
print("\n=== Part 3: logging ===")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

lr, batch_size = 0.001, 32
logger.info("Starting training: lr=%.4f, batch_size=%d", lr, batch_size)
logger.warning("Loss spike detected: %.4f at step %d", 15.3, 42)
logger.error("NaN loss at step %d, stopping", 100)

print("Log file written to /tmp/training.log")

# ========================
# Part 4: Timer
# ========================
print("\n=== Part 4: Timer ===")

class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.4f}s")

model = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 10))
data = torch.randn(256, 512)

with Timer("forward pass"):
    output = model(data)

with Timer("backward pass"):
    loss = output.sum()
    loss.backward()

with Timer("data simulation"):
    time.sleep(0.1)
    batch = torch.randn(256, 512)

# ========================
# Part 6: Memory profiling
# ========================
print("\n=== Part 6: Memory ===")

# CPU memory
tracemalloc.start()
model = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 10))
data = torch.randn(512, 1024)
output = model(data)

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")
print("Top 3 memory allocations:")
for stat in top_stats[:3]:
    print(f"  {stat}")

# GPU memory
if torch.cuda.is_available():
    print(f"\nGPU Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Cached:    {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# GPU memory test
if torch.cuda.is_available():
    print("\n=== GPU Memory Test ===")
    model_gpu = nn.Sequential(nn.Linear(1024, 4096), nn.ReLU(), nn.Linear(4096, 10)).cuda()
    data_gpu = torch.randn(512, 1024).cuda()
    output_gpu = model_gpu(data_gpu)
    print(f"GPU Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    print(f"GPU Cached:    {torch.cuda.memory_reserved() / 1e9:.3f} GB")
    del model_gpu, data_gpu, output_gpu
    torch.cuda.empty_cache()
    print(f"After cleanup - Allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")

# ========================
# Part 7: Common AI bugs
# ========================
print("\n=== Part 7: Common bugs ===")

# 1. Shape check
def check_shapes(model, sample_input):
    print(f"Input: {sample_input.shape}")
    hooks = []
    def make_hook(name):
        def hook(module, inp, out):
            if hasattr(out, 'shape'):
                print(f"  {name}: {inp[0].shape} -> {out.shape}")
        return hook
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(sample_input)
    for h in hooks:
        h.remove()

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
print("Shape trace:")
check_shapes(model, torch.randn(4, 10))

# 2. NaN detector
def detect_nan(model, loss, step):
    if torch.isnan(loss):
        print(f"NaN loss at step {step}")
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"  NaN gradient in {name}")
        return True
    return False

loss_nan = torch.tensor(float('nan'))
detect_nan(model, loss_nan, step=42)

# 3. Wrong device
def check_devices(model, *tensors):
    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    for i, t in enumerate(tensors):
        status = "OK" if t.device == model_device else f"WARNING: tensor on {t.device}"
        print(f"  tensor {i}: {t.device} — {status}")

print("\nDevice check:")
check_devices(model, torch.randn(4, 10), torch.randn(4, 10).cuda() if torch.cuda.is_available() else torch.randn(4, 10))

# ========================
# Part 8: TensorBoard
# ========================
print("\n=== Part 8: TensorBoard ===")

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("/tmp/runs/experiment_1")

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for step in range(100):
    data = torch.randn(32, 10)
    target = torch.randn(32, 1)
    output = model(data)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar("loss/train", loss.item(), step)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

    if step % 20 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, step)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, step)

writer.close()
print("TensorBoard logs written to /tmp/runs/experiment_1")
print("Launch with: tensorboard --logdir=/tmp/runs")
