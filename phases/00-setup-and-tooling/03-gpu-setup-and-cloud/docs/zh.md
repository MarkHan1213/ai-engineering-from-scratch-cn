# GPU Setup & Cloud / GPU 设置与云环境

> 用 CPU 训练足够学习概念。真正训练时，你会需要 GPU。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 0, Lesson 01
**时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 使用 `nvidia-smi` 和 PyTorch 的 CUDA API 验证本地 GPU 是否可用
- 在 Google Colab 中配置免费的 T4 GPU，用于云端实验
- 对 CPU 与 GPU 上的矩阵乘法做基准测试，并衡量加速比
- 用 fp16 经验规则估算你的 VRAM 能容纳多大的模型

## The Problem / 问题

Phase 1-3 的大多数课程用 CPU 就能跑得很好。但当你开始训练 CNN、transformer 或 LLM（Phase 4+），就需要 GPU 加速。CPU 上需要 8 小时的训练，GPU 上可能 10 分钟就结束。

你有三个选择：本地 GPU、云 GPU，或者免费的 Google Colab。

## The Concept / 概念

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## Build It / 动手构建

### Option 1: Local NVIDIA GPU / 方案 1：本地 NVIDIA GPU

先检查你是否有可用 GPU：

```bash
nvidia-smi
```

安装带 CUDA 的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Option 2: Google Colab / 方案 2：Google Colab

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. 运行 `!nvidia-smi` 验证

你可以把本课程的 notebook 直接上传到 Colab。

### Option 3: Cloud GPU / 方案 3：云 GPU

对于 Lambda Labs、RunPod 或 Vast.ai：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### No GPU? No problem. / 没有 GPU？没关系。

大多数课程都能在 CPU 上运行。需要 GPU 的课程会明确说明，并提供 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Build It: GPU vs CPU benchmark / 动手构建：GPU 与 CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Use It / 应用它

后续课程会在需要 GPU 时明确说明。日常学习可以先用 CPU；一旦进入 CNN、Transformer、LLM 训练或较大的矩阵运算，就切换到本地 GPU、Colab 或云 GPU。

## Ship It / 交付它

这一课交付的是一段可重复运行的 GPU 检查与 benchmark：你可以用它确认当前机器是否有 CUDA、GPU 名称是什么、矩阵乘法相对 CPU 快多少。

## Exercises / 练习

1. 运行上面的 benchmark，对比 CPU 和 GPU 的耗时
2. 如果你没有 GPU，就在 Google Colab 上运行它并对比结果
3. 检查你的 GPU memory 有多大，并估算能放下的最大模型（经验规则：fp16 下每个参数约 2 bytes）

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| CUDA | “GPU programming” | NVIDIA 的并行计算平台，让代码可以在 GPU 上运行 |
| VRAM | “GPU memory” | GPU 上的视频内存，独立于系统 RAM，会限制模型大小 |
| fp16 | “Half precision” | 16-bit 浮点格式，相比 fp32 内存减半，通常精度损失很小 |
| Tensor Core | “Fast matrix hardware” | 专门用于矩阵乘法的 GPU 核心，比普通核心快 4-8 倍 |
