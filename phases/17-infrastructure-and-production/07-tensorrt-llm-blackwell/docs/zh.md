# TensorRT-LLM on Blackwell with FP8 and NVFP4 / Blackwell 上的 TensorRT-LLM：FP8 与 NVFP4

> TensorRT-LLM 是 NVIDIA-only，但它在 Blackwell 上胜出。SemiAnalysis InferenceX 在 GB200 NVL72 + Dynamo orchestration 上测得，2026 年 Q1-Q2，一个 120B model 的成本为每百万 tokens $0.012；而 H100 + vLLM 是 $0.09/M，经济差距 7x。这个 stack 是三种 floating-point regimes 的叠加：FP8 对 KV cache 和 attention kernels 仍然关键，因为它们需要动态范围；NVFP4（4-bit microscaling）处理 weights 和 activations；multi-token prediction（MTP）和 disaggregated prefill/decode 再叠加 2-3x。Day-0 model support 可以直接加载 FP4 weights，不需要 post-training conversion。2026 年工程团队的代价是：TRT-LLM 是封闭 NVIDIA stack，采用它等于用 portability 换 throughput。承诺之前先对你的 models 和 hardware mix 做数学。

**类型：** 学习
**语言：** Python（stdlib, toy FP8/NVFP4 memory and cost calculator）
**前置知识：** 第 17 阶段 · 04（vLLM Serving Internals）, 第 10 阶段 · 13（Quantization）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释为什么 weights 使用 NVFP4 时，KV cache 和 attention 仍然需要 FP8。
- 计算 frontier model 在 BF16、FP8、NVFP4 下的 HBM footprint，并推理节省来自哪里。
- 说出 TRT-LLM 使用的 Blackwell-specific features（day-0 FP4、MTP、disaggregated serving、all-to-all primitives）。
- 判断 TRT-LLM 的 NVIDIA-lock 何时值得换取相对 Hopper 上 vLLM 的 7x cost gap。

## The Problem / 问题

2026 年推理经济学的 frontier 是“每美元多少 tokens”。答案取决于四个叠加选择：硬件代际（Hopper H100/H200 vs Blackwell B200/GB200）、precision（BF16 → FP8 → NVFP4）、serving engine（vLLM vs SGLang vs TRT-LLM）和 orchestration（plain vs disaggregated vs Dynamo）。

在 Hopper + vLLM 上，一个 120B MoE 约 $0.09 per million tokens。在 Blackwell + TRT-LLM + Dynamo 上，同一模型约 $0.012，便宜 7x。部分差距来自硬件（Blackwell 每 GPU LLM throughput 比 Hopper 高 11-15x）。部分来自 stack：FP4 weights、MTP draft、disaggregated prefill/decode，以及用于 MoE expert communication 的 NVLink 5 all-to-all。

你无法在 NVIDIA stack 外复刻它。这就是取舍：portability 换 economics。理解哪些 stack choices 贡献了差距中的哪一部分，就是本课重点。

## The Concept / 概念

### Why FP8 is still the floor for KV cache / 为什么 KV cache 仍以 FP8 为底线

2026 年一个常见错误是假设 NVFP4 可以用到所有地方。不能。KV cache 需要 FP8（8-bit floating point），因为它存储的 attention keys 和 values 跨越较宽动态范围。把 KV 量化到 FP4 会导致灾难性 accuracy loss：分布尾部掉落，attention scores 崩塌。FP8 的 exponent bits 给了 KV cache 所需范围。

NVFP4（2025-2026）用于 weights 和 activations。Microscaling：每个 weights block 有自己的 scale factor，让小 blocks 能覆盖不同动态范围，而不会受到 per-tensor scale loss。对 activations 来说，FP4 能撑住是因为 layer 内 activations 范围更小。

典型 Blackwell config：

- Weights：NVFP4（4-bit microscaling）。
- Activations：NVFP4。
- KV cache：FP8。
- Attention accumulator：FP32（softmax stability）。

### The Blackwell-specific primitives TRT-LLM uses / TRT-LLM 使用的 Blackwell 专用 primitives

- **Day-0 FP4 weights**：model providers 直接发布 FP4 weights；TRT-LLM 无需 post-training conversion 即可加载。FP4 不需要 AWQ / GPTQ step。
- **Multi-token prediction (MTP)**：与 EAGLE（Phase 17 · 05）类似，但集成进 TRT-LLM build。
- **Disaggregated serving**：prefill 和 decode 分别放在不同 GPU pools，KV cache 通过 NVLink 或 InfiniBand 传输。与 Dynamo（Phase 17 · 20）同一思想。
- **All-to-all communication primitives**：NVLink 5 相比 Hopper 将 MoE expert communication latency 降低 3x。TRT-LLM 的 MoE kernels 针对此优化。
- **NVFP4 + MXFP8 microscaling**：Blackwell Tensor Cores 上硬件加速的 scale-factor handling。

### The numbers you should memorize / 你应该背下来的数字

- HGX B200 上 GPT-OSS-120B 通过 TRT-LLM 达到 $0.02/M tokens。
- GB200 NVL72 通过 Dynamo（orchestrating TRT-LLM）达到 $0.012/M tokens。
- H100 + vLLM 在可比 workload 上约 $0.09/M tokens。
- TRT-LLM 三个月更新带来 2.8x throughput gain（2026）。
- Blackwell vs Hopper：每 GPU LLM throughput 高 11-15x。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 统治所有提交任务。

### What FP4 actually costs in quality / FP4 真正牺牲什么质量

NVFP4 很激进。在 reasoning-heavy workloads（chain-of-thought、math、long-context code-gen）上，FP4 weights 会有可见退化。Per-block calibration 能缓解但不能消除。服务 reasoning models 的团队常用 FP8 weights + FP4 activations 折中，或者在 H200 上全程使用 FP8。

规则：生产采用 NVFP4 weights 前，必须在你的 eval set 上验证 task quality。

### Why this is an NVIDIA-lock decision / 为什么这是 NVIDIA-lock 决策

TRT-LLM 是 C++ + CUDA + closed-source kernels。Models 需要为特定 GPU SKU 编译。没有 AMD、没有 Intel、没有 ARM。如果你的 infra strategy 是 multi-vendor，TRT-LLM-served tier 就不成立；你仍可在混合硬件上用 vLLM 服务。如果你是 NVIDIA-only，7x gap 可以为 lock-in 买单。

### 2026 practical recipe / 2026 年实用 recipe

如果 annual inference bill 超过 $100M，继续跑 Hopper + vLLM 等于把 7-10x 留在桌上。把成本主导 workload 迁到 Blackwell + TRT-LLM + Dynamo。把 experimentation tier 保留在 H100 + vLLM，维持 model iteration speed。每个 NVFP4-converted model 上生产前都要验证 quality。

### The disaggregation bonus / Disaggregation 的额外收益

TRT-LLM 的 disaggregated serving（分离 prefill 和 decode pools）会在 Phase 17 · 20 深入讲。在 Blackwell 上，乘数会叠加：FP4 weights × MTP speedup × disaggregated placement × cache-aware routing。7x 数字假设采用完整 stack。

```figure
pipeline-parallel
```

## Build It / 动手构建

用 `code/main.py` 建一个简化成本模型：分别输入 BF16、FP8、NVFP4/FP8 的 bytes、带宽和 GPU 价格，拆出 7x 经济差距来自哪些层。

## Use It / 应用它

`code/main.py` 会在三套 stack 上计算模型的 HBM footprint、decode throughput（memory-bound regime）和 $/M-tokens：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它，观察叠加效应，以及每个变化贡献了差距的哪一部分。

## Ship It / 交付它

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定 workload、model size 和 annual token volume，它会判断 Blackwell + TRT-LLM stack 是否值得接受 NVIDIA-lock。

## Exercises / 练习

1. 运行 `code/main.py`。在一个 active parameters 为 30% 的 120B MoE 上，计算 H100 BF16、H100 FP8、B200 NVFP4/FP8 的 memory-bandwidth-limited decode throughput。最大跃迁来自哪里？
2. 某客户在 H100 + vLLM 上每年花 $2M。假设 7x economic gap，要在 12 个月内摊销迁到 TRT-LLM，需要买多少 Blackwell GPUs 才 break even？
3. NVFP4 weight conversion 后你在 MATH 上看到 accuracy 下降 3 points。说出两条恢复路径：quality-first（keep FP8 weights）和 cost-first（用 in-domain data calibration）。
4. 阅读 MLPerf v6.0 inference results。哪个 task 的 Blackwell-over-Hopper gap 最小？为什么？
5. 计算一个 405B model 在 NVFP4 weights + FP8 KV cache、128k context 下需要多少 HBM。能否放进单个 GB200 NVL72 node？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| FP8 | “eight-bit float” | 8-bit floating point；因动态范围用于 KV cache 和 attention |
| NVFP4 | “four-bit micro” | NVIDIA 的 4-bit microscaling FP format；Blackwell 上的 weights 和 activations |
| MXFP8 | “MX eight” | Microscaling FP8 variant；Blackwell Tensor Cores 硬件加速 |
| Day-0 FP4 | “ship FP4 weights” | Model providers 直接发布 FP4 weights；无 post-train conversion step |
| MTP | “multi-token prediction” | TRT-LLM 集成的 speculative-decoding draft（Phase 17 · 05） |
| Disaggregated serving | “split prefill/decode” | Prefill 和 decode 放不同 GPU pools；KV 经 NVLink/IB 传输 |
| All-to-all | “MoE expert comm” | 把 tokens 路由到 expert GPUs 的通信模式；NVLink 5 降低 3x |
| InferenceX | “SemiAnalysis inference bench” | 2026 年行业认可的 cost-per-token benchmark |

## Further Reading / 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf results。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 和 MoE kernels。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方 engine documentation。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的 disaggregated orchestration。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数字的 benchmark suite。
