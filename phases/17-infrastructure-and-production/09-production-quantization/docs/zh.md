# Production Quantization — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4 / 生产量化：AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> Quantization format 不是通用选择，而是 hardware、serving engine 和 workload 的函数。GGUF Q4_K_M 或 Q5_K_M 统治 CPU 和 edge，通过 llama.cpp 与 Ollama 交付。GPTQ 在 vLLM 内需要同一 base 上 multi-LoRA 时胜出。AWQ 搭配 Marlin-AWQ kernels，在 7B class model 上约 741 tok/s，并拥有 INT4 下最佳 Pass@1，是 2026 年 datacenter production 默认。FP8 是 Hopper、Ada、Blackwell 上的可靠中间地带：near-lossless 且广泛支持。NVFP4 和 MXFP4（Blackwell microscaling）很激进，需要 per-block validation。两个陷阱最常咬团队：calibration dataset 必须匹配 deployment domain；KV cache 与 weight quantization 是两回事，AWQ 让“我的模型现在 4 GB”这句话忘掉了生产 batch sizes 下 10-30 GB 的 KV cache。

**类型：** 学习
**语言：** Python（stdlib，跨格式内存与吞吐量对比 toy）
**前置知识：** 第 10 阶段 · 13（Quantization foundations）, 第 17 阶段 · 04（vLLM Serving Internals）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 说出六种 production quantization formats，以及它们在 2026 年的 sweet spots。
- 根据 hardware（CPU vs GPU、Hopper vs Blackwell）、engine（vLLM、TRT-LLM、llama.cpp）和 workload（routine chat、reasoning、multi-LoRA）选择格式。
- 计算某种格式节省的 weight memory，以及未被触碰的 KV cache。
- 说出会让量化模型在 domain traffic 上退化的 calibration-dataset pitfall。

## The Problem / 问题

Quantization 降低 memory 和 HBM bandwidth，而这正是 decode 需要的。一个 FP16 70B model 是 140 GB weights。把 weights 量化到 INT4（AWQ 或 GPTQ），模型变成 35 GB，可以放进一张 H100，并留出 KV cache 空间；这很重要，因为在 128 concurrent sequences、2k context 下，KV cache 自己就有 20-30 GB。

但 quantization 不是免费的。激进量化会降低质量，尤其是 reasoning-heavy tasks。不同格式适配不同 engine。不同 hardware 原生支持不同 precisions。2026 年的 format zoo 是真实存在的，你不能复制别人的选择，必须按自己的 stack 选择。

## The Concept / 概念

### The six formats / 六种格式

| Format | Bits | Sweet spot | Engines |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, edge, laptops | llama.cpp, Ollama |
| GPTQ | 4-8 | Multi-LoRA on vLLM | vLLM, TGI |
| AWQ | 4 | Datacenter GPU production | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell datacenter | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell multi-user | TRT-LLM |
| NVFP4 | 4 | Blackwell multi-user | TRT-LLM |

### GGUF — the CPU/edge default / GGUF：CPU/edge 默认

GGUF 是 file format，不严格是量化算法本身；它把 K-quant variants（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包进一个 container。Q4_K_M 和 Q5_K_M 是生产默认，4-5 bits 下接近 BF16 质量。它是 CPU 或 edge serving 的最佳选择，因为 llama.cpp 是目前最快的 CPU inference engine。

在 vLLM 中 throughput penalty 约 93 tok/s on 7B；这个格式没有针对 GPU kernels 优化。只有 deployment target 是 CPU/edge 时使用 GGUF。否则不要用。

### GPTQ — multi-LoRA in vLLM / GPTQ：vLLM 中的 multi-LoRA 路径

GPTQ 是带 calibration pass 的 post-training quantization algorithm。Marlin kernels 让它在 GPU 上很快（相比 non-Marlin GPTQ 加速 2.6x），7B 上约 712 tok/s。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA adapters。如果你服务一个 base model 加 10-50 个 fine-tuned variants（每个作为 LoRA），GPTQ 是你的路径。截至 2026 年初，NVFP4 还不支持 LoRA。

### AWQ — the datacenter GPU default / AWQ：datacenter GPU 默认

Activation-aware Weight Quantization。在量化时保护约 1% 最显著 weights。Marlin-AWQ kernels：相比 naive 加速 10.9x。7B 上约 741 tok/s，是 INT4 formats 中 Pass@1 最好者。

新 GPU serving 默认选 AWQ，除非你需要 multi-LoRA（GPTQ）或激进 Blackwell FP4（NVFP4）。

### FP8 — the reliable middle / FP8：可靠中间地带

8-bit floating point。Near-lossless。广泛支持。Hopper Tensor Cores 原生加速 FP8。Blackwell 继承。质量不可妥协时（reasoning、medical、code-gen），FP8 是 2026 年安全默认。Memory savings 只有 INT4 的一半，但质量风险低得多。

### MXFP4 / NVFP4 — Blackwell aggressive / MXFP4 / NVFP4：Blackwell 激进选项

Microscaling FP4。每个 weights block 有自己的 scale factor。在 Blackwell Tensor Cores 上硬件加速。相比 FP8，每 token 字节数减半；这就是 Phase 17 · 07 的经济收益来源。

Caveats:
- 还没有 LoRA support（2026 年初）。
- reasoning-heavy workloads 上质量下降明显。
- 每个模型都要在自己的 eval set 上验证。

### The calibration trap / calibration 陷阱

AWQ 和 GPTQ 都需要 calibration dataset，通常是 C4 或 WikiText。对 domain models（code、medical、legal），用 generic web text 做 calibration 会让算法错误判断哪些 weights 需要保护。HumanEval Pass@1 可能下降几个点。

修复方式：用 in-domain data calibration。通常几百条 domain samples 就足够。上线前在 eval set 上测试。

### The KV cache trap / KV cache 陷阱

AWQ 把 weights 缩到 4 bits。KV cache 是独立的，仍保持 FP16/FP8。对一个 70B model 使用 AWQ：

- Weights：约 35 GB（从 140 GB INT4）。
- KV cache at 128 concurrent × 2k context：约 20 GB。
- Activations：约 5 GB。
- Total：约 60 GB，可以放进 H100 80GB。

天真地说“我把模型量化到 4 GB”会忘记另外 30-50 GB。要整体预算 HBM。

另外，KV cache quantization（FP8 KV 或 INT8 KV）是另一项选择，有独立 tradeoffs；它直接影响 attention accuracy，不是免费胜利。

### AWQ INT4 is hazardous for reasoning / AWQ INT4 对 reasoning 有风险

Chain-of-thought、math、long-context code-gen 会明显受到激进量化影响。AWQ INT4 在 MATH 上会丢约 3-5 points。对 reasoning-heavy workloads，发布 FP8 或 BF16，接受 memory cost。

### 2026 picking guide / 2026 选择指南

- CPU/edge serve：GGUF Q4_K_M。结束。
- GPU serve、routine chat、no LoRA：AWQ。
- GPU serve、multi-LoRA：GPTQ with Marlin。
- Reasoning workload：FP8。
- Blackwell datacenter、quality 已验证：NVFP4 + FP8 KV。
- 不确定：对每个 candidate format 跑 1,000-sample eval。

```figure
gpu-memory-breakdown
```

## Build It / 动手构建

用 `code/main.py` 把 weights、KV cache 和 activations 分开算 HBM footprint，避免只看量化后权重大小而误判是否能放进单卡。

## Use It / 应用它

`code/main.py` 会在一系列 model sizes 上，跨六种格式计算 memory footprint（weights + KV + activations）和 relative throughput。它展示 KV cache 什么时候主导、weight compression 什么时候生效、FP8 什么时候是安全选择。

## Ship It / 交付它

本课产出 `outputs/skill-quantization-picker.md`。给定 hardware、model size、workload type 和 quality tolerance，它会选择格式，并产出 calibration/validation plan。

## Exercises / 练习

1. 运行 `code/main.py`。对一个 70B model，在 128 concurrent、2k context 下，计算每种格式的 total HBM。哪种格式能放进一张 H100 80GB？
2. 你有一个 7B coding model。选择格式并说明理由。如果你误判了 quality tolerance，恢复路径是什么？
3. 计算 medical domain model 做 AWQ calibration 所需 calibration-dataset size。为什么更多数据不总是更好？
4. 阅读 Marlin-AWQ kernel paper 或 release notes。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s，而 raw GPTQ 约 712。
5. 什么时候组合 AWQ weights + FP8 KV cache，比保持 KV at BF16 更有意义？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| GGUF | “llama.cpp format” | 打包 K-quant variants 的 file format；CPU/edge 默认 |
| Q4_K_M | “Q4 K M” | 4-bit K-quant medium；production GGUF 默认 |
| GPTQ | “gee pee tee q” | 带 calibration 的 post-train INT4；vLLM 中支持 LoRA |
| AWQ | “a w q” | Activation-aware INT4；Marlin kernels；INT4 下 Pass@1 最好 |
| Marlin kernels | “fast INT4 kernels” | Hopper 上的 custom CUDA INT4 kernels；10x speedup |
| FP8 | “eight-bit float” | Hopper/Ada/Blackwell 上的安全 precision 默认 |
| MXFP4 / NVFP4 | “microscaling four” | Blackwell 4-bit FP，带 per-block scale factors |
| Calibration dataset | “cal data” | 用于选择 quantization parameters 的输入文本；必须匹配 domain |
| KV cache quantization | “KV INT8” | 独立于 weights 的选择；影响 attention accuracy |

## Further Reading / 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — comparative benchmarks。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — throughput numbers by format。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — format-by-format picking。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — supported formats 和 flags。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — original AWQ formulation。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — original GPTQ formulation。
