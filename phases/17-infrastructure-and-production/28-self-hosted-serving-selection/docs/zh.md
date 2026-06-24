# Self-Hosted Serving Selection — llama.cpp, Ollama, TGI, vLLM, SGLang / 自托管 Serving 选型：llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年自托管推理由四个 engines 主导。按 hardware、scale 和 ecosystem 选择。**llama.cpp** 在 CPU 上最快：model support 最广，对 quantization 和 threading 控制最完整。**Ollama** 是 dev-laptop one-command install，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP serialization），prod-like load 下 throughput gap 可到 3x。**TGI entered maintenance mode December 11, 2025**：之后只有 bug fixes；raw throughput 比 vLLM 慢约 10%，但 historically observability 和 HF-ecosystem integration 最好。这个 maintenance status 让它成为长期风险；新项目默认选择 SGLang 或 vLLM 更安全。**vLLM** 是 general-purpose production default：v0.15.1（2026 年 2 月）加入 PyTorch 2.10、RTX Blackwell SM120、H200 optimization。**SGLang** 是 agentic multi-turn / prefix-heavy specialist：在 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 生产中部署 400,000+ GPUs。硬件约束：CPU-only → 只能 llama.cpp。AMD / non-NVIDIA → vLLM only（TRT-LLM 是 NVIDIA-locked）。2026 pipeline pattern：dev = Ollama，staging = llama.cpp，prod = vLLM 或 SGLang。全程使用相同 GGUF/HF weights。

**类型：** 学习
**语言：** Python（stdlib, engine-decision tree walker）
**前置知识：** 第 17 阶段引擎相关课程（04、06、07、09、18）
**时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 根据 hardware（CPU / AMD / NVIDIA Hopper / Blackwell）、scale（1 user / 100 / 10,000）和 workload（general chat / agent / long-context）选择 engine。
- 说出 2026 年 TGI maintenance-mode status（2025 年 12 月 11 日），以及为什么它让新项目偏向 vLLM 或 SGLang。
- 描述使用相同 GGUF 或 HF weights 贯穿 dev/staging/prod 的 pipeline。
- 解释为什么 “CPU only” 强制选择 llama.cpp，而 “AMD” 排除 TRT-LLM。

## The Problem / 问题

你的团队启动一个新的 self-hosted LLM project。一个工程师说 Ollama，另一个说 vLLM，第三个说“不是 TGI out of the box 就能用吗？”三个人在不同上下文里都对，但没有一个对所有场景都对。

2026 年的选择树很重要：hardware first、scale second、workload third。还有一个具体 2025 事件：TGI 在 12 月 11 日进入 maintenance mode，这改变了新项目的默认选择。

## The Concept / 概念

### The five engines / 五个 engines

| Engine | Best for | Notes |
|--------|----------|-------|
| **llama.cpp** | CPU / edge / minimal deps / widest model support | Fastest on CPU, full control |
| **Ollama** | Dev laptops, single user, one-command install | 15-30% slower than llama.cpp; 3x prod throughput gap |
| **TGI** | HF ecosystem, regulated industries | **Maintenance mode Dec 11, 2025** |
| **vLLM** | General-purpose production, 100+ users | Broad production default; v0.15.1 Feb 2026 |
| **SGLang** | Agentic multi-turn, prefix-heavy workloads | 400,000+ GPUs in production |

### Hardware-first decision / 先看硬件

**CPU only** → llama.cpp。Ollama 也能用，但更慢。其他 engine 在 CPU 上没有竞争力。

**AMD GPU** → vLLM（AMD ROCm support）。SGLang 也能用。TRT-LLM 是 NVIDIA-locked，所以排除。

**NVIDIA Hopper (H100 / H200)** → vLLM、SGLang 或 TRT-LLM。三者都是 top-tier。

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM 是 throughput leader（Phase 17 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon (M-series)** → llama.cpp（Metal）。Ollama 包装这一层。

### Scale-second decision / 再看规模

**1 user / local dev** → Ollama。一条命令，seconds 级 first-token。

**10-100 users / small team** → vLLM single-GPU。

**100-10k users / production** → vLLM production-stack（Phase 17 · 18）或 SGLang。

**10k+ users / enterprise** → vLLM production-stack + disaggregated（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### Workload-third decision / 最后看 workload

**General chat / Q&A** → vLLM 作为 broad default 胜出。

**Agentic multi-turn（tools、planning、memory）** → SGLang 的 RadixAttention（Phase 17 · 06）占优。

**RAG with heavy prefix reuse** → SGLang。

**Code generation** → vLLM 足够；SGLang 在 cache 上略好。

**Long context（128K+）** → vLLM + chunked prefill；SGLang + tiered KV。

### The TGI maintenance trap / TGI maintenance 陷阱

Hugging Face TGI 在 2025 年 12 月 11 日进入 maintenance mode，之后只做 bug fixes。历史上：observability 顶级，HF-ecosystem integration（model cards、safety tools）最佳，raw throughput 稍落后 vLLM。

2026 年新项目：默认避开 TGI。现有 TGI deployments 可以继续，但最终应迁移。SGLang 和 vLLM 是更安全默认。

### The pipeline pattern / pipeline pattern

Dev（Ollama）→ staging（llama.cpp）→ prod（vLLM）。全程使用同一 GGUF 或 HF weights。工程师在 laptop 上快速迭代；staging 镜像 production quantization；prod 是 serving target。

### Ollama caveat / Ollama caveat

Ollama 很适合 dev。不适合 shared production：Go HTTP serialization 增加 overhead，concurrency management 比 vLLM 简单，OpenTelemetry support 滞后。在它擅长的地方用 Ollama：一个用户，一条命令；shared 时切到 vLLM。

### Self-hosted vs managed is a separate decision / Self-hosted 与 managed 是另一个决策

Phase 17 · 01（managed hyperscalers）、· 02（inference platforms）覆盖 managed。本课假设你已经决定 self-host。选择 self-host 的原因：data residency、custom fine-tune、规模化 total cost ownership、domain model 不在 hosted 上。

### Numbers you should remember / 你应该记住的数字

- TGI maintenance mode：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 support。
- SGLang production footprint：400,000+ GPUs。
- Ollama vs llama.cpp throughput gap：慢 15-30%；prod load 下 3x。

```figure
data-parallel
```

## Build It / 动手构建

在 `code/main.py` 中把 hardware、scale 和 workload 输入决策树，练习先按硬件排除不可行 engine，再按规模和 prompt pattern 选生产默认。

## Use It / 应用它

`code/main.py` 是 decision-tree walker：给定 hardware + scale + workload，它会选择 engine 并解释原因。

## Ship It / 交付它

本课产出 `outputs/skill-engine-picker.md`。给定 constraints，它会选择 engine 并写出 migration plan。

## Exercises / 练习

1. 用你的 hardware / scale / workload 运行 `code/main.py`。输出是否符合你的直觉？
2. 你的 infra 有 12 张 H100 和 8 张 MI300X AMD。选什么 engine？为什么 TRT-LLM 不在表内？
3. 团队想在 2026 年用 TGI，因为“it's what we know”。论证迁移理由。
4. Ollama dev 到 vLLM prod：quantization、configuration 和 observability 会变化什么？
5. RAG product 的 P99 prefix length 为 8K，并在 tenants 间高复用。选择 engine，并与 Phase 17 · 11 + 18 叠加。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| llama.cpp | “the CPU one” | model support 最广，CPU 上最快 |
| Ollama | “the laptop one” | 一条命令安装，dev-grade throughput |
| TGI | “HF's serving” | 自 2025 年 12 月进入 maintenance mode |
| vLLM | “the default” | 2026 broad production baseline |
| SGLang | “the agentic one” | Prefix-heavy，RadixAttention |
| TRT-LLM | “NVIDIA-locked” | Blackwell throughput leader，仅 NVIDIA |
| GGUF | “llama.cpp format” | 打包 K-quant variants |
| Production-stack | “vLLM K8s” | Phase 17 · 18 reference deployment |
| Pipeline pattern | “dev→stage→prod” | Ollama → llama.cpp → vLLM，同一 weights |

## Further Reading / 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
