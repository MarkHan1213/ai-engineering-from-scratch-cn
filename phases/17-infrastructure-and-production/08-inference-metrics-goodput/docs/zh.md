# Inference Metrics — TTFT, TPOT, ITL, Goodput, P99 / 推理指标：TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定一次推理部署是否真的可用。TTFT 是 prefill 加 queue 加 network。TPOT（也等价于 ITL）是每个 token 的 memory-bound decode 成本。End-to-end latency 是 TTFT 加 TPOT 乘 output length。Throughput 是整套 fleet 聚合的 tokens per second。但对产品真正重要的是 goodput：同时满足每个 SLO 的请求比例。高 throughput 但低 goodput，意味着你在处理不会及时到达用户的 tokens。2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct 的参考数字：mean TTFT 162 ms，mean TPOT 7.33 ms，mean E2E 1,093 ms。永远报告 P50、P90、P99，不要只报 mean。还要注意 measurement trap：GenAI-Perf 在 ITL 计算中排除 TTFT，LLMPerf 包含 TTFT；同一次运行，两种工具会给出不同 TPOT。

**类型：** 学习
**语言：** Python（stdlib, toy percentile calculator and goodput reporter）
**前置知识：** 第 17 阶段 · 04（vLLM Serving Internals）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 准确定义 TTFT、TPOT、ITL、E2E、throughput 和 goodput，并指出每个指标测量的组件。
- 解释为什么 mean 对 LLM serving 是错误统计量，以及如何阅读 P50/P90/P99。
- 构造一个 SLO multi-constraint（例如 TTFT<500 ms AND TPOT<15 ms AND E2E<2 s），并据此计算 goodput。
- 说出两个会对同一次运行给出不同 TPOT 的 benchmark tools，并解释原因。

## The Problem / 问题

“我们的 throughput 是 15,000 tokens per second。”所以呢？如果 40% 请求超过 2 秒 end-to-end，用户已经离开了。Throughput 本身不能告诉你产品是否可用。

推理有多个 latency 维度，每个维度的失败方式不同。Prefill 是 compute-bound，随 prompt length 增长。Decode 是 memory-bound，随 batch size 变化。Queuing delay 是运营问题。Network 是物理距离问题。你需要分别测量这些指标，需要 percentiles，还需要一个 composite 指标回答“用户是否拿到了预期体验”——这就是 goodput。

## The Concept / 概念

### TTFT — time to first token / TTFT：首 token 时间

`TTFT = queue_time + network_request + prefill_time`

长 prompt 下 prefill 主导。H100 上的 Llama-3.3-70B FP8，一个 32k prompt 纯 prefill 约 800 ms。Queue time 是负载下 scheduler behavior。Network request 是包括 TLS 在内的 wire time。TTFT 是用户在任何内容流回前看到的等待时间。

### TPOT / ITL — inter-token latency / TPOT / ITL：token 间延迟

多个名字描述同一个量。`TPOT`（time per output token）、`ITL`（inter-token latency）、`decode latency per token` 都是同一回事：第一个 token 之后，连续 streamed tokens 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一 Llama-3.3-70B H100 stack 上，使用 chunked prefill 时 TPOT mean 约 7 ms。没有 chunked prefill 时，邻近 sequence 发生长 prefill，TPOT 可飙到 50 ms。看 P99，不要只看 mean。

### E2E latency / E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

长输出（>500 tokens）下，E2E 由 TPOT 主导。短输出但长 prompt 下，E2E 由 TTFT 主导。报告 E2E 时要按 output length 条件化。

### Throughput / 吞吐

`throughput = total_output_tokens / elapsed_time`

这是聚合指标，说明 fleet efficiency。它不能说明单个请求是否健康。

### Goodput — the metric you actually care about / Goodput：你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是 multi-constraint。只有每个约束都满足，请求才算“good”。Goodput 是这个比例。60% goodput 下的高 throughput 是失败。99% goodput 下的稍低 throughput 才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 submissions 和 AI platform providers 内部 SLA tracking 使用的指标。

### Why mean is the wrong statistic / 为什么 mean 是错误统计量

LLM latency distributions 是右偏的。一个 decode batch 中，一个长 prefill 邻居可能让 500 个 tokens 的 TPOT 约 7 ms，却让 20 个 tokens 的 TPOT 约 60 ms。Mean TPOT 是 9 ms。P99 TPOT 是 65 ms。用户经常撞上 P99，这就是他们离开的原因。

永远报告三元组（P50、P90、P99）。用户体验上，P99 是你要优化的对象。

### Reference numbers — Llama-3.1-8B-Instruct on TRT-LLM, 2026 / 参考数字：Llama-3.1-8B-Instruct on TRT-LLM, 2026

- mean TTFT: 162 ms
- mean TPOT: 7.33 ms
- mean E2E: 1,093 ms
- P99 TPOT: 根据 chunked-prefill configuration 不同，在 10-25 ms 间变化。

这些是 NVIDIA 发布的 reference points。它们会随 model size（70B 会显示 3-5x）、hardware（H100 vs B200 约 3x）和 load 改变。

### The measurement trap / 测量陷阱

2026 年两个最常用 benchmark tools 会对同一次运行给出不同 TPOT：

- **NVIDIA GenAI-Perf**：ITL 计算排除 TTFT。ITL 从 token 2 开始。
- **LLMPerf**：包含 TTFT。ITL 从 token 1 开始。

对于一个 TTFT 500 ms、100 output tokens、总 decode 700 ms 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择会改变数字。

永远说明使用的工具。永远公开定义。

### Constructing an SLO / 构造 SLO

2026 年一个面向消费者的 70B chat model 的合理 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- <300-token outputs 的 E2E P99 <= 3 s。
- Goodput target >= 99%。

Enterprise SLO 会收紧 TTFT（200-400 ms），放宽 E2E。重点是写下来，同时测三者，并把 goodput 作为单一 composite 跟踪。

### How to measure / 如何测量

- 跑真实流量，或 realistic synthetic（LLMPerf 搭配 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- Benchmark run 目标设置为 2x peak concurrency。
- 运行 30-50 iterations，对合并样本取 percentiles。
- 发布时带上 tool name、tool version、model、hardware、concurrency、prompt distribution。

```figure
throughput-latency
```

## Build It / 动手构建

在 `code/main.py` 中生成带 tail spikes 的 latency trace，分别计算 TTFT、TPOT、E2E percentiles 和 goodput，体会 throughput 与产品可用性之间的差距。

## Use It / 应用它

`code/main.py` 是一个 toy goodput calculator。生成 synthetic latency distribution，应用 SLO，然后计算 goodput。它也展示同一 trace 上 GenAI-Perf vs LLMPerf 的 TPOT 差异。

## Ship It / 交付它

本课产出 `outputs/skill-slo-goodput-gate.md`。给定 workload 和 SLO，它会产出 CI/CD-ready benchmark recipe，用 goodput 而不是 throughput gate deploys。

## Exercises / 练习

1. 运行 `code/main.py`。生成带 1% tail spike 的 distribution。当你把 P99 TPOT 从 30 ms 收紧到 15 ms，goodput 如何变化？
2. Vendor 报价 “15,000 tok/s on Llama 3.3 70B H100”。信之前要问哪三个问题？
3. 为什么 chunked prefill 保护 P99 TPOT，却不保护 mean TPOT？
4. 为 voice assistant 构造一个 consumer SLO（first token 是被听见，不是被看见）。哪个 metric 最显著影响用户？
5. 阅读 LLMPerf README 和 GenAI-Perf docs。找出另外三个工具定义不一致的 metrics。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| TTFT | “time to first token” | Queue + network + prefill；长 prompt 下由 prefill 主导 |
| TPOT | “time per output token” | 第一个 token 后每个 token 的 memory-bound decode 成本 |
| ITL | “inter-token latency” | 多数工具中等同 TPOT（并非全部，见 GenAI-Perf） |
| E2E | “end to end” | TTFT + TPOT * output_len；再叠加 response-side network |
| Throughput | “tok/s” | Fleet efficiency；没有 latency percentiles 就没有意义 |
| Goodput | “SLO-met rate” | 同时满足每个 SLO constraint 的请求比例 |
| P99 | “tail” | 1-in-100 worst-case latency；用户体验指标 |
| SLO multi-constraint | “the joint” | 三个 latency bounds 的 AND；任一违背即失败 |
| GenAI-Perf vs LLMPerf | “the tool trap” | 工具对 ITL 是否包含 TTFT 的定义不同 |

## Further Reading / 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的 canonical definition。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — alternative definitions 和 measurement recipe。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — real deployments 上的 applied measurement。
- [LLMPerf](https://github.com/ray-project/llmperf) — Ray-based open-source benchmark。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA benchmark tool。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业认可的 goodput-based benchmark。
