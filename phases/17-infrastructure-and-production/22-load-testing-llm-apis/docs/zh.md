# Load Testing LLM APIs — Why k6 and Locust Lie / LLM API 压测：为什么 k6 和 Locust 会骗你

> 传统 load testers 不是为 streaming responses、variable output lengths、token-level metrics 或 GPU saturation 设计的。多数团队会踩两个坑。GIL trap：Locust 的 token-level measurement 在 Python GIL 下运行 tokenization，高并发时会与 request generation 竞争；tokenization backlog 会抬高报告的 inter-token latency，瓶颈是你的 client，不是 server。Prompt-uniformity trap：循环中的相同 prompt 只测试 token distribution 上的一个点；真实流量长度可变，prefix matches 也多样。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 修复这一点。2026 年工具映射：LLM-specialized（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token-level accuracy；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）** 支持 streaming-aware、Kubernetes-native distributed via TestRun/PrivateLoadZone CRDs，最适合 CI/CD gates；Vegeta 适合 Go constant-rate saturation；Locust 2.43.3 只有搭配 LLM-Locust extension 才适合 streaming。Load patterns：steady-state、ramp、spike（autoscaling test）、soak（memory leaks）。

**类型：** 构建
**语言：** Python（stdlib, toy realistic-prompt generator + latency collector）
**前置知识：** 第 17 阶段 · 08（Inference Metrics）, 第 17 阶段 · 03（GPU Autoscaling）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释两个 anti-patterns（GIL trap、prompt-uniformity trap），它们会让通用 load testers 在 LLM APIs 上说谎。
- 为目的选择工具：LLMPerf（benchmark run）、k6 + streaming extension（CI gate）、guidellm（large-scale synthetic）、GenAI-Perf（NVIDIA reference）。
- 设计四种 load patterns（steady、ramp、spike、soak），并说出每种抓什么 failure mode。
- 使用 input tokens 的 mean + stddev 构造 realistic prompt distribution，而不是固定长度。

## The Problem / 问题

你用 k6 在 LLM endpoint 上压了 500 concurrent users。它撑住了。你上线。生产中 200 个真实用户就把 service 打爆：P99 TTFT 爆炸，GPUs pinned。

发生了两件事。第一，k6 发送了 500 个相同 prompts：你的 request-coalescing 和 prefix caching 让它看起来像处理了 500 个并发 decodes，实际上只处理了一个。第二，k6 不会以人眼体验的方式跟踪 streaming responses 的 inter-token latency；它看到一条 HTTP connection，而不是 500 个以不同间隔到达的 tokens。

LLM 压测是一门独立学科。

## The Concept / 概念

### The GIL trap (Locust) / GIL 陷阱（Locust）

Locust 使用 Python，并在 GIL 下 client-side 运行 tokenization。高并发时 tokenizer 会排在 request generation 后面。报告的 inter-token latency 包含 client-side tokenization backlog。你以为 server 慢，其实是 test harness 慢。

修复：LLM-Locust extension 把 tokenization 移到独立 processes，或使用 compiled-language harness（k6、使用 tokenizers.rs 的 LLMPerf）。

### The prompt-uniformity trap / prompt-uniformity 陷阱

所有已知 load testers 都允许你配置一个 prompt。在 10,000 iterations 的 loop test 中，每次发送完全相同的 prompt。Server 每次看到同一个 prefix，prefix cache hits 接近 100%，throughput 看起来很好。

修复：从 prompt distribution 中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`，提供多样长度和内容。

### Four load patterns / 四种 load patterns

1. **Steady-state** — 恒定 RPS 30-60 分钟。捕捉：baseline performance regressions。
2. **Ramp** — 15 分钟内从 0 线性升到 target RPS。捕捉：capacity breakpoint、warm-up anomalies。
3. **Spike** — 突然 3-10x RPS 持续 2 分钟再恢复。捕捉：autoscaling latency、queue saturation、cold-start impact。
4. **Soak** — steady-state 持续 4-8 小时。捕捉：memory leaks、connection-pool drift、observability overflow。

### 2026 tool mapping / 2026 工具映射

**LLMPerf**（Anyscale）— Python，但 Rust-backed tokenization。支持 mean/stddev prompts。Streaming-aware。性能运行的默认首选。

**NVIDIA GenAI-Perf** — NVIDIA reference。使用 Triton client；metrics coverage 全。注意它的 ITL 排除 TTFT；LLMPerf 包含 TTFT。同一 server 两个工具会给出不同 TPOT。

**LLM-Locust**（TrueFoundry）— 修复 GIL trap 的 Locust extension。保留熟悉的 Locust DSL + streaming metrics。

**guidellm** — large-scale synthetic benchmarking。

**k6 v2026.1.0** + **k6 Operator 1.0 GA (Sept 2025)**:
- k6 本身（Go，compiled，无 GIL）加入 streaming-aware metrics。
- k6 Operator 使用 TestRun / PrivateLoadZone CRDs 做 Kubernetes-native distributed testing。
- 最适合 CI/CD gates 和 SLA testing。

**Vegeta** — Go，比 k6 更简单。Constant-rate HTTP saturation。不理解 LLM，但适合 gateway / rate-limit testing。

**Locust 2.43.3 stock** — 对 LLM 有 GIL trap。必须搭配 LLM-Locust extension。

### SLA gate in CI / CI 中的 SLA gate

在 PR 上跑 k6：

- Baseline RPS 下每次 30-50 iterations。
- Gate：P50/P95 TTFT、5xx < 5%、TPOT under threshold。
- Breach 时 break build。

### Realistic prompt distribution / 真实 prompt distribution

从真实 traffic samples 构造（如果有），或使用已发布分布（例如 chat 用 ShareGPT prompts，code 用 HumanEval）。把 mean + stddev 喂给 LLMPerf。必须避免 loop-with-one-prompt。

### Numbers you should remember / 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：streaming-aware metrics。
- 典型 LLMPerf run：concurrency X 下 100-1000 requests。
- 典型 CI gate：每 PR 30-50 iterations。
- 四种 patterns：steady、ramp、spike、soak。

## Build It / 动手构建

在 `code/main.py` 中生成 uniform prompts 和 realistic prompt distribution 两组压测输入，比较 prefix cache、token length variance 与 client-side 测量偏差。

## Use It / 应用它

`code/main.py` 模拟带 realistic prompt distribution 的 load test，测量 effective TPOT，并展示 uniform-prompt trap。

## Ship It / 交付它

本课产出 `outputs/skill-load-test-plan.md`。给定 workload 和 SLA，它会选择工具并设计四种 load patterns。

## Exercises / 练习

1. 运行 `code/main.py`。比较 uniform vs realistic distribution，差距在哪里？
2. 写一个 CI gate 的 k6 script：100 concurrent 下 TTFT P95 < 800 ms，运行 5 分钟。
3. Soak test 显示 memory 每小时增长 50 MB。说出三个原因，以及用于区分它们的 instrumentation。
4. Spike test 从 10 RPS 到 100 RPS。如果 Karpenter + vLLM production-stack 已就位（Phase 17 · 03 + 18），预期恢复时间是多少？
5. 同一 server 上 GenAI-Perf 报 TPOT=6ms；LLMPerf 报 TPOT=11ms。解释原因。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| LLMPerf | “the LLM harness” | Anyscale benchmark tool，streaming-aware |
| GenAI-Perf | “NVIDIA tool” | NVIDIA reference harness |
| LLM-Locust | “Locust for LLMs” | 修复 GIL trap 的 Locust extension |
| guidellm | “synthetic benchmark” | Large-scale synthetic tool |
| k6 Operator | “K8s k6” | 基于 CRD 的 distributed k6 |
| GIL trap | “Python client overhead” | Tokenization backlog 抬高报告 latency |
| Prompt-uniformity trap | “single-prompt lie” | 同一 prompt loop 命中 cache，虚高 throughput |
| Steady-state | “constant load” | 固定 RPS 持续 N 分钟 |
| Ramp | “linear up” | 在指定时长内从 0 到 target |
| Spike | “burst test” | 突然放大后恢复 |
| Soak | “long test” | 数小时，用于 leak detection |

## Further Reading / 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
