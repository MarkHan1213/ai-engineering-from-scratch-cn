# Capstone 14 — Speculative-Decoding Inference Server / Speculative-Decoding 推理服务

> vLLM 0.7 中的 EAGLE-3 在真实流量上提供 2.5-3x throughput。P-EAGLE（AWS 2026）把 parallel speculation 又往前推了一步。SGLang 的 SpecForge 能大规模训练 draft heads。Red Hat 的 Speculators hub 发布了常见 open models 的 aligned drafts。TensorRT-LLM 把 speculative decoding 做成 NVIDIA 上的一等能力。2026 年的生产 serving stack 是 vLLM 或 SGLang + EAGLE-family drafts、FP8 或 INT4 quantization，以及基于 queue-wait 的 HPA。本 capstone 要让两个 open models 以 2.5x+ baseline throughput 服务，并给出完整 tail-latency report。

**类型：** 综合项目
**语言：** Python（serving）, C++ / CUDA（kernel inspection）, YAML（configs）
**前置知识：** 第 03 阶段（deep learning）, 第 07 阶段（transformers）, 第 10 阶段（LLMs from scratch）, 第 17 阶段（infrastructure）
**Phases exercised:** P3 · P7 · P10 · P17
**时间：** 30 小时

## Learning Objectives / 学习目标

- 理解 draft model、target verification、acceptance rate 与 tail latency 的关系
- 在 vLLM 或 SGLang 中部署 EAGLE-3 / P-EAGLE speculative decoding
- 对两个 open target models 测量 baseline、speedup、acceptance rate 和 p99 tail-latency
- 比较 ShareGPT、HumanEval 和 domain traffic 对 draft-target alignment 的影响
- 用 K8s HPA 和 cost report 交付可运营的推理服务

## Problem / 问题

Speculative decoding 在 2026 年已经商品化。EAGLE-3 draft heads 基于 target model hidden states 训练，并预测未来 N 个 tokens；target model 一次 forward pass 验证这些 tokens。60-80% 的 acceptance rate 会转化为 2-3x 的端到端 throughput。vLLM 0.7 原生集成这套能力。SGLang + SpecForge 给你训练 pipeline。Red Hat 的 Speculators 发布了 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 的 aligned drafts。

工艺在 serving operations，而不在模型本身。acceptance rate 会随着 traffic distribution 漂移（ShareGPT vs code vs domain data）。在拒绝路径上，tail latency 可能比不用 speculation 更糟，因此必须报告多个 batch size 下的 p99，而不能只看 steady-state tokens/sec。与 Anthropic / OpenAI API 的 cost per 1M tokens 对比，是报告可信度的杠杆。

## Concept / 概念

Speculative decoding 有两层。**draft** model（EAGLE-3 head、ngram 或较小的 target-aligned model）每步提出 k 个 candidate tokens。**target** model 一次验证全部 k 个 tokens；被接受的 prefix 替代 greedy path。acceptance rate 取决于 draft-target alignment 和 input distribution。

EAGLE-3 在大多数流量上优于 ngram drafts。P-EAGLE 用更深的 draft tree 做 parallel speculation。取舍在于：rejection 时 P99 latency 更高，因为 verify pass 更大。serving config 必须按 batch-size bucket 报告 latency，才能暴露这个问题。

部署在 Kubernetes 上。vLLM 0.7 每个 GPU 或 tensor-parallel shard 运行一个 replica。HPA 使用 queue-wait，而不是 CPU。FP8（Marlin）和 INT4（AWQ）quant 让 GPU memory 控制在 H100 / H200 范围内。端到端报告包括 throughput、acceptance rate、batch 1/8/32 的 p50/p99，以及 $/1M tokens。

## Architecture / 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## Stack / 技术栈

- Serving: vLLM 0.7 或 SGLang 0.4
- Speculative methods: EAGLE-3 draft heads、P-EAGLE parallel speculation、ngram fallback
- Draft training: SpecForge（SGLang）或 Red Hat Speculators
- Target models: Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- Quantization: FP8 (Marlin)、INT4 AWQ
- Deployment: Kubernetes + NVIDIA device plugin；HPA 使用 queue-wait metric
- Eval: ShareGPT、MT-Bench-v2、GSM8K、HumanEval，用于 domain-spread acceptance measurement
- Reference: TensorRT-LLM speculative decoding 作为 vendor baseline

## Build It / 动手构建

1. **Target model prep.** 选择 Llama 3.3 70B。通过 Marlin 量化到 FP8。在 1xH100（或 2x tensor-parallel）上用 vLLM 0.7 部署。

2. **Draft source.** 从 Red Hat Speculators 拉一个 aligned EAGLE-3 draft head（或用 SpecForge 训练）。加载到 vLLM 的 speculative-decoding config。

3. **Baseline numbers.** 开启 speculation 前：测 tokens/s at batch 1/8/32、p50/p99 latency、GPU utilization。发布。

4. **Enable EAGLE-3.** 翻转 config；重跑同一 benchmark。报告 speedup、acceptance rate、p99 tail-latency delta。

5. **P-EAGLE.** 启用 parallel speculation；测量 deeper draft tree 与 serial EAGLE-3。报告 P-EAGLE 帮助与伤害的拐点。

6. **Domain traffic.** 把 ShareGPT、HumanEval 和 domain-specific traffic 送进同一 server。按 distribution 测量 acceptance rate。识别 drafts 何时 drift。

7. **Second target model.** 在 Qwen3-Coder-30B MoE 上运行同一 pipeline。draft 更难（MoE routing noise）。报告。

8. **K8s HPA.** 在 K8s 中部署，并让 HPA 跟踪 `queue_wait_ms`。演示 load triples 时 scale-out。

9. **Cost comparison.** 在同一 eval 上计算 $/1M tokens，并与 Anthropic Claude Sonnet 4.7、OpenAI GPT-5.4 对比。发布。

## Use It / 应用它

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## Ship It / 交付它

`outputs/skill-inference-server.md` 描述交付物：一个经过测量的 serving stack，包含 speculative decoding、完整 benchmark report 和 K8s deployment。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Measured speedup vs baseline | 两个模型上 matched quality 的 2.5x+ throughput |
| 20 | Acceptance rate on realistic traffic | 按 distribution 的 acceptance-rate report |
| 20 | P99 tail-latency discipline | batch 1/8/32 下 speculation 前后的 p99 |
| 20 | Ops | K8s deploy、HPA on queue-wait、rollout smooth |
| 15 | Write-up and methodology | 清晰说明改了什么以及为什么 |
| **100** | | |

## Exercises / 练习

1. 测量 draft 比 target 落后一个版本时的 acceptance-rate degradation（例如 Llama 3.3 -> 3.4 drift）。构建 monitoring alert。

2. 实现 ngram-fallback：如果 EAGLE-3 acceptance 低于阈值，切到 ngram drafts。报告可靠性提升。

3. 做一个受控 MoE 实验：同一个 Qwen3-Coder-30B，分别注入 routing noise 和不注入。测量 draft acceptance sensitivity。

4. 扩展到 H200（141 GB）。报告 model-size-per-replica headroom 增益，以及能否服务 unquantized Llama 3.3 70B。

5. 在同一 H100 硬件上 benchmark TensorRT-LLM speculative decoding。报告它相对 vLLM 的优势场景。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Draft model | “Speculator” | 为 target 提出 N 个 tokens 的小模型 |
| EAGLE-3 | “2026 draft architecture” | 基于 target hidden states 训练的 draft head；约 75% acceptance |
| P-EAGLE | “Parallel speculation” | 在一次 target pass 中验证 draft branches tree |
| Acceptance rate | “Hit rate” | 不需要 resampling 就被接受的 drafted tokens 比例 |
| Quantization | “FP8 / INT4” | 用低精度 weights 让更多模型放进 GPU memory |
| Queue wait | “HPA metric” | 请求开始推理前在 pending queue 中等待的时间 |
| Speculators hub | “Aligned drafts” | Red Hat Neural Magic 为常见 open models 发布的 EAGLE drafts hub |

## Further Reading / 延伸阅读

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — reference serving stack
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — parallel speculative decoding paper + integration
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — draft-head training pipeline
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — aligned draft hub
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — vendor alternative
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — commercial reference
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — method paper
- [vLLM repository](https://github.com/vllm-project/vllm) — code and benchmarks
