# vLLM Production Stack with LMCache KV Offloading / 使用 LMCache KV Offloading 的 vLLM Production Stack

> vLLM 的 production-stack 是参考 Kubernetes 部署：router、engines 和 observability 已经连好。LMCache 是 KV-offloading layer，它把 KV cache 从 GPU memory 中抽出，并跨 queries 和 engines 复用（先 CPU DRAM，再 disk/Ceph）。vLLM 0.11.0 KV Offloading Connector（2026 年 1 月）通过 Connector API（v0.9.0+）让这一过程变成 asynchronous 且 pluggable。Offload latency 不直接面对用户。即使没有 shared prefixes，LMCache 也有价值：当 GPU 用完 KV slots，被 preempted 的请求可以从 CPU 恢复，而不是重新计算 prefill。已发布 benchmark 在 4 台 a3-highgpu-4g 的 16x H100（80GB HBM）上显示：当 KV cache 超出 HBM 时，native CPU offload 和 LMCache 都显著提高 throughput；KV footprint 低时，所有 configs 与 baseline 相同，只有小 overhead。

**类型：** 学习
**语言：** Python（stdlib, toy KV-spill simulator）
**前置知识：** 第 17 阶段 · 04（vLLM Serving Internals）, 第 17 阶段 · 06（SGLang/RadixAttention）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 画出 vLLM production-stack layers：router、engines、KV offload、observability。
- 解释 KV Offloading Connector API（v0.9.0+），以及 0.11.0 asynchronous path 如何隐藏 offload latency。
- 量化 LMCache CPU-DRAM 什么时候有帮助（KV > HBM），什么时候只增加 overhead（KV 小到可放入 HBM）。
- 根据部署约束，在 native vLLM CPU offload 与 LMCache connector 之间选择。

## The Problem / 问题

你的 vLLM serving 显示 GPU HBM 100%，并且并发上来时出现 preemption events。请求被 evict、requeue，同一个 2K-token prompt 一分钟里被重复 prefill 四次。GPU compute 浪费在冗余 prefill 上；goodput 远低于 raw throughput。

加 GPU 成本线性增长。加 HBM 不可能。但 CPU DRAM 很便宜：单 socket 512 GB+，latency 比 HBM 差几个数量级，但足够做“临时温热”的 KV cache。

LMCache 把 KV cache 抽到 CPU DRAM，preempted requests 可以快速恢复，跨 engines 的重复 prefixes 也能共享 cache，而不需要每个 engine 都重新 prefill。

## The Concept / 概念

### vLLM production-stack / vLLM production-stack

`github.com/vllm-project/production-stack` 是参考 Kubernetes 部署：

- **Router** — cache-aware（Phase 17 · 11）。消费 KV events。
- **Engines** — vLLM workers。每 GPU 或每 TP/PP group 一个。
- **KV cache offload** — LMCache deployment 或 native connector。
- **Observability** — Prometheus scrape、Grafana dashboards、OTel traces。
- **Control plane** — service discovery、config、rolling updates。

以 Helm chart + operator 形式发布。

### The KV Offloading Connector API (v0.9.0+) / KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入 Connector API，用于 pluggable KV cache backends。Engine 把 blocks offload 到 connector；connector 存储它们（RAM、disk、object storage、LMCache）。请求需要某个 block 时，connector 再加载回来。

vLLM 0.11.0（2026 年 1 月）加入 asynchronous offload path：常见情况下 offload 可以在后台发生，engine 不会阻塞。End-to-end latency 和 throughput 仍取决于 workload shape、KV cache hit rate 与 system pressure；vLLM 自己的 notes 指出，custom-kernel offload 在低 hit rate 下可能降低 throughput，async scheduling 与 speculative decoding 之间也有已知交互问题。

### Native CPU offload vs LMCache / Native CPU offload 与 LMCache

**Native vLLM CPU offload**：engine-local。KV blocks 存在同一 engine 的 host RAM。实现快，没有 network hop。不跨 engines。

**LMCache connector**：cluster-scale。Blocks 存在共享 LMCache server（CPU DRAM + Ceph/S3 tier）。任意 engine 都可访问。已有 16x H100 benchmark。

单 engine 有 HBM pressure 时选 native。多个 engines 共享 prefixes（RAG with common system prompts、多租户共享 templates）时选 LMCache。

### Benchmark behavior / benchmark 行为

16x H100（80 GB HBM）分布在 4 台 a3-highgpu-4g 的测试：

- Low KV footprint（短 prompts、低并发）：所有 configs 与 baseline 匹配，LMCache 增加约 3-5% overhead。
- Moderate footprint：LMCache 开始在 cross-engine prefix reuse 上有帮助。
- KV exceeds HBM：native CPU offload 和 LMCache 都显著提高 throughput；LMCache gain 更大，因为跨 engine 共享。

### When LMCache is decisive / LMCache 什么时候决定性有用

- 多租户 serving 中，system prompts 跨 tenants 共享。
- RAG 中，document chunks 跨 queries 重复。
- 同一 base 上 fine-tuned variants（LoRA），base-model KV reuse 减少重复工作。
- Preemption-heavy workloads：从 CPU 恢复比 re-prefill 更便宜。

### When NOT to enable / 什么时候不要启用

- HBM pressure 小：只付 overhead，没有收益。
- Short contexts（<1K tokens）：transfer time > re-prefill。
- Single-tenant single-prompt workload：没有 reuse 可捕捉。

### Integration with disaggregated serving / 与 disaggregated serving 集成

Phase 17 · 17 disaggregated serving + LMCache 会叠加收益：KV 从 prefill pool 传到 decode pool，如果未被使用则落入 LMCache；后续 queries 从 LMCache 拉取。Phase 17 · 11 cache-aware router 可以路由到 local 或 LMCache-shared cache 命中的 engine。

### Numbers you should remember / 你应该记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（Jan 2026）：asynchronous offload path；end-to-end latency impact 取决于 workload、KV hit rate 和 system pressure（不是绝对保证）。
- 16x H100 benchmark：KV footprint 超过 HBM 时 LMCache 有帮助。
- 小 HBM pressure：3-5% overhead，无收益。

```figure
zero-sharding
```

## Build It / 动手构建

在 `code/main.py` 中模拟 HBM KV slots 被打满后的 preemption 与 re-prefill，比较 native CPU offload、LMCache 和不 offload 的差异。

## Use It / 应用它

`code/main.py` 模拟一个 preemption-heavy workload，对比有无 LMCache。它报告 avoided re-prefills、throughput gain 和 break-even HBM utilization。

## Ship It / 交付它

本课产出 `outputs/skill-vllm-stack-decider.md`。给定 workload shape 和 vLLM deployment，它会判断 native vs LMCache vs neither。

## Exercises / 练习

1. 运行 `code/main.py`。HBM utilization 到多少时 LMCache 开始划算？
2. 某 tenant 每小时 200 queries，共享 6K-token system prompt。计算每 tenant 预期 LMCache savings。
3. LMCache server 是 single point of failure。设计 HA strategy（replicas、fallback to native）。
4. LMCache 存到 spinning disk 上的 Ceph。对一个 70B FP8、4K-token KV（500 MB），read time 与 re-prefill 相比如何？
5. 论证 vLLM 0.11.0 asynchronous path 是否“免费”：overhead 藏在哪里？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Production-stack | “the reference deployment” | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | “KV backend interface” | vLLM 0.9.0+ pluggable KV store interface |
| Native CPU offload | “engine-local spill” | 在同一 engine 的 host RAM 存 KV |
| LMCache | “cluster KV cache” | CPU DRAM + disk 上的 cross-engine KV cache server |
| 0.11.0 async | “non-blocking offload” | Offload 隐藏在 engine stream 后 |
| Preemption | “evict to make room” | HBM 满时的 KV cache shuffle |
| Prefix reuse | “same system prompt” | 多个 queries 共享开头；cache hit |
| Ceph tier | “disk tier” | cache hierarchy 中低于 DRAM 的 durable storage |

## Further Reading / 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector implementation。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — asynchronous path details。
