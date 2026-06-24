# Disaggregated Prefill/Decode — NVIDIA Dynamo and llm-d / Disaggregated Prefill/Decode：NVIDIA Dynamo 与 llm-d

> Prefill 是 compute-bound；decode 是 memory-bound。把两者跑在同一 GPU 上会浪费其中一种资源。Disaggregation 把它们拆到不同 pools，并通过 NIXL（RDMA/InfiniBand 或 TCP fallback）传输 KV cache。NVIDIA Dynamo（GTC 2025 announce，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上；它的 Planner Profiler + SLA Planner 会自动匹配 prefill:decode ratios 以满足 SLOs。NVIDIA 发布的 throughput gains 大致在这个量级：developer.nvidia.com（2025-06）显示 GB200 NVL72 + Dynamo 上 DeepSeek-R1 MoE 在 medium-latency regime 中约 6x improvement；Dynamo product page（developer.nvidia.com，undated）宣称 GB300 NVL72 + Dynamo 相比 Hopper，MoE throughput 最高 50x。“30x” 是 full-stack Blackwell + Dynamo + DeepSeek-R1 报告的 community aggregate；我们没有找到单一 primary source 精确说 30x，因此应把它当方向性说法。llm-d（Red Hat + AWS）是 Kubernetes-native：prefill / decode / router 都是独立 Services，并按角色 HPA。llm-d 0.5 增加 hierarchical KV offloading、cache-aware LoRA routing、UCCL networking、scale-to-zero。经济上：多个客户披露的内部汇总表明，在 $2M-class inference spend 上，从 colocated serving 切到 constant SLA 下的 Dynamo disaggregation，可节省 30–40%（即 $600-800K/year）；具体 $2M→$600-800K 是内部 composite，不是单一 published case study，只能作为数量级锚点。短 prompts（<512 tokens，短 output）不值得支付 transfer cost。

**类型：** 学习
**语言：** Python（stdlib, toy disaggregated-vs-colocated simulator）
**前置知识：** 第 17 阶段 · 04（vLLM Serving Internals）, 第 17 阶段 · 08（Inference Metrics）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释为什么 prefill 和 decode 有不同最优 GPU allocations，并量化 colocation 下的浪费。
- 画出 disaggregated architecture：prefill pool、decode pool、KV transfer via NIXL、router。
- 说出 disaggregation 不划算的条件（short prompts、short outputs）。
- 区分 NVIDIA Dynamo（stack-above）和 llm-d（Kubernetes-native），并把它们匹配到 operational context。

## The Problem / 问题

你在 8 张 H100 上跑 Llama 3.3 70B。在混合 workload（long prompts + short outputs）下，decode 期间 GPU 空闲，因为大部分 compute 已花在 prefill。另一个 workload（short prompts + long outputs）下情况相反。Colocated prefill + decode 意味着你同时 over-provision 两种资源。

预算影响：20-40% GPU time 浪费在错误资源上。你在用 H100 compute 跑 memory-bound decode，或用 H100 HBM bandwidth 跑 compute-bound prefill。两者都是昂贵浪费。

Disaggregation 把 prefill 和 decode 分到独立 pools，并按各自 bottleneck 调整大小。KV cache 通过高带宽互连从 prefill pool 传到 decode pool。

## The Concept / 概念

### Why the bottlenecks differ / 为什么瓶颈不同

**Prefill** — 对完整 input prompt 做一次 transformer forward。矩阵乘主导，compute-bound。H100 FP8 提供约 2000 TFLOPS useful throughput。Batch efficiency 好，一次 forward 处理许多 tokens。

**Decode** — 一次生成一个 token，每次 iteration 读取完整 weights。Memory-bandwidth-bound。HBM3 提供约 3 TB/s。只有在高并发下 batch efficiency 才好，因为 weights read 可在 batch 间摊销。

把它们 colocate：你买的是同时适合两者的 GPU。H100 两者都擅长，但无论哪种都同价。规模化后，你希望 prefill pool 用 H100 / compute-heavy，decode pool 用 H200 / memory-heavy，或搭配激进量化。

### The architecture / 架构

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的 inter-node transport。可用时使用 RDMA/InfiniBand，否则 TCP fallback。Transfer latency 是真实成本：70B FP8 上 4K-token prompt 的 KV cache 通常需要 20-80 ms。这也是短 prompts 不值得 disaggregation 的原因：transfer tax 超过节省。

### Dynamo vs llm-d / Dynamo 与 llm-d

**NVIDIA Dynamo**（GTC 2025 announce，1.0 GA）：
- 作为 orchestrator 位于 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 测量 workload，SLA Planner 自动配置 prefill:decode ratios。
- Rust core，Python extensibility。
- Throughput gains：NVIDIA 报告 GB200 NVL72 + Dynamo 上 DeepSeek-R1 MoE 在 medium-latency regime 中 6x（developer.nvidia.com，2025-06）；full Blackwell + Dynamo + DeepSeek-R1 stacks 的 community “up to 30x” 没有单一 primary source，应视作方向性。
- GB300 NVL72 + Dynamo：Dynamo product page 称相比 Hopper，MoE throughput 最高 50x（developer.nvidia.com，undated）。

**llm-d**（Red Hat + AWS，Kubernetes-native）：
- Prefill / decode / router 都作为独立 Kubernetes Services。
- Per-role HPA：prefill 用 queue depth，decode 用 KV utilization。
- `topologyConstraint packDomain: rack` 把 prefill+decode cliques 放在同一 rack，提高 KV transfer 带宽。
- llm-d 0.5（2026）：hierarchical KV offloading、cache-aware LoRA routing、UCCL networking、scale-to-zero。

如果你想要 managed stack-above orchestrator，用 Dynamo。如果你想要 Kubernetes-native primitives，并且已经投入 CNCF ecosystem，用 llm-d。

### Economics / 经济性

内部 composite（非单一 published case study，只能作数量级锚点）：

- Colocated serving 年推理成本 $2M。
- 切到 disaggregated with Dynamo。
- 请求量相同，P99 latency SLA 相同。
- 报告节省：$600K–$800K/year（30–40% reduction）。
- 无新增硬件。

该数字综合多个客户披露，而非单一可引用案例；最接近的已发布数据点是 Baseten 的 2x faster TTFT / 61% higher throughput with Dynamo KV routing（baseten.co，2025-10），以及 VAST + CoreWeave 在 40–60% KV hit rate 下 60–130% more tokens/$ 的预测（vastdata.com，2025-12）。节省来自 right-sizing each pool；prefill-heavy workloads（RAG with 8K+ prefixes）收益更高。

### When NOT to disaggregate / 什么时候不要 disaggregate

- Prompts < 512 tokens 且 outputs < 200 tokens：transfer tax 主导收益。
- 小集群（< 4 GPUs）：pool diversity 不够。
- 团队无法运营带 per-role scaling 的两套 GPU pools：Dynamo 有帮助，但不等于零成本。
- 没有 RDMA fabric：TCP transfer tax 更重。

### The router integrates with Phase 17 · 11 / Router 与 Phase 17 · 11 集成

Disaggregated routers 是 KV-cache-aware（Phase 17 · 11）。请求落到持有其 prefix 的 decode pool；如果无匹配，则走 prefill → decode。Hit rate 与 disaggregation 会叠加收益，cache-aware router 决定是否甚至需要新的 prefill。

### MoE on Blackwell is where the real numbers are / 真正大数字来自 Blackwell 上的 MoE

GB300 NVL72 + Dynamo 相比 Hopper baselines 显示 50x MoE throughput。MoE expert routing 在 prefill 上 compute-heavy，在 decode 上 memory-heavy（expert caches），因此 disaggregation 是双赢。2026 年 frontier model serving 以 MoE 为主（DeepSeek-V3、未来 GPT-5 variants）。

### Numbers you should remember / 你应该记住的数字

Benchmark numbers 会漂移：NVIDIA 和 inference stack 每季度都会发布更新结果。引用前要重查。

- GB200 NVL72 + Dynamo 上 DeepSeek-R1：medium-latency regime 下相比 baseline 约 6x throughput（developer.nvidia.com，2025-06）；full Blackwell + Dynamo stacks 的 community “up to 30x” 是缺少单一 primary source 的方向性聚合。
- GB300 NVL72 + Dynamo：相比 Hopper，MoE throughput 最高 50x（developer.nvidia.com，undated）。
- Savings anchor（internal composite，非单一 case study）：$2M annual spend 下 constant SLA 节省 $600-800K/year。
- Disaggregation threshold：prompts >512 tokens + outputs >200 tokens。
- KV transfer via NIXL：70B FP8 上 4K-prompt KV 约 20-80 ms。

## Build It / 动手构建

用 `code/main.py` 建立 colocated 与 disaggregated 两种服务模型，把 prompt length、output length 和 KV transfer time 作为变量，找出 crossover。

## Use It / 应用它

`code/main.py` 模拟 colocated vs disaggregated serving。它报告 throughput、cost per request 和 prompt-length crossover。

## Ship It / 交付它

本课产出 `outputs/skill-disaggregation-decider.md`。给定 workload 和 cluster，它会判断是否 disaggregate。

## Exercises / 练习

1. 运行 `code/main.py`。prompt length 到多少时 disaggregation 胜过 colocation？
2. 为一个 P99 prefix length 8K、output 300 的 RAG service 设计 prefill pool 和 decode pool。
3. Dynamo vs llm-d：一个 pure-Kubernetes shop 且没有 Python runtime preference，应选哪个？
4. 计算 KV transfer cost：70B FP8 的 4K prefill 约 500 MB KV。RDMA 100 GB/s 下 transfer = 5 ms；TCP 10 GB/s 下 = 50 ms。哪个影响你的 SLA？
5. MoE expert routing 改变 KV access patterns。每个 token 激活不同 experts 的 MoE 下，disaggregation 如何表现？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Disaggregated serving | “split prefill/decode” | 为两个阶段使用独立 GPU pools |
| NIXL | “NVIDIA transport” | Dynamo 的 inter-node KV transfer（RDMA/TCP） |
| NVIDIA Dynamo | “the orchestrator” | vLLM/SGLang/TRT-LLM 之上的 stack-above coordinator |
| llm-d | “Kubernetes native” | Red Hat + AWS K8s disaggregated stack |
| Planner Profiler | “Dynamo auto-config” | 测 workload，配置 pool ratios |
| SLA Planner | “Dynamo policy” | 自动匹配 prefill:decode 以满足 SLOs |
| `packDomain: rack` | “llm-d topology” | 把 prefill+decode 放同 rack，加速 KV |
| UCCL | “unified collective” | llm-d 0.5 的 networking layer，用于 scale-to-zero |
| MoE expert routing | “expert per token” | DeepSeek-V3 模式；disaggregation 有帮助 |

## Further Reading / 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
