# Multi-Region LLM Serving and KV Cache Locality / 多区域 LLM Serving 与 KV Cache Locality

> Round-robin load balancing 对 cached LLM inference 是主动有害的。请求如果没有落到持有其 prefix 的节点，就要支付完整 prefill cost：长 prompt 上 P50 约 800 ms，而 cache hit 时约 80 ms。2026 年生产模式是 cache-aware router（Rust 写的 vLLM Router、llm-d router），它消费 KV-cache events 并基于 prefix-hash match 路由。近期研究（GORGO）把 cross-region network latency 显式纳入 routing objective。商业 “cross-region inference” 产品（Bedrock cross-region inference、GKE multi-cluster gateways）把 inference 当黑盒：它们处理 availability，不处理 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月跑 us-east-1 failover，约 22 分钟。DR 现实是：32% 的 LLM DR 失败来自团队备份了 weights，却忘了 tokenizer files 或 quantization configs。

**类型：** 学习
**语言：** Python（stdlib, toy prefix-cache-aware router simulator）
**前置知识：** 第 17 阶段 · 04（vLLM Serving）, 第 17 阶段 · 06（SGLang RadixAttention）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释为什么 round-robin load balancing 会破坏 cached inference，并量化 TTFT penalty。
- 画出 cache-aware router：inputs（KV-cache events）、algorithm（prefix-hash match）、tie-breaker（GPU utilization）。
- 说出 32% DR failure driver（缺少 tokenizer files / quantization configs），并给出三文件 DR checklist。
- 区分 commercial cross-region offerings（Bedrock CRI、GKE Multi-Cluster Gateway）和 KV-aware routing。

## The Problem / 问题

你的 service 跑在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个 ALB，使用 round-robin。生产 prefix cache hit rate 降到 8%。TTFT P50 翻了三倍。vLLM logs 显示每个请求都在支付完整 prefill cost。

Round-robin 适合 stateless services。LLM inference 天生 stateful：KV cache 编码了模型已经看过的所有内容。盲路由就是路由到错误 cache。

另外，你的团队有 DR plan。你把 model weights cross-region 备份到 S3。某 region outage 发生；你尝试 failover；replica 拒绝启动。你忘了 tokenizer.json、quantization config 和 RoPE scaling config 在另一个没有同步的 bucket 里。

Multi-region LLM serving 是 cache 问题、routing 问题和 DR hygiene 问题，不是 load-balancer 问题。

## The Concept / 概念

### Cache-aware routing / Cache-aware routing

请求带着 prompt 到达。Router hash prefix（比如前 512 tokens）；它询问每个 replica：“你是否缓存了这个 prefix？”Replicas 在分配和驱逐 blocks 时，通过 pub/sub channel 发布 KV-cache events。Router 选择匹配的 replica；如果没人匹配，再按 GPU-util-based tie-breaker 回退。

**vLLM Router**（Rust，2026 production-stack）：订阅 `kv.cache.block_added` events，维护 prefix-hash → replica index，并用 O(1) lookup 路由。无匹配时回退到 least-queue-depth。

**llm-d router**：同样模式，Kubernetes-native。通过 ControlPlane API 发布 events。

**SGLang RadixAttention**（Phase 17 · 06）是 intra-replica 等价物。Cross-replica routing 严格来说在它上游。

### Numbers / 数字

2K-token prompt，Llama 3.3 70B FP8，H100 上的 TTFT P50：
- Cache hit（same replica，prefix resident）：约 80 ms。
- Cache miss（cold prefill）：约 800 ms。

10x gap。如果 router 在 replicas 间达到 60-80% prefix cache hit，就能在 N-replica capacity 下逼近 single-replica performance。如果只有 10%，就接近 naive scaling。

### Cross-region has a new constraint — network latency / Cross-region 增加了 network latency 约束

Inter-region RTT:
- us-east-1 ↔ us-west-2: ~65 ms。
- us-east-1 ↔ eu-west-1: ~75 ms。
- us-east-1 ↔ ap-southeast-1: ~220 ms。

如果 routing 把 us-east-1 请求发到 ap-southeast-1 的 hot prefix，省下的 prefill（800 → 80 ms）会被 440 ms round-trip 吃掉。GORGO（2026 research）把这一点显式化：联合最小化 `prefill_time + network_latency`，而不是只看 prefill。多数情况下答案是保持 regional routing，除非 prefix 巨大到多 MB，让 prefill 主导。

### Commercial "cross-region inference" does not help here / 商业 “cross-region inference” 不能解决这个问题

AWS Bedrock cross-region inference 会在 capacity pressure 下自动把请求路由到其他 regions。它优化 availability，不优化 TTFT，并把 inference 当黑盒。GKE Multi-Cluster Gateway 也类似：service-level failover，不理解 KV cache。

即使用这些产品，你仍然需要 app-layer cache-aware router。它们处理 “us-east-1 着火” 的情况；cache-aware routing 处理 TTFT。

### DR hygiene — the 32% missing-files problem / DR hygiene：32% 缺文件问题

2026 年常引用统计：32% 的 LLM DR 失败发生在团队备份 weights，却忘了：

- `tokenizer.json` 或 `tokenizer.model`
- Quantization configs（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- Model-specific configs（RoPE scaling、attention masks、chat templates）
- Engine config（`vllm_config.yaml`、sampling defaults、LoRA adapter manifests）

修复方式是三文件最小 DR manifest：

1. HF model repo 下所有文件（weights + configs + tokenizer）。
2. Engine-specific serving config。
3. Deployment manifest（K8s YAML、Dockerfile、dependency lock）。

再加一条：每季度跑 DR drill。JPMorgan 2024 年 11 月 us-east-1 drill 能在 22 分钟恢复，是因为 playbook 被演练过。

### Data residency is orthogonal / Data residency 是正交约束

EU customer PHI 不能离开 EU。如果 cache-aware router 为了 prefix match 把 Paris-originated request 发到 us-east-1，你就违反了 GDPR，不管 TTFT gain 有多大。先按 residency boundary 分区 router，再优化 cache。

### Numbers you should remember / 你应该记住的数字

- Cache hit vs miss TTFT gap：约 10x（2K prompt 上 80 ms vs 800 ms）。
- Inter-region RTT US-EU：约 75 ms。
- DR failure：32% 缺 tokenizer/quant configs。
- JPMorgan us-east-1 failover 2024 年 11 月：22 分钟（30-min SLA）。

## Build It / 动手构建

用 `code/main.py` 模拟多个 region 和 prefix cache 状态，让 round-robin、regional cache-aware、global cache-aware 三种 router 在同一 workload 上竞争。

## Use It / 应用它

`code/main.py` 在 multi-region workload 上模拟三种 routing strategies（round-robin、cache-aware regional、cache-aware global）。它报告 cache hit rate、TTFT P50/P99 和 cross-region bill。

## Ship It / 交付它

本课产出 `outputs/skill-multi-region-router.md`。给定 regions、residency constraints 和 SLA，它会设计 routing plan。

## Exercises / 练习

1. 运行 `code/main.py`。给定 75 ms RTT，prompt length 到多少时 cross-region routing 才胜过 local-only routing？
2. 你的 cache hit rate 从 70% 降到 12%。诊断三个可能原因，以及能确认它们的 observables。
3. 为一个 vLLM 中服务、带 5 个 LoRA adapters 的 70B AWQ-quantized model 设计 DR manifest。列出所有文件和 config。
4. 论证 Bedrock cross-region inference 对一个有严格 TTFT SLO 的 fintech 是否“足够”。引用具体行为。
5. 一个 Paris-origin request 匹配 us-east-1 中的 prefix。你会路由过去吗？写出 policy。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Cache-aware routing | “smart LB” | 按 prefix-hash match 路由到持有 KV-cache 的 replica |
| KV-cache events | “cache pub-sub” | Replicas 发布 block add/evict；router 建索引 |
| Prefix hash | “cache key” | 前 N tokens 的 hash，用作 router lookup |
| GORGO | “cross-region routing research” | arXiv 2602.11688；把 network latency 显式纳入目标 |
| Cross-region inference | “Bedrock CRI” | AWS 产品；availability failover，而非 TTFT awareness |
| DR manifest | “the backup list” | 恢复所需所有文件，不只是 weights |
| Data residency | “GDPR boundary” | 哪个 region 可以看到用户数据的法律约束 |
| RTT | “round-trip time” | 网络延迟；US-EU 75 ms，US-APAC 220 ms |
| LLM-aware LB | “cache-hit LB” | Cache-aware router 作为产品类别 |

## Further Reading / 延伸阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 带 network latency term 的 cross-region KV-cache reuse。
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — availability failover documentation。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — cache-aware router source。
