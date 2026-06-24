# Cold Start Mitigation for Serverless LLMs / Serverless LLM 的冷启动缓解

> 一个 20 GB model image 从 cold 到 serving，7B 要 5-10 分钟，70B 要 20+ 分钟。在真正 serverless 世界里，这不是 warm-up，而是 outage。缓解措施分布在五层：pre-seeded node images（AWS 上 Bottlerocket、dual-volume arch）、model streaming（NVIDIA Run:ai Model Streamer，vLLM 原生）、GPU memory snapshots（Modal checkpoints，restart 最快 10x）、warm pools（`min_workers=1`）、tiered loading（ServerlessLLM 的 NVMe→DRAM→HBM pipeline，10-200x latency reduction），以及只移动 input tokens（KB）而不是 KV cache（GB）的 live migration。Modal 公布 2-4s cold start 作为下限；Baseten 默认 5-10s，pre-warming 后 sub-second。本课教你测量、预算，并叠加这几层。

**类型：** 学习
**语言：** Python（stdlib, toy cold-start path simulator）
**前置知识：** 第 17 阶段 · 02（Inference Platform Economics）, 第 17 阶段 · 03（GPU Autoscaling）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 枚举五层 cold-start mitigation，并为每层说出一个工具或模式。
- 把 70B model 的总 cold-start time 计算为 (node provision) + (weights download) + (weights load into HBM) + (engine init)。
- 解释为什么 live migration 传输 input tokens（KB）而不是 KV cache（GB），以及代价是什么（recomputation）。
- 说出 warm-pool trade-off（为空闲 GPU 付费，或接受 cold-start tail）以及 `min_workers > 0` 变成必须项的 SLA 阈值。

## The Problem / 问题

你的 serverless LLM endpoint 夜间 scale to zero。早上 8 点流量激增。第一个请求等待：

1. Karpenter 供应 GPU node：45-60s。
2. Container 拉取带 weights 的 30 GB image：120-300s。
3. Engine 把 weights 加载进 HBM：45-120s，取决于 model size 和 storage speed。
4. vLLM 或 TRT-LLM 初始化 CUDA graphs、KV cache pool、tokenizer：10-30s。

总计：220-510s（约 3-8 分钟），才返回第一个 token。你的 SLA 是 2s。你上了 warm-pool（`min_workers=1`），问题似乎消失了；但现在你 24x7 为一张空闲 GPU 付费。如果 service 有 5 个产品，每个一个 warm replica，就是 5 × 24 × 30 = 3,600 GPU-hours/month，不管有没有用户调用。

Cold-start mitigation 的目标是保留 serverless economics，同时逼近 always-on 的 latency。

## The Concept / 概念

### Layer 1 — pre-seeded node images (Bottlerocket) / 第 1 层：预置节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的 dual-volume architecture 把 OS 和 data 分离。你可以把 container image 预拉到 data volume 后创建 snapshot，并在 `EC2NodeClass` 中引用 snapshot ID。新节点启动时 weights 已在本地 NVMe 上，步骤 2 和部分步骤 3 消失。它与 Karpenter 原生配合。大型模型 cold start 通常节省 2-4 分钟。

GCP 等价模式：预烘焙 container layers 的 custom VM images。Azure：managed disk snapshots。

### Layer 2 — model streaming (Run:ai Model Streamer) / 第 2 层：model streaming（Run:ai Model Streamer）

不要等整个文件加载完再回答第一个请求，而是逐层把 weights stream 进 GPU memory，只要第一个 transformer block resident 就开始处理。NVIDIA Run:ai Model Streamer 在 vLLM 2026 中原生提供。支持 S3、GCS、本地 NVMe。通过重叠 I/O 和 compute setup，把大型模型 weight-load time 大致减半。

### Layer 3 — GPU memory snapshots (Modal) / 第 3 层：GPU memory snapshots（Modal）

Modal 在第一次加载后，对 GPU state（weights、CUDA graphs、KV cache region）做 checkpoint。后续 restarts 直接反序列化到 HBM，比重新初始化快 10x。这最接近“2 秒启动一个 warm GPU”。代价：snapshots 与 GPU topology 绑定，如果 Karpenter 把你迁到不同 SKU，就要重新 checkpoint。

### Layer 4 — warm pools (min_workers=1) / 第 4 层：warm pools（min_workers=1）

最简单的缓解：始终保持一个 replica ready。成本是一张 GPU 的 hourly rate，24x7。小模型上算术很残酷（你每小时付 $0.85-$1.50，只为避免 30s cold start），大模型上更合理（付 $4/hr 避免 5 分钟 cold start）。Warm pools 成为必需的 SLA 阈值：通常是 70B+ model 上 TTFT P99 < 60s。

### Layer 5 — tiered loading (ServerlessLLM) / 第 5 层：tiered loading（ServerlessLLM）

ServerlessLLM 把 storage 看作层级：NVMe（快但大）、DRAM（中等但可分层）、HBM（小但即时）。Weights 预加载到 DRAM，再 on-demand 加载进 HBM。论文报告，相比 naive disk-to-HBM，cold loads latency 降低 10-200x。生产采用还早，但已有 vLLM 集成。

### Layer 6 — live migration (bonus pattern) / 第 6 层：live migration（加分模式）

当节点不可用（spot eviction、node drain）时，传统模式是 cold-start 另一个 replica 并 drain request queue。Live migration 把 input tokens（kilobytes）移动到已经加载模型的目标节点，并在目标上 recompute KV cache。Recomputation 比通过网络传输 GB 级 KV cache 更便宜。适用于 disaggregated deployments。

### The warm-pool math / warm-pool 数学

对 P99 TTFT SLA 为 2s 的 service，问题不是“要不要 warm pool”，而是“多少 warm replicas，哪些路径需要”。

- 高价值 interactive paths（live chat、voice agent）：`min_workers=1-2`。
- Background batch paths（nightly classification）：接受 scale-to-zero，容忍 5-10 分钟 cold start。
- Premium tier：按 tenant 设置 `min_workers` 和 dedicated capacity。

### Measure before optimizing / 先测量再优化

一个 fresh node 上 70B model 的 cold-start anatomy（示意）：

| Phase | Time | Mitigation |
|-------|------|-----------|
| Node provision | 50s | Bottlerocket + pre-seeded image, warm pool |
| Image pull | 180s | Pre-seeded data volume (eliminate) |
| Weights to HBM | 75s | Model streamer (halve); GPU snapshot (eliminate) |
| Engine init | 20s | Persistent CUDA graph cache |
| First forward | 3s | Min inherent latency |
| **Total cold** | **328s** | |
| **Total with mitigations** | **~15s** | 22x reduction |

### Numbers you should remember / 你应该记住的数字

- Modal cold start：2-4s（with GPU snapshots）。
- Baseten default cold start：5-10s；pre-warming 后 sub-second。
- Raw 70B cold start：3-8 分钟。
- Run:ai Model Streamer：约 2x weight-load speedup。
- ServerlessLLM tiered loading：10-200x latency reduction（论文数字）。

## Build It / 动手构建

在 `code/main.py` 中把 cold-start path 拆成节点供应、镜像拉取、weights to HBM 和 engine init，再逐层打开 mitigation 看总时间如何下降。

## Use It / 应用它

`code/main.py` 建模一个 cold-start path，并分别启用/禁用各项 mitigation。它报告 total cold-start time、warm-pool cost，以及 warm pool 相比 cold-start tax 的 break-even request rate。

## Ship It / 交付它

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、model size 和 traffic shape，它会选择需要叠加的 mitigations。

## Exercises / 练习

1. 运行 `code/main.py`。计算 warm replica 比通过额外 SLO request drops 支付 cold-start tax 更便宜的 break-even request rate。
2. 你部署一个 13B model，P99 TTFT SLA 为 3s。选择能达到目标的最小 mitigation stack（层数最少）。
3. Bottlerocket pre-seeding 消除 image pull，但 weights 仍要从 snapshot 加载到 HBM。如果 snapshot-backed NVMe 读速为 7 GB/s，计算 70B model 的 wall-clock。
4. 你的 serverless provider 提供 GPU snapshots（Modal），团队拒绝的理由是“snapshots leak PII”。正反两边都论证：真实风险是什么，缓解是什么（ephemeral snapshots、encryption、namespace isolation）？
5. 设计 tiered warm-pool policy：paid users、trial users 和 batch workloads 各保留多少 warm replicas？展示数学。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Cold start | “the big pause” | fresh replica 上从请求到第一个 token 的时间 |
| Warm pool | “always-on minimum” | `min_workers >= 1`，至少保持一个 replica ready |
| Pre-seeded image | “baked AMI” | container weights 预驻留的 node image |
| Bottlerocket | “AWS node OS” | AWS container-optimized OS，支持 dual-volume snapshot |
| Model streamer | “streaming load” | 重叠 weights I/O 与 compute setup |
| GPU snapshot | “checkpoint to HBM” | 序列化 post-load GPU state；restart 时反序列化 |
| Tiered loading | “NVMe + DRAM + HBM” | 存储层级；按需加载 |
| Live migration | “move tokens” | 传输 input（KB），在目标上 recompute KV |
| `min_workers` | “warm replicas” | Serverless minimum keep-alive count |
| Scale-to-zero | “full serverless” | 空闲无成本；接受完整 cold-start tax |

## Further Reading / 延伸阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal benchmarks 和 checkpoint architecture。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — pre-seeded data volume snapshot pattern。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 重叠 weights load 与 compute setup。
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — pre-warming playbook。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — tiered loading design。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — disaggregated deployments 的 live migration。
