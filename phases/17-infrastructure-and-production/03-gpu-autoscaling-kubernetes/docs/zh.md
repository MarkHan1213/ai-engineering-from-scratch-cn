# GPU Autoscaling on Kubernetes — Karpenter, KAI Scheduler, Gang Scheduling / Kubernetes 上的 GPU 自动扩缩容：Karpenter、KAI Scheduler、Gang Scheduling

> 是三层，不是一层。Karpenter 动态供应节点（1 分钟内，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 gang scheduling、topology awareness 和 hierarchical queues，避免 8 块 GPU 需要一起到位时只拿到 7 块、7 个节点空等并烧钱的 partial allocation trap。Application-level autoscalers（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于推理专用信号扩缩容：queue depth、KV cache utilization，而不是 CPU/DCGM duty cycle。经典 HPA 陷阱是 `DCGM_FI_DEV_GPU_UTIL` 是 duty-cycle measurement：100% 可能是 10 个请求，也可能是 100 个。vLLM 会预分配 KV cache memory，所以 memory 永远不会触发 scale-down。本课教你组合三层，并避开默认 Karpenter `WhenEmptyOrUnderutilized` 策略，因为它会在推理过程中终止正在运行的 GPU jobs。

**类型：** 学习
**语言：** Python（stdlib, toy queue-depth autoscaler simulator）
**前置知识：** 第 17 阶段 · 02（Inference Platform Economics）, 第 17 阶段 · 04（vLLM Serving Internals）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 画出三层 autoscaling（node provisioning、gang scheduling、application-level），并说出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 对 vLLM 来说是错误 HPA signal，并说出两个替代信号（queue depth、KV cache utilization）。
- 描述 gang scheduling，以及 KAI Scheduler 避免的 partial-allocation failure mode（8 块 GPU 只拿到 7 块，全部 idle）。
- 说出会终止运行中 GPU jobs 的 Karpenter consolidation policy（`WhenEmptyOrUnderutilized`），并说明 2026 年安全替代方案。

## The Problem / 问题

你的团队在 Kubernetes 上交付 LLM-serving service。你用 `DCGM_FI_DEV_GPU_UTIL` 作为信号配置 HPA。业务时间内 service 固定在 100% utilization。HPA 从不 scale up，因为它已经认为你“满了”。你手工加一个 replica；TTFT 降了。HPA 还是不扩。这个信号在骗你。

另一边，你用 Cluster Autoscaler 加节点。凌晨 2 点来了一个 1M-token prompt；集群花 3 分钟供应节点，请求超时。

再另一边，你部署一个需要跨 2 个节点、共 8 块 GPU 的 70B model。集群有 7 块 GPU 空闲，另 1 块分散在 3 个节点上。Cluster Autoscaler 为缺的 1 块 GPU 供应一个节点。七个节点等 4 分钟并持续烧钱，直到 Kubernetes 把最后一块 GPU 准备好。

三层，三种不同 failure modes。2026 年 GPU-aware autoscaling 不是“打开 HPA”，而是组合 node provisioning、gang scheduling 和 application-signal autoscaling。

## The Concept / 概念

### Layer 1 — node provisioning (Karpenter) / 第 1 层：节点供应（Karpenter）

Karpenter 观察 pending pods，并在约 45-60 秒内供应节点（Cluster Autoscaler 的 GPU 节点通常需要 90-120 秒）。它会按 `NodePool` constraint 动态选择 instance types：如果 pod 需要 8 个 H100，而集群没有匹配节点，Karpenter 会直接供应一个，而不是扩已有 group。

**The consolidation trap**：Karpenter 默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU pools 很危险。它会终止一个正在运行的 GPU 节点，把 pods 迁到更便宜、更合适的 instance。对 inference workloads 来说，这意味着驱逐正在运行的请求，并在新节点上重新加载 70B model。损失是数分钟容量和请求失败。

GPU pools 的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

这让 Karpenter 在节点真正空闲一小时后再 consolidate，但永远不驱逐正在运行的 job。

### Layer 2 — gang scheduling (KAI Scheduler) / 第 2 层：gang scheduling（KAI Scheduler）

KAI Scheduler（项目名曾为 “Karp”，后改名）处理默认 kube-scheduler 不处理的事情：

**Gang scheduling** — 全有或全无地调度。一个需要 8 块 GPU 的 distributed inference pod，要么 8 个都一起启动，要么一个都不启动。否则你会遇到 partial-allocation trap：8 个 pods 启动 7 个，无限等待并烧钱。

**Topology awareness** — 知道哪些 GPU 共享 NVLink，哪些在同一 rack，哪些之间有 InfiniBand。然后按拓扑放置 pods。DeepSeek-V3 67B tensor-parallel workload 必须待在一个 NVLink domain；KAI Scheduler 会尊重这一点。

**Hierarchical queues** — 多个团队争夺同一 GPU pool，带 priority 和 quota。Team A 的生产紧急情况是否会被 Team B 的 training job 预占，只能由 priority rules 决定。

KAI 与 kube-scheduler 并行部署，作为 secondary scheduler；你通过 annotation 让 workloads 使用它。Ray 和 vLLM production-stack 都有集成。

### Layer 3 — application-level signals / 第 3 层：应用级信号

**The HPA trap**：`DCGM_FI_DEV_GPU_UTIL` 是 duty-cycle metric。它衡量的是每个采样窗口里 GPU 是否在工作。100% utilization 可能表示 10 个并发请求，也可能表示 100 个；GPU 都是在忙。用 duty cycle 扩缩容就是盲扩缩。

更糟的是，vLLM 和类似 engine 会预分配 KV cache memory（最高到 `--gpu-memory-utilization`）。即使只有一个请求，memory usage 也接近 90%。基于 memory 的 HPA 永远不会 scale down。

**2026 replacement signals**：

- Queue depth（等待 prefill 的请求数）。
- KV cache utilization（分配给 active sequences 的 block 占比）。
- Per-replica P99 TTFT（你的 SLA signal）。
- Goodput（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩缩 replicas。它们在 LLM serving 上直接替代 HPA。

### When to use what / 什么时候用什么

| Scale decision | Tool |
|----------------|------|
| Add/remove nodes | Karpenter |
| Schedule multi-GPU jobs | KAI Scheduler |
| Add/remove replicas | Dynamo Planner / llm-d WVA (or custom HPA on queue depth) |
| Choose GPU type | Karpenter NodePool |
| Preempt low-priority | KAI Scheduler queues |

### Disaggregated prefill/decode complicates everything / Disaggregated prefill/decode 会让一切更复杂

如果运行 disaggregated prefill/decode（Phase 17 · 17），你有两类 pod，它们有不同扩缩容触发器：prefill pods 按 queue depth 扩，decode pods 按 KV cache pressure 扩。llm-d 会把它们作为独立 `Services` 暴露，并按角色配置 HPA。不要试图在两者前面放一个单一 HPA。

### Cold start matters here too / 冷启动在这里同样重要

Cold-start mitigation（Phase 17 · 10）会让 node provisioning time 变成用户可见。Karpenter 的 45-60 秒 warm-up，加上 20GB model load 和 engine init，意味着 from-zero request 要 2-5 分钟。对 SLO-critical paths 保持 warm pool（`min_workers=1`），或在 application layer 使用 Modal-style checkpointing。

### Numbers you should remember / 你应该记住的数字

- Karpenter node provisioning：约 45-60s；Cluster Autoscaler GPU nodes 约 90-120s。
- KAI Scheduler 避免 partial-allocation waste：7-of-8 trap。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA signal：错误；改用 queue depth 或 KV utilization。
- Karpenter `WhenEmptyOrUnderutilized`：会终止运行中 GPU jobs。Inference 使用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## Build It / 动手构建

在 `code/main.py` 中把 bursty traffic、GPU provisioning、gang scheduling 和 queue-depth scaling 放进同一个模拟器，观察单层 HPA 为什么会误判。

## Use It / 应用它

`code/main.py` 在一个 bursty GPU workload 上模拟三层 autoscaler。它比较 naive HPA（duty cycle）、queue-depth HPA 和 KAI-gang-scheduled scaling，并报告 unmet requests、idle-GPU minutes 和 composite score。

## Ship It / 交付它

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定 cluster topology、workload shape 和 SLO，它会设计三层 autoscaling plan。

## Exercises / 练习

1. 运行 `code/main.py`。在 bursty workload 下，naive duty-cycle HPA 会比 queue-depth HPA 多丢多少请求？差异从哪里来？
2. 为一个服务 Llama 3.3 70B FP8 on H100 SXM5 的集群设计 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个阻止非 GPU workload 上这些节点的 taint。
3. 团队报告 deployments 卡在 Pending，因为“GPUs available but pod won't schedule”。诊断：这是 Karpenter、kube-scheduler，还是 KAI Scheduler？哪些 metrics 可以确认？
4. 为 disaggregated prefill pods 选择一个 autoscale signal，再为 decode pods 选择一个不同 signal。解释理由。
5. 计算 `WhenEmptyOrUnderutilized` consolidation trap 对一个 24x7 production service 的成本：平均每天 60 次 request-dropping events，P99 TTFT > 10s。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Karpenter | “the node provisioner” | Kubernetes node autoscaler；sub-minute provisioning |
| Cluster Autoscaler | “the old scaler” | Kubernetes node autoscaler 前身；更慢，基于 group |
| KAI Scheduler | “the GPU scheduler” | 负责 gang + topology + queues 的 secondary scheduler |
| Gang scheduling | “all or nothing” | 原子调度 N 个 pods，否则全部延后 |
| Topology awareness | “rack-aware” | 按 NVLink/IB/rack placement 放置 pods |
| `DCGM_FI_DEV_GPU_UTIL` | “GPU utilization” | Duty-cycle metric；不是 LLM scaling signal |
| Queue depth | “waiting requests” | prefill-bound scaling 的正确 HPA signal |
| KV cache utilization | “memory pressure” | decode-bound scaling 的正确 HPA signal |
| Consolidation | “Karpenter consolidation” | 为更便宜 instance type 终止节点 |
| `WhenEmpty + 1h` | “safe consolidation” | 不驱逐运行中 GPU jobs 的策略 |

## Further Reading / 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — design docs 和 configuration examples。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — consolidation policy semantics 和 GPU-safe defaults。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner scaling signals。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray integration pattern。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — managed-Kubernetes-specific guidance。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler design。
