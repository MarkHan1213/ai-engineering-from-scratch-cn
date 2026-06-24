# EAGLE-3 Speculative Decoding in Production / 生产中的 EAGLE-3 Speculative Decoding

> Speculative decoding 把一个快速 draft model 与 target model 配对。Draft 提议 K 个 tokens；target 用一次 forward 验证；被接受的 tokens 几乎免费。到 2026 年，EAGLE-3 是 production-grade 变体：它在 target model 的 hidden states 上训练 draft head，而不是在 raw tokens 上训练，把通用聊天的 acceptance rate alpha 推到 0.6-0.8。正确问题不是“draft 有多快”，而是“我的流量上的 alpha 是多少？”如果 alpha 低于约 0.55，在高并发下 speculative decoding 会变成负收益，因为每个 rejected draft 都会消耗第二次 target forward。本课教你先测 alpha，再开 flag。

**类型：** 学习
**语言：** Python（stdlib, toy acceptance-rate simulator）
**前置知识：** 第 17 阶段 · 04（vLLM Serving Internals）, 第 10 阶段 · 18（Multi-Token Prediction）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出 speculative decoding 的三代，并解释 EAGLE-3 相比 EAGLE-2 和经典 draft model 改了什么。
- 定义 acceptance rate alpha，根据 alpha 和 K（draft length）计算 expected speedup，并识别目标并发下的 break-even alpha。
- 解释为什么 vLLM 2026 中 speculative decoding 是 opt-in（不是 default），以及不测 alpha 就打开为什么是 production anti-pattern。
- 写一个 measurement plan：哪个 benchmark、哪种 prompt distribution、哪个 concurrency point、用哪个 metric 做 gate。

## The Problem / 问题

Decode 是 memory-bound。在 H100 上跑 Llama 3.3 70B FP8 时，每个 decoded token 都读取约 140 GB/s 的 weights 并输出一个 token。decode 时 GPU compute 几乎空闲，瓶颈是 HBM bandwidth，不是 matmul throughput。

Speculative decoding 利用这个差距。用便宜 draft model 生成 K 个 candidate tokens，然后让 target model 在一次 forward pass 中验证这 K 个 tokens。每个 verified token 实际上是免费的（摊到 target 本来就要做的 batch-of-K forward 里）。

经典 draft-model 方法使用同族较小模型（例如 Llama 3.2 1B 为 Llama 3.3 70B 起草）。它可用，但 acceptance rate 一般，因为小模型分布偏离 target。EAGLE、EAGLE-2、EAGLE-3 直接在 target model 的内部状态上训练轻量 draft head，所以 draft 分布更贴近 target。这就是 alpha 从 draft-model 的 0.4 提升到 EAGLE-3 的 0.6-0.8 的原因。

问题是：EAGLE-3 在 vLLM 2026 中是 opt-in。必须显式设置 `speculative_config`。没有 flag，就没有加速。很多团队在真实流量上没测 alpha 就打开，结果 tail latency 变差而不是变好。

## The Concept / 概念

### What speculative decoding actually buys / Speculative decoding 实际买到什么

没有 spec decode 时，每个 token 的成本是一次 target forward。使用 draft length K 和 acceptance alpha 的 spec decode 时，target forward 每次期望产生的 tokens 是 `1 + K * alpha`。speedup 是 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是 draft-plus-verify overhead。K=5、alpha=0.7 时：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。真实世界通常在 2-3x，因为生产流量 alpha 很少那么高，且 epsilon 会在高 batch size 下增长。

### Why alpha is the only metric that matters / 为什么 alpha 是唯一关键指标

Rejected tokens 不会消失；它们会迫使 target 为第一个 rejected token 再做一次 forward。在 alpha 降到 0.4 的 workload 上，你要付 draft overhead + verification + re-roll。高并发下（例如 256 concurrent），decode batch 已经足够大，“target alone” 与 “target with verify” 之间的 memory-bandwidth gap 会缩小。多数 2026 硬件上，alpha 低于 0.55 时，spec decode 是净负收益。

Alpha 会随 workload 变化。在 ShareGPT 风格通用聊天上，基于 ShareGPT 训练的 EAGLE-3 可达 0.6-0.8。在 domain-specific traffic（code、medical、legal）上，通用数据训练的 draft head 会降到 0.4-0.6。训练 domain-specific draft head 可以恢复 alpha；相比 target finetuning，这是一个轻量、快速的训练任务。

### EAGLE generations at a glance / EAGLE 各代速览

- **Classic draft model**：同族小模型。Alpha 0.3-0.5。基础设施简单：加载两个模型，draft 每次 target forward 前跑 K 次 forward。
- **EAGLE-1 (2024)**：在 target hidden states（last layer）上训练的单一 draft head。Alpha 约 0.5-0.6。target 之上少量参数开销。
- **EAGLE-2 (2025)**：adaptive draft length 和 tree-based drafts（一次 target pass 验证多个 branches）。Alpha 约 0.6-0.7。draft scheduler 更复杂。
- **EAGLE-3 (2025-2026)**：draft head 在多个 target layers 上训练（不只是 last），alignment 更好。通用聊天 alpha 约 0.6-0.8。

### The 2026 production recipe / 2026 年生产 recipe

1. 先以 plain target model 上线。测 baseline TTFT、ITL、target concurrency 下的 throughput。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 draft。重跑 benchmark。
3. 记录 acceptance rate alpha。vLLM V1 以 `spec_decode_metrics.accepted_tokens_per_request` 报告。除以 requested draft length 得到 alpha。
4. 如果生产流量分布上的 alpha < 0.55，禁用 spec decode，或训练 domain-specific EAGLE-3 draft。
5. 在生产并发下重跑。确认 P99 ITL 没变差。

### The production pitfall: P99 tail / 生产陷阱：P99 tail

启用 spec decode 后 mean ITL 会下降。但如果不调参，P99 可能变差。Rejected drafts 会触发两阶段序列（draft + verify-fail + reroll）。在 full batch 下，这两次 pass 会串行。看 P99 ITL，不要只看 P50。

### Where EAGLE-3 is already deployed / EAGLE-3 已经部署在哪里

Google 在 2025 年把 speculative decoding 部署到 AI Overviews（质量相同、响应更快）。vLLM V1 以 `speculative_config` 作为文档化接口；V1 中的 N-gram GPU speculative decoding 是兼容 chunked prefill 的变体。SGLang 支持 EAGLE-3，并把它作为 prefix-heavy workloads 的推荐 draft path。

### Break-even math in one line / 一行 break-even 数学

Expected speedup：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 得到：`alpha_breakeven = verify_overhead / K`。典型 verify_overhead ~0.15，K=5，则 `alpha_breakeven = 0.03`。但这是 raw decode math。高并发下 verify overhead 会升高，decode batch 已经把 memory reads 摊到多个 sequences 上，所以实践中的 effective alpha_breakeven 会升到约 0.45-0.55。

### When not to use speculative decoding / 什么时候不要用 speculative decoding

- Batch-1 offline generation，且 latency 不重要。用 plain target。
- 极短输出（少于 50 tokens）。Draft overhead 和 verify cost 主导。
- 没有 domain-trained draft head 的专业领域。Alpha 太低。
- vLLM v0.18.0 + draft-model spec decode + `--enable-chunked-prefill`。这个组合无法编译。文档例外是 V1 中的 N-gram GPU spec decode。

## Build It / 动手构建

使用 `code/main.py` 扫描 alpha 与 draft length K 的组合，把 speculative decoding 的收益、开销和 tail behavior 画成可比较的表。

## Use It / 应用它

`code/main.py` 会在一系列 alpha 值和 draft length K 上，模拟有无 speculative decoding 的 decode loop。它打印 break-even alpha、measured speedup 和 tail behavior。用多个 (alpha, K) 组合运行，观察 speculative decoding 在哪里不再划算。

## Ship It / 交付它

本课产出 `outputs/skill-eagle3-rollout.md`。给定 target model、traffic distribution description 和 concurrency target，它会产出 staged EAGLE-3 rollout plan：benchmark baseline、enable config、measure alpha、用 alpha >= 0.55 做 gate、观察 P99 ITL。

## Exercises / 练习

1. 运行 `code/main.py`。K=5 时，要达到 2x speedup 需要什么 alpha？3x 呢？它对 verify_overhead 有多敏感？
2. 假设生产流量 70% general chat、30% code。General chat 使用 ShareGPT 训练的 EAGLE-3 得到 alpha 0.7；code 得到 alpha 0.4。blended alpha 是多少？spec decode 是否 net-positive？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（draft model、EAGLE、N-gram），并指出哪一种兼容 chunked prefill。
4. 你启用 EAGLE-3 后看到 mean ITL 下降 25%，但 P99 ITL 上升 15%。诊断并提出缓解方案。
5. 计算 Llama 3.3 70B 的 EAGLE-3 draft head memory cost。它与运行 Llama 3.2 1B 作为 classic draft 相比如何？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Speculative decoding | “draft plus verify” | 用便宜模型提议 K 个 tokens，target 一次 forward 验证全部 |
| Acceptance rate alpha | “spec accept rate” | 被 target 接受的 draft tokens 比例；唯一关键指标 |
| Draft length K | “spec k” | 每次 target forward 前 draft 提议的 token 数；典型 4-8 |
| Verify overhead epsilon | “spec overhead” | verify-and-reroll 相比 plain target forward 的额外成本；随 batch 增长 |
| EAGLE-3 | “latest EAGLE” | 2025-2026 变体；在多个 target layers 上训练 draft head；通用聊天 alpha 0.6-0.8 |
| `speculative_config` | “vLLM spec config” | vLLM V1 中显式 opt-in；默认不开就没有加速 |
| N-gram spec decode | “N-gram draft” | GPU-side draft，使用 prompt 中 N-gram lookups；兼容 chunked-prefill |
| Break-even alpha | “no-op alpha” | spec decode 零加速时的 alpha；生产并发下要监控 |
| Rejected-draft two-pass | “reroll cost” | drafts reject 时的两次 target forward；驱动 P99 tail |

## Further Reading / 延伸阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — `speculative_config` 和 V1 中 chunked-prefill compatibility 的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 精确字段集合。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — 原始 EAGLE draft-head formulation。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — adaptive drafts 和 trees。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 带 speculative decoding 的高效 LLM system。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — production rollout checklist。
