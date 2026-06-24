# Shadow Traffic, Canary Rollout, and Progressive Deployment for LLMs / LLM 的 Shadow Traffic、Canary Rollout 与 Progressive Deployment

> LLM rollout 结合了软件部署最难的部分：没有 unit tests、failure modes 分散、信号延迟。顺序是：（1）shadow mode：把 prod requests 复制给 candidate model，log 并比较，零用户影响；能抓明显 distribution issues，但不是质量保证；（2）canary rollout：按 10% → 25% → 50% → 75% → 100% progressive traffic shift，每步 gate；跟踪 latency percentiles、cost/request、error/refusal rate、output length distribution、user-feedback rate；（3）稳定确认后，对明显不同的 alternatives 做 A/B testing。Non-determinism 不可消除：相同 inputs 由于 GPU FP non-associativity 与 batch-size variance，run-to-run accuracy variation 最高可达 15%。Cost 是变量，不是常量：一个好 20% 的模型可能每次调用贵 3x。Rollback speed 具有决定性：如果 rollback 需要 redeploy，你就太慢。Policy 放在 config/flags；model 放在 registry 并 pin digests；rollback = flip policy + revert threshold + pin old model，几秒完成。

**类型：** 学习
**语言：** Python（stdlib, toy canary-progression simulator）
**前置知识：** 第 17 阶段 · 13（Observability）, 第 17 阶段 · 21（A/B Testing）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 shadow mode（zero-impact compare）、canary（live traffic progressive）和 A/B（stability-confirmed comparison）。
- 枚举五个 LLM-specific canary metrics（latency、cost/request、error/refusal、output-length distribution、user feedback）。
- 解释为什么 LLM non-determinism（最高 15%）会改变 rollout 中“stable”的含义。
- 设计一个耗时秒级（policy flip）而不是小时级（redeploy）的 rollback path。

## The Problem / 问题

你上线了一个新模型。Offline evals 显示 accuracy 提升 3%。你直接在生产打开。24 小时内，成本上升 40%，user thumbs-down 上升 8%，三个客户工单报告“weird answers”。你回滚。Redeploy 花 3 小时。周末毁了。

这一切都可以避免。Shadow mode 会在任何用户看到前抓到 40% cost spike。Canary 会在 10% 时因 thumbs-down 变化而停住。Policy-flag rollback 只要 30 秒。纪律就是填补“offline evals 看起来不错”和“真实用户满意”之间的空白。

## The Concept / 概念

### Shadow mode / Shadow mode

Candidate 接收与 production 相同的 requests；outputs 被记录，不返回给用户。零用户影响。记录：

- Output content（与 production diff）。
- Token counts（cost delta）。
- Latency。
- Refusal 和 error。

能抓：cost blow-ups、length regressions、明显 refusal changes、hard errors。不能抓：用户能感知的 quality delta。Shadow 是 smoke test，不是 quality test。

### Canary rollout / Canary rollout

带 gates 的 progressive traffic shift。典型 progression：1% → 10% → 25% → 50% → 75% → 100%。每步基于 5 个 metrics gate：

1. **Latency percentiles** — P50、P95、P99。Breach：canary P99 > 1.5x baseline。
2. **Cost per request** — blended $。Breach：>20% above baseline。
3. **Error / refusal rate** — 5xx 加显式拒绝。Breach：2x baseline。
4. **Output length distribution** — mean + P99。Breach：distributional shift。
5. **User-feedback rate** — thumbs-down / ticket filings。Breach：1.5x baseline。

### Non-determinism is the new variance / Non-determinism 是新方差

相同 inputs 会产生不同 outputs。原因：

- GPU FP non-associativity（floating-point reduction order 随 batch 改变）。
- Batch-size variance（同一 prompt 在 batch of 128 与 batch of 16 中不同）。
- Sampling（temperature > 0）。

实测：相同 eval sets 上 run-to-run accuracy variation 最高可达 15%。Rollout 中的“stable”意味着 metrics 在期望方差内，而不是与 baseline 完全相同。Gates 要高于 noise floor。

### Cost is a variable / Cost 是变量

一个好 20% 的模型可能每次调用贵 3x。Cost/request 是五个 gates 之一。上线一个“更好”但破坏 unit economics 的模型，是 rollback case。

### Rollback is the weapon / Rollback 是武器

- Policy flag（feature flag system）：在 config 中翻转百分比；几秒完成。
- Model pinning（registry digest）：pinned model 不自动升级。
- Rollback = revert flag + set pinned digest to previous。秒级，不是小时级。

如果你的 stack 要 redeploy 才能 rollback，先修它，再 rollout。

### Tooling / 工具

**Argo Rollouts** / **Flagger** — Kubernetes progressive delivery controllers。与 Istio/Linkerd weighted routing 集成。

**Istio weighted routing** — service-mesh-level traffic split。

**KServe / Seldon Core** — 带内置 canary 的 model serving。

**Feature flags** — LaunchDarkly、Flagsmith、Unleash。Policy-level flip，无需 redeploy。

### Metrics cadence / Metrics 节奏

Canary gates 每 5-15 分钟检查一次，取决于 traffic volume。1% traffic 且 10 req/min 时，每个窗口只有 50-150 data points：足够看 latency，但 user feedback 噪声很大。10% 会多约 10x。每步 progression 都要停足够久，积累足够样本。

### The A/B step is optional / A/B 步骤可选

如果新模型明显不同（行为、成本曲线、语气都不同），canary 通过后在 50% 做 A/B。只是改进版本时，canary gates 通过后直接到 100%。

### Numbers you should remember / 你应该记住的数字

- Canary progression：1% → 10% → 25% → 50% → 75% → 100%。
- Non-determinism ceiling：相同 inputs 上 run-to-run variance 最高 15%。
- Five canary metrics：latency、cost、error/refusal、output length、user feedback。
- Cost gate：>20% above baseline 是 breach。
- Rollback：秒级，不是小时级。

## Build It / 动手构建

在 `code/main.py` 里把 canary 阶段、metric gates 和 rollback policy 编成状态机，注入成本、latency 或 refusal regression，验证 rollout 会在哪里停下。

## Use It / 应用它

`code/main.py` 模拟带注入 regressions 的 canary rollout。它报告 rollout 停在哪个阶段，以及哪个 gate 触发。

## Ship It / 交付它

本课产出 `outputs/skill-rollout-runbook.md`。给定 candidate model、baseline 和 risk tolerance，它会设计 shadow→canary→100% plan。

## Exercises / 练习

1. 运行 `code/main.py`。注入 25% cost regression。canary 会停在哪个阶段？
2. 新模型 offline accuracy gain 3%，但 cost/request +18%。要不要 ship？取决于 policy，写出两条路径。
3. 设计一个 end-to-end 60 秒内完成的 rollback。列出所需基础设施。
4. Non-determinism 在你的 eval 上显示 ±7%。设置 canary gates，避免 false-alarm。你用什么 multipliers？
5. Shadow mode 在 canary 前抓到 40% cost spike。写出触发 alert 的 rule。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Shadow mode | “duplicate to new” | 零影响地把请求发给 candidate 用于 logging |
| Canary | “progressive traffic” | 带 gates 的逐步用户暴露 rollout |
| Gates | “rollout checks” | 阻止 progression 的 metric thresholds |
| Non-determinism | “LLM variance” | 不可消除的 run-to-run differences |
| Policy flag | “flag flip rollback” | Config-level rollback，秒级不是小时级 |
| Model pin | “registry digest” | 指向 model version 的 immutable reference |
| Argo Rollouts | “K8s progressive” | Kubernetes-native canary/rollback controller |
| KServe | “inference K8s” | 带 canary primitives 的 model serving |
| Istio weighted | “mesh split” | Service-mesh traffic splitter |

## Further Reading / 延伸阅读

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
