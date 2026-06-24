# Chaos Engineering for LLM Production / 面向 LLM 生产系统的 Chaos Engineering

> 2026 年，面向 LLM 的 chaos engineering 是独立学科。生产中运行 experiments 前的 prerequisites：定义 SLI/SLO，trace+metric+log observability，automated rollback，runbooks，on-call。架构有四个 planes：control（experiment scheduler）、target（services、infra、data stores）、safety（guards + abort + traffic filters）、observability（metrics + traces + logs），以及反馈到 SLO adjustments 的 feedback loop。Guardrails 必须存在：如果 daily error-budget burn > 2x expected，burn-rate alerts 暂停 experiments；suppression windows + trace-ID correlation 去重 alert noise。节奏：每周 small canary + SLO review；每月 game day + postmortem；每季度 cross-team resilience audit + dependency mapping。LLM-specific experiments：memory overload、network failures、provider outages、malformed prompts、KV cache eviction storms。Tooling：Harness Chaos Engineering（LLM-derived recommendations、blast-radius downscaling、MCP tool integration）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes-native）。

**类型：** 学习
**语言：** Python（stdlib, toy chaos experiment runner）
**前置知识：** 第 17 阶段 · 23（SRE for AI）, 第 17 阶段 · 13（Observability）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出五个 chaos engineering prerequisites（SLI/SLO、observability、rollback、runbooks、on-call），并解释跳过任一项为何会破坏实践。
- 画出四个 planes（control、target、safety、observability），以及反馈到 SLO 的 loop。
- 枚举五个 LLM-specific experiments（memory overload、network fail、provider outage、malformed prompt、KV eviction storm）。
- 根据 stack 选择工具：Harness、LitmusChaos、Chaos Mesh。

## The Problem / 问题

传统栈中的 chaos testing 已经成熟。LLM stack 添加了新的 failure modes。一个带 poison character 的 4K-token prompt 会让 tokenizer 卡 12 秒。上游 provider 返回 429；你的 gateway retry；service 因 retry-amplified concurrency OOM。Burst load 下 KV cache eviction storm 导致 re-prefill cascades，进而打满 compute。

这些不会出现在 unit tests 里。Chaos engineering 是你在用户之前发现它们的方式。

## The Concept / 概念

### Prerequisites / 前置条件

没有以下条件，不要在生产跑 chaos：

1. **SLI/SLO** — 已定义 service-level indicators 和 objectives。
2. **Observability** — traces、metrics、logs 已接入 dashboards。
3. **Automated rollback** — Phase 17 · 20 policy-flag rollback。
4. **Runbooks** — 结构化，见 Phase 17 · 23。
5. **On-call** — 有人响应。

缺任一项，chaos 就会变成真实 incident。

### Four planes + feedback / 四个 planes + feedback

**Control plane** — experiment scheduler（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**Target plane** — services、pods、nodes、load balancers、data stores。

**Safety plane** — kill switch、suppression windows、blast-radius limits、error-budget gates。

**Observability plane** — 常规 metrics + trace-ID correlation，用于区分 chaos-induced 与 natural failures。

**Feedback loop** — findings 反馈到 SLO adjustment、runbook updates、code fixes。

### Guardrails are mandatory / Guardrails 必须存在

- **Burn-rate alert**：如果 daily error-budget burn 超过 expected 的 2x，暂停 experiment。
- **Suppression windows**：experiment 期间，在 blast radius 内静默非 experiment alerts。
- **Trace-ID correlation**：所有 experiment-induced errors 都带 tag，让 on-call 去重。

### Five LLM-specific experiments / 五个 LLM-specific experiments

1. **Memory overload** — 通过高并发长上下文请求制造 KV cache preemption storm。观察：service 是 graceful shed 还是 crash？

2. **Network failure** — 切断 inference gateway 与 provider 之间的连接。观察：fallback 是否在 SLA 内触发？（Phase 17 · 19）

3. **Provider outage simulation** — OpenAI 100% 429。观察：routing 是否 failover 到 Anthropic？（Phase 17 · 16、19）

4. **Malformed prompt** — 注入 tokenizer-stalling payload（例如 deeply nested unicode、huge UTF-8 codepoint）。观察：单个请求是否会锁死 worker？

5. **KV eviction storm** — 饱和 vLLM block budget 强制 eviction。观察：LMCache 恢复，还是 service degrade？

### Cadence / 节奏

- **Weekly** — staging 中 small canary experiments，可能 5% prod。
- **Monthly** — 针对具体 scenario 的 scheduled game day；跨团队参加；postmortem。
- **Quarterly** — cross-team resilience audit；dependency map update。

### Tooling / 工具

- **Harness Chaos Engineering** — commercial；AI-derived experiment recommendations；blast-radius downscaling；MCP tool integration。
- **LitmusChaos** — CNCF graduated；Kubernetes workflow-based。
- **Chaos Mesh** — CNCF sandbox；Kubernetes-native CRD style。
- **Gremlin** — commercial；支持范围广。
- **AWS FIS** / **Azure Chaos Studio** — managed cloud offerings。

### Starting small / 从小开始

第一个 experiment：在 steady traffic 下 pod-kill 一个 decode replica。观察 rerouting 和 recovery。如果它安全可控，再升级到 network chaos。

第一个 LLM-specific experiment：注入一次 provider 429，持续 5 分钟。观察 fallback。多数团队会发现 fallback 从未被完整测试。

### Numbers you should remember / 你应该记住的数字

- Four planes：control、target、safety、observability。
- Burn-rate pause：daily budget burn 超过 expected 2x。
- Cadence：weekly canary、monthly game day、quarterly audit。
- Five LLM experiments：memory、network、provider、malformed prompt、KV storm。

## Build It / 动手构建

在 `code/main.py` 中定义 experiment、blast radius、burn-rate gate 和 abort condition，先用模拟结果验证 safety plane，再设计真实 chaos run。

## Use It / 应用它

`code/main.py` 用 safety plane gates 模拟三个 chaos experiments。它报告哪些 experiments 会触发 burn-rate abort。

## Ship It / 交付它

本课产出 `outputs/skill-chaos-plan.md`。给定 stack 和 maturity，它会选择前三个 experiments 和工具。

## Exercises / 练习

1. 运行 `code/main.py`。哪个 experiment 触发 burn-rate gate？为什么？
2. 为一个 vLLM-based RAG service 设计前五个 chaos experiments，并包含 success criteria。
3. Burn-rate alert 暂停了 experiment。你如何判断 root cause：chaos 还是 natural？
4. 论证 chaos 应该在 production 还是只在 staging 运行。什么时候 production 是正确答案？
5. 说出三个 generic network-chaos 无法复现的 LLM-specific failure modes。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| SLI / SLO | “service targets” | Indicator + objective；必需前置条件 |
| Blast radius | “scope” | 受 experiment 影响的 services / users 集合 |
| Burn-rate alert | “budget gate” | error-budget burn rate > 2x expected 时触发 |
| Game day | “monthly drill” | 计划性的跨团队 chaos exercise |
| LitmusChaos | “CNCF workflow” | Graduated CNCF Kubernetes chaos tool |
| Chaos Mesh | “CNCF CRD” | CNCF sandbox Kubernetes-native chaos |
| Harness CE | “commercial AI-assisted” | 带 AI recommendations 的 Harness chaos |
| Malformed prompt | “tokenizer bomb” | 让 tokenization 卡住的输入 |
| KV eviction storm | “preemption cascade” | 大规模 eviction 触发 re-prefills |

## Further Reading / 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
