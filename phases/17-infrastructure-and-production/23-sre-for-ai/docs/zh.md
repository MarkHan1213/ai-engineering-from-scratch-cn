# SRE for AI — Multi-Agent Incident Response, Runbooks, Predictive Detection / AI SRE：多 Agent 事故响应、Runbooks 与预测性检测

> AI SRE 使用以 infrastructure data（logs、runbooks、service topology）为 grounding 的 LLM，通过 RAG 自动化 investigation、documentation 和 coordination 阶段。2026 年架构模式是 multi-agent orchestration：专门 agents（logs、metrics、runbooks）由 supervisor 协调；AI 提出 hypotheses 和 queries，人类批准 judgment calls。Datadog Bits AI 和 Azure SRE Agent 已作为 managed products 发布。Runbooks 也在演进：NeuBird Hawkeye 使用 adversarial evaluation（两个模型分析同一 incident；一致 = confidence，不一致 = uncertainty）；operational memory 跨团队变动持久化。Auto-remediation 保持谨慎：AI 建议，人类批准。完全 autonomous action 范围很窄（restart pod、rollback specific deploy），带紧 guardrails；任何销售“set it and forget it”的人都在过度承诺。新前沿是 pre-incident prediction。MIT 研究报告，一个基于 historical logs + GPU temps + API error patterns 训练的 LLM，可提前 10-15 分钟预测 89% outages。预测：到 2026 年底，95% enterprise LLMs 会有 automated failover。

**类型：** 学习
**语言：** Python（stdlib, toy multi-agent incident triage simulator）
**前置知识：** 第 17 阶段 · 13（Observability）, 第 17 阶段 · 24（Chaos Engineering）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 画出 multi-agent AI SRE architecture：supervisor + specialized agents（logs、metrics、runbooks）+ human approval gate。
- 解释为什么 auto-remediation 是 narrow（restart pod、revert deploy），而不是 broad（re-architect service）。
- 说出 adversarial evaluation pattern（NeuBird Hawkeye）：两个模型一致 = confidence；不一致 = escalate。
- 引用 MIT 89% early-detection result，并说明 operational constraint：没有 actuation 的 predictions 只是 dashboards。

## The Problem / 问题

凌晨 3 点，on-call engineer 收到 page：“High error rate in checkout。”他们查看 Datadog、Loki、三份 runbooks、deploy log。30 分钟后才发现 root cause 是 KV cache spike 导致的 vLLM OOM。重启 pod，error 消失。

到 2026 年，这个调查过程的前 20 分钟可以自动化。按 service 聚合 logs、关联最近 deploys、匹配 runbooks：这些都是 RAG + tool-use。一个 supervised agent 可以在人类打开 Datadog 之前完成 first-pass triage，并展示 hypothesis。

Fully autonomous remediation 是另一件事。Restart pod：安全。Scale GPU pool：policy 允许时安全。Re-architect service：绝对不行。纪律在于画出窄边界。

## The Concept / 概念

### Multi-agent architecture / Multi-agent 架构

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor 把 incident 拆成 sub-queries。Specialized agents 拥有 tool access（log search、PromQL、doc retrieval）。Supervisor 负责 synthesis，把 hypothesis + evidence 呈现给 human。Human approve 或 redirect。

### Auto-remediation scope / Auto-remediation 范围

**Safe (narrow)**：restart pod、revert specific deploy、在预批准边界内 scale pool、启用预批准 feature flag。

**Not safe (broad)**：改变 service topology、修改 resource limits、部署新代码、修改 IAM、改数据库。

任何销售“set it and forget it”的人都在过度承诺。随着 AI SRE 成熟，safe set 会扩大，但边界真实存在。

### Adversarial evaluation (NeuBird Hawkeye) / Adversarial evaluation（NeuBird Hawkeye）

两个模型独立分析同一 incident。如果它们对 root cause 一致，confidence 高。如果不一致，把两个 hypotheses 都展示给 human 并 escalate。这个简单模式能有效过滤幻觉 root causes。

### Operational memory / Operational memory

Team turnover 是传统 SRE 的隐形杀手：tribal knowledge 会离开。AI SRE 把 runbooks + post-mortems 存入 vector DB；每次新 incident 都检索。新工程师加入时，AI 拥有完整历史。

### Pre-incident prediction / Pre-incident prediction

MIT 2025 研究：在测试集上，基于 historical logs、GPU temperatures、API error patterns 训练的 LLM，可在 outage 发生前 10-15 分钟预测 89% outages。

Reality check：没有 actuation 的 predictions 只是 dashboards。运营问题是“当我们预测到，会做什么？”Pre-emptive drain？Pager？Auto-scale？答案由 policy 决定。

### Products in 2026 / 2026 产品

- **Datadog Bits AI** — Datadog 内的 managed SRE copilot。
- **Azure SRE Agent** — Azure-native。
- **NeuBird Hawkeye** — adversarial eval + operational memory。
- **PagerDuty AIOps** — triage + deduplication。
- **Incident.io Autopilot** — incident commander + coordination。

### Runbooks as code / Runbooks as code

Runbooks 正从 Confluence pages 演进为带结构化 sections（symptom、hypothesis、verify、act）的 versioned markdown。结构化 runbooks 会让 RAG retrieval 更好。任何 AI-SRE rollout 都应先把 unstructured runbooks 转成 structured。

### Numbers you should remember / 你应该记住的数字

- MIT early-detection：89% outages，10-15 min lead time。
- Multi-agent triage：supervisor +（logs、metrics、runbooks）+ human。
- Safe auto-remediation set：restart pod、revert deploy、scale within bounds。
- Adversarial eval：两个模型独立；agreement = confidence。

## Build It / 动手构建

用 `code/main.py` 把 log agent、metric agent 和 runbook agent 的输出汇总到 supervisor，练习用 evidence rank hypotheses，而不是直接让模型给结论。

## Use It / 应用它

`code/main.py` 模拟 multi-agent triage：log agent 找到 error，metric agent 找到 CPU spike，runbook agent 匹配 known issue。Supervisor 排序 hypotheses。

## Ship It / 交付它

本课产出 `outputs/skill-ai-sre-plan.md`。给定当前 on-call、incident volume 和 team maturity，它会设计 AI SRE rollout。

## Exercises / 练习

1. 运行 `code/main.py`。如果 log agent 和 metric agent 不一致，supervisor 如何解决？
2. 为你的 service 定义三个“safe” auto-remediation actions。逐个说明理由。
3. 写一个 structured runbook template：sections、required fields、verification commands。
4. Predictive detection 提前 12 分钟触发。你的 policy 是 pager、pre-drain，还是两者？
5. 论证一个 3-person team 在 2026 年是否应该采用 AI SRE，还是等待。考虑 maturity、volume、risk。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| AI SRE | “agent for on-call” | LLM-backed incident investigation + coordination |
| Supervisor agent | “the orchestrator” | 把 incidents 拆成 sub-queries 的顶层 agent |
| Specialized agent | “domain agent” | 有工具访问权的 sub-agent（logs、metrics、runbooks） |
| Auto-remediation | “AI fixes it” | 窄范围预批准动作；不是 broad re-architecture |
| Operational memory | “vector runbooks” | 为 RAG 存入 vector DB 的 post-mortems + runbooks |
| Adversarial eval | “two-model check” | 独立分析；agreement = confidence |
| NeuBird Hawkeye | “the adversarial one” | 带 adversarial-eval + memory pattern 的产品 |
| Bits AI | “Datadog's SRE agent” | Datadog-managed AI SRE |
| Pre-incident prediction | “early detection” | outage prediction 的 10-15 min lead time |

## Further Reading / 延伸阅读

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
