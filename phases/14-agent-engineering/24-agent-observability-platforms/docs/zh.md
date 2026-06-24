# Agent Observability: Langfuse, Phoenix, Opik / Agent 可观测性：Langfuse、Phoenix、Opik

> 2026 年，三大开源 Agent observability platforms 占据主流。Langfuse（MIT）— 每月 600 万+ installs，tracing + prompt management + evals + session replay。Arize Phoenix（Elastic 2.0）— 深度 agent-specific evals、RAG relevancy、OpenInference auto-instrumentation。Comet Opik（Apache 2.0）— automated prompt optimization、guardrails、LLM-judge hallucination detection。

**类型：** 学习
**语言：** Python（stdlib）
**前置知识：** 第 14 阶段 · 23（OTel GenAI）
**时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 说出三大开源 Agent observability platforms 及其 licenses。
- 区分各平台最强的方向：Langfuse（prompt mgmt + sessions）、Phoenix（RAG + auto-instrumentation）、Opik（optimization + guardrails）。
- 解释为什么到 2026 年，89% 的组织报告已经部署 agent observability。
- 实现一个 stdlib trace-to-dashboard pipeline，并带 LLM-judge evaluation。

## The Problem / 问题

OTel GenAI（Lesson 23）给了你 schema。你还需要平台来 ingest spans、运行 evaluations、存储 prompt versions，并暴露 regressions。三个主要候选者分别强调生命周期中的不同部分。

## The Concept / 概念

### Langfuse (MIT)

- 每月 600 万+ SDK installs，19k+ GitHub stars。
- Features：tracing、带 versioning + playground 的 prompt management、evaluations（LLM-as-judge、user feedback、custom）、session replays。
- 2025 年 6 月：原商业模块（LLM-as-a-judge、annotation queues、prompt experiments、Playground）以 MIT 协议开源。
- 最强项：端到端 observability，以及紧密的 prompt-management loop。

### Arize Phoenix (Elastic License 2.0)

- 更深的 agent-specific evaluation：trace clustering、anomaly detection、RAG 的 retrieval relevancy。
- 原生 OpenInference auto-instrumentation。
- 与 managed Arize AX 搭配用于生产。
- 没有 prompt versioning — 定位是与更宽的平台配套使用的 drift/behavioral-regression tool。
- 最强项：RAG relevancy、behavioral drift、anomaly detection。

### Comet Opik (Apache 2.0)

- 通过 A/B experiments 做 automated prompt optimization。
- Guardrails（PII redaction、topical constraints）。
- LLM-judge hallucination detection。
- 来自 Comet 自身测量的 benchmark：Opik logs + evals 用 23.44s，而 Langfuse 用 327.15s（约 14x 差距）— vendor benchmarks 只当方向性信号。
- 最强项：optimization loop、automated experimentation、guardrail enforcement。

### Industry data / 行业数据

根据 Maxim（2026 field analysis）：89% 的组织已经部署 agent observability；quality issues 是生产落地的首要障碍（32% 受访者提到）。

### Picking one / 如何选择

| Need | Pick |
|------|------|
| All-in-one with prompt management | Langfuse |
| Deep RAG evaluation + drift | Phoenix |
| Automated optimization + guardrails | Opik |
| Open licensing, no ELv2 | Langfuse (MIT) or Opik (Apache 2.0) |
| Datadog / New Relic integration | Any — they all export OTel |

### Where this pattern goes wrong / 这种模式容易出错的地方

- **No eval strategy.** 没有 evaluation 的 tracing 只是昂贵的 logging。
- **Self-rolled LLM-judge without grounding.** CRITIC pattern（Lesson 05）仍然适用：judge 需要外部工具做 factual verification。
- **Prompt versions not tied to traces.** 生产回归时，你无法 bisect 到造成问题的 prompt。

## Build It / 动手构建

`code/main.py` 实现了一个 stdlib trace collector + LLM-judge evaluator：

- Ingest GenAI-shaped spans。
- 按 session 分组，标记 failed runs（guardrail trips、low-confidence evals）。
- 一个 scripted LLM-judge，按 rubric 给 agent responses 打分。
- dashboard-like summary：failure rate、top failure reasons、eval score distribution。

运行：

```
python3 code/main.py
```

输出：per-session eval scores 和 failure categorization，形式上类似 Langfuse/Phoenix/Opik 会展示的内容。

## Use It / 应用它

- **Langfuse** self-hosted 或 cloud；通过 OTel 或它们的 SDK 接入。
- **Arize Phoenix** self-hosted；auto-instrument OpenInference。
- **Comet Opik** self-hosted 或 cloud；automated optimization loop。
- **Datadog LLM Observability** 适合已经运行 Datadog 的 ops+ML 混合团队。

## Ship It / 交付它

`outputs/skill-obs-platform-wiring.md` 会选择一个平台，并把 traces + evals + prompt versions 接入现有 agent。

## Exercises / 练习

1. 把一周 OTel traces 导出到 Langfuse cloud（free tier）。哪些 sessions 失败了？为什么？
2. 为你的 domain 写一个 LLM-judge rubric（factual correctness、tone、scope adherence）。在 50 条 traces 上测试。
3. 比较 Langfuse prompt versioning 和 Phoenix trace clustering。哪个能更快告诉你哪里坏了？
4. 阅读 Opik 的 guardrail docs。给某次 agent run 接入 PII redaction guardrail。
5. 在你的 corpus 上 benchmark 三个平台。忽略 vendor-published numbers；测你自己的。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Tracing | “Spans collector” | Ingest OTel / SDK spans；按 session 建索引 |
| Prompt management | “Prompt CMS” | 与 traces 绑定的 versioned prompts |
| LLM-as-judge | “Automated eval” | 另一个 LLM 按 rubric 给 agent output 打分 |
| Session replay | “Trace playback” | 回放历史 runs 以便调试 |
| RAG relevancy | “Retrieval quality” | retrieved context 是否匹配 query |
| Trace clustering | “Behavioral grouping” | 聚类相似 runs，用于 drift detection |
| Guardrail enforcement | “Policy at log time” | 对 logged content 做 PII/toxicity/scope checks |

## Further Reading / 延伸阅读

- [Langfuse docs](https://langfuse.com/) — tracing, evals, prompt mgmt
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — auto-instrumentation, drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — optimization + guardrails
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the schema all three consume
