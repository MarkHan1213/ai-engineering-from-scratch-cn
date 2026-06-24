# LLM Observability Stack Selection / LLM 可观测性栈选型

> 2026 年 observability 市场分成两类。Development platforms（LangSmith、Langfuse、Comet Opik）把 monitoring 与 evals、prompt management、session replays 打包在一起。Gateway/instrumentation tools（Helicone、SigNoz、OpenLLMetry、Phoenix）聚焦 telemetry。Langfuse 是 MIT-licensed core，OSS 平衡很好（cloud 免费 50K events/month）。Phoenix 是 OpenTelemetry-native，Elastic License 2.0：非常适合 drift/RAG visualization，但不是持久 production backend。Arize AX 使用 zero-copy Iceberg/Parquet integration，声称比 monolithic observability 便宜 100x。LangSmith 对 LangChain/LangGraph 领先，$39/user/mo，只在 Enterprise self-host。Helicone 是 proxy-based，15-30 分钟接入，免费 100K req/mo，但 agent traces 深度较弱。常见生产模式：Gateway（Helicone/Portkey）+ eval platform（Phoenix/TruLens），用 OpenTelemetry 黏合。

**类型：** 学习
**语言：** Python（stdlib, toy trace-sampling simulator）
**前置知识：** 第 17 阶段 · 08（Inference Metrics）, 第 14 阶段（Agent Engineering）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 development platforms（bundled: evals + prompts + sessions）和 gateway/telemetry tools（traces + metrics only）。
- 把六个主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到 license、pricing 和 sweet-spot use cases。
- 解释 OpenTelemetry-glue pattern，说明如何把 gateway tool 与独立 eval platform 组合。
- 说出 2026 年的成本差异点（Arize AX 的 zero-copy approach vs monolithic ingest）以及约 100x multiplier。

## The Problem / 问题

你上线了一个 LLM feature。它能跑。你却看不到 prompt failures、tool loops、latency regressions、cost spikes 或 prompt-cache hit rate。你搜索 “LLM observability”，得到八个工具，都声称以三种价位解决同一个问题。

它们并不解决同一个问题。LangSmith 回答“为什么这个 LangGraph run 失败？”Phoenix 回答“我的 RAG pipeline 是否 drift？”Helicone 回答“哪个 app 在烧 tokens？”Langfuse 回答“我能否 self-host 整套东西？”工具不同，受众不同。

选型涉及四个轴：stack（LangChain？raw SDK？multi-vendor？）、license tolerance（只要 MIT？Elastic 可接受？commercial 可接受？）、budget（free tier？$100/mo？$1000/mo？）、self-host（必须？nice-to-have？绝不？）。

## The Concept / 概念

### Two categories / 两大类

**Development platforms** 把 observability 与 evals、prompt management、dataset versioning、session replay 打包。你运行 experiments，查看哪个 prompt 有效，把新 prompt 在 dataset regression 上与旧赢家比较。LangSmith、Langfuse、Comet Opik 属于这类。

**Gateway/telemetry tools** instrument inference calls：prompt、response、tokens、latency、model、cost。Helicone、SigNoz、OpenLLMetry、Phoenix。更小而专注。可通过 OpenTelemetry 与独立 eval tool 组合。

### Langfuse — OSS balance / Langfuse：OSS 平衡

- Core Apache / MIT licensed；可通过 Docker self-host。
- Cloud free tier：50K events/month。Paid：team $29/mo。
- Evals、prompt management、traces、datasets。四类 dev-platform features 覆盖合理。
- Sweet spot：你想要 LangSmith 级功能，但必须 self-host 或保持 OSS license。

### Phoenix (Arize) — telemetry-first, OpenTelemetry-native / Phoenix（Arize）：telemetry-first、OpenTelemetry-native

- Elastic License 2.0；self-host 很简单。
- RAG 和 drift visualization 很强。Embedding-space scatter plots 是 first-class。
- 不是为持久 production backend 设计，主要是 development-time observability。
- Sweet spot：RAG pipeline development、drift debugging，生产上搭配独立 gateway。

### Arize AX — the scale play / Arize AX：规模化打法

- Commercial。通过 Iceberg/Parquet 做 zero-copy data lake integration。
- 声称在规模上比 monolithic observability（Datadog-class）便宜约 100x。数学是：traces 存在你自己的 S3 Parquet，Arize 直接读取。
- Sweet spot：>10M traces/day，已有 data lake，想要 LLM-specific dashboards 但不想付 Datadog pricing。

### LangSmith — LangChain/LangGraph first / LangSmith：LangChain/LangGraph 优先

- Commercial，$39/user/month。Self-host 仅 Enterprise。
- 对 LangChain 和 LangGraph stack 最强。如果你不用两者，吸引力会下降。
- Sweet spot：团队已承诺 LangChain，并愿意付费。

### Helicone — proxy-based minimum viable / Helicone：proxy-based 最小可行

- 通过把 `OPENAI_API_BASE` 切到 Helicone proxy，15-30 分钟接入。
- MIT licensed；免费 100K req/mo，paid $20/mo+。
- 包含 failover、caching、rate limits，也像 gateway。
- agent / multi-step traces 深度较弱。
- Sweet spot：quick start、single-stack app，需要 gateway + observability 一体。

### Opik (Comet) — OSS dev platform / Opik（Comet）：OSS dev platform

- Apache 2.0，完全 OSS。
- 功能集接近 Langfuse，带 Comet 传承。
- Sweet spot：ML teams 已经使用 Comet，希望 LLM observability 在同一视图里。

### SigNoz — OpenTelemetry-first full APM / SigNoz：OpenTelemetry-first full APM

- Apache 2.0。处理 general APM，并通过 OpenTelemetry 处理 LLM。
- Sweet spot：跨 services 和 LLM calls 的统一 observability。

### The glue: OpenTelemetry + GenAI semantic conventions / 胶水：OpenTelemetry + GenAI semantic conventions

OpenTelemetry 在 2025 年末发布 GenAI semantic conventions（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。能消费 OTel 的工具可以互操作。正在形成的生产模式：

1. 每个 LLM call 都用 GenAI conventions 发 OTel。
2. 日常通过 gateway（Helicone / Portkey）。
3. 双写到 eval platform（Phoenix / Langfuse）做 regressions。
4. 存档到 data lake（Iceberg），供 Arize AX 或 DuckDB 长期分析。

### The trap: instrumenting at the wrong layer / 陷阱：在错误层埋点

在 agent framework 内部埋点（例如添加 LangSmith traces）会把你绑定到 framework。在 HTTP/OpenAI-SDK 层埋点（通过 OpenLLMetry 或 gateway）才可移植。

### Sampling — you can't keep everything / Sampling：你不能保留所有东西

超过 1M requests/day 后，full-trace retention 的成本可能高于 LLM calls。按规则采样：100% errors、100% high-cost、5% success。Aggregates 永远保留；raw 只保留长尾。

### Numbers you should remember / 你应该记住的数字

- Langfuse free cloud：50K events/month。
- LangSmith：$39/user/month。
- Helicone free：100K req/month。
- Arize AX claim：规模上比 monolithic 便宜约 100x。
- OpenTelemetry GenAI conventions：2025 shipping，2026 widely adopted。

## Build It / 动手构建

用 `code/main.py` 生成一天 trace volume，并对比 full retention、rule-based sampling 和 error-only retention，让工具选型落到存储与可诊断性权衡上。

## Use It / 应用它

`code/main.py` 会模拟一天 1M traces 在不同 retention strategies（100% ingest、sampling、sampling + errors）下的情况。它报告 storage cost，以及每种策略会丢什么。

## Ship It / 交付它

本课产出 `outputs/skill-observability-stack.md`。给定 stack、scale、budget 和 license posture，它会选择工具组合。

## Exercises / 练习

1. 你的团队使用 LangChain，希望 OSS self-hosted observability。选择 Langfuse 或 Opik，并说明理由。
2. 每天 5M traces，Datadog 报价 $150K/month。计算 Arize AX 的 break-even。
3. 设计一套 OpenTelemetry GenAI attributes，你的组织应要求每个 LLM call 都带上。
4. 论证 Phoenix alone 是否足够生产使用。什么时候不够？
5. Helicone proxy overhead 是 20ms。P99 TTFT 300 ms 时可接受吗？如果 SLA 是 100 ms 呢？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| OpenLLMetry | “OTel for LLMs” | 面向 LLM 的 open-source OpenTelemetry instrumentation |
| GenAI conventions | “OTel attributes” | LLM calls 的标准 OTel attribute names |
| LangSmith | “LangChain observability” | 与 LangChain ecosystem 捆绑的 commercial platform |
| Langfuse | “OSS LangSmith” | MIT OSS，功能集相近 |
| Phoenix | “Arize dev tool” | OpenTelemetry-native dev/eval platform |
| Arize AX | “scale observability” | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | “proxy observability” | 收集 LLM telemetry + gateway features 的 HTTP proxy |
| Opik | “Comet LLM” | Comet 的 Apache 2.0 OSS dev platform |
| Session replay | “trace rerun” | 带 tool calls 的完整 agent session replay |
| Eval | “offline test” | 在 labeled dataset 上运行 candidate model/prompt |

## Further Reading / 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
