# Capstone 11 — LLM Observability & Eval Dashboard / LLM 可观测性与评测仪表盘

> Langfuse 转向 open-core。Arize Phoenix 发布了 2026 GenAI semconv mappings。Helicone 和 Braintrust 都继续强化 per-user cost attribution。Traceloop 的 OpenLLMetry 成为事实上的 SDK instrumentation。生产形态是 ClickHouse 存 traces、Postgres 存 metadata、Next.js 做 UI，再加一组 eval jobs（DeepEval、RAGAS、LLM-judge）跑在 sampled traces 上。构建一个 self-hosted dashboard，从至少四类 SDK 摄取数据，并演示在五分钟内捕获注入的 regression。

**类型：** 综合项目
**语言：** TypeScript（UI）, Python / TypeScript（ingest + evals）, SQL（ClickHouse）
**前置知识：** 第 11 阶段（LLM engineering）, 第 13 阶段（tools）, 第 17 阶段（infrastructure）, 第 18 阶段（safety）
**Phases exercised:** P11 · P13 · P17 · P18
**时间：** 25 小时

## Learning Objectives / 学习目标

- 基于 OpenTelemetry GenAI semantic conventions 设计统一 ingest schema
- 用 ClickHouse、Postgres 和 S3 分层存储 spans、metadata 与 raw events
- 构建 DeepEval、RAGAS、custom LLM-judge 等 eval jobs，并把 eval spans 关联回原 trace
- 实现 drift detection、Prometheus / Alertmanager 告警链路与 Next.js dashboard
- 用注入 regression 验证 dashboard 能在五分钟内发现生产问题

## Problem / 问题

到 2026 年，每个跑生产流量的 AI 团队都会在模型旁边保留一层 observability plane。Cost attribution、hallucination detection、drift monitoring、jailbreak signal、SLO dashboards、PII leak alerts。开源参考系统 Langfuse、Phoenix、OpenLLMetry 都收敛到 OpenTelemetry GenAI semantic conventions 作为 ingest schema。你现在可以用同一个 SDK instrument OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM，并发送兼容 spans。

你要构建一个 self-hosted dashboard，从至少四类 SDK families 摄取数据，在 sampled traces 上运行一组 eval jobs，检测 drift，并触发告警。测量标准：给定一个故意注入的 regression（某个 prompt 开始泄漏 PII），dashboard 能在五分钟内捕获并发出 alert。

## Concept / 概念

Ingest 使用 OTLP HTTP。SDK 产生 GenAI-semconv spans：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Spans 写入 ClickHouse 做列式分析；metadata（users、sessions、apps）写入 Postgres。

Evals 作为 batch jobs 跑在 sampled traces 上。DeepEval 评分 faithfulness、toxicity 和 answer relevance。当 trace 携带 retrieval context 时，RAGAS 评分 retrieval metrics。Custom LLM-judges 跑 domain-specific checks（PII leak、off-policy response）。Eval runs 把结果写回同一个 ClickHouse，作为链接到 parent trace 的 eval spans。

Drift detection 监控 embedding-space distributions 的时间变化（prompt embeddings 上的 PSI 或 KL divergence）以及 eval-score trends。Alerts 进入 Prometheus Alertmanager，再到 Slack / PagerDuty。UI 使用 Next.js 15 和 Recharts。

## Architecture / 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Stack / 技术栈

- Ingest: OpenTelemetry SDKs + GenAI semantic conventions；OTLP HTTP transport
- Collector: OpenTelemetry Collector，带 tail-sampling processor（用于 cost control）
- Storage: ClickHouse 存 spans，Postgres 存 metadata，S3 归档 raw events
- Evals: DeepEval、RAGAS 0.2、Arize Phoenix evaluator pack、custom LLM-judge
- Drift: pooled prompt embeddings（sentence-transformers）上每周计算 PSI / KL
- Alerting: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- SDKs supported out of the box: OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## Build It / 动手构建

1. **Collector config.** OpenTelemetry Collector 配置 OTLP HTTP receiver、tail-sampler（100% 保留 errored traces、10% 保留 successes），以及导出到 ClickHouse 和 S3 的 exporters。

2. **ClickHouse schema.** 表 `spans` 的列镜像 GenAI semconv：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，再加 JSON bag 存 long payloads。按 user_id 和 app_id 添加 secondary indexes。

3. **SDK coverage test.** 用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）写一个小 client app，并使用 OpenLLMetry auto-instrument。验证每个都能产出 canonical GenAI spans 并落到 ClickHouse。

4. **Eval jobs.** 定时任务读取 last-15-min sampled traces，运行 DeepEval faithfulness、toxicity 和 answer relevance。输出作为 eval spans 链接到 parent trace。

5. **Custom LLM-judge.** PII-leak judge：给定 response，调用 guard LLM 评分 PII leak 可能性。高分 responses 进入 triage queue。

6. **Drift detection.** 每周 job 计算本周 pooled prompt embeddings 与 trailing 4-week baseline 之间的 PSI。高于阈值则告警。

7. **Dashboard.** Next.js 15 页面：overview（spans/sec、cost/user、p95 latency）、traces（search + waterfall）、evals（faithfulness trend、toxicity）、drift（PSI over time）、alerts。

8. **Alerting chain.** Prometheus exporter 读取 eval score aggregates 和 latency percentiles；Alertmanager 将 warnings 路由到 Slack，critical breaches 路由到 PagerDuty。

9. **Regression probe.** 注入 bug：被评估的 chatbot 1% 概率泄漏 fake SSNs。测量 MTTR：从 bug 部署到 Slack alert。

## Use It / 应用它

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Ship It / 交付它

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM application，dashboard 能摄取 traces、运行 evals、对 drift 告警，并在 Next.js 中展示 cost/user breakdown。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Trace-schema coverage | 产出 canonical GenAI spans 的 SDK families 数量（target: 6+） |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | injected regression 的 MTTR（目标低于 5 分钟） |
| 20 | Cost / scale | 1k spans/sec sustained ingest 且无 backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain 端到端演练 |
| **100** | | |

## Exercises / 练习

1. 为 Haystack framework 添加 custom instrumentation。验证 canonical spans 带着忠实的 `gen_ai.*` attributes 落到 ClickHouse。

2. 在同一 traces 上把 DeepEval 换成 Phoenix evaluators。测量两个 eval engines 的 score drift。

3. 强化 drift detector：按 app-id 而不是全局计算 PSI。展示 per-app drift trails。

4. 添加 “user impact” 页面：cost-per-user 和 failure-rate-per-user，并带 sparklines。

5. 构建 tail-sampling policy：保留 100% toxicity > 0.5 的 traces，并对其余 traces 做 10% stratified sample。测量引入的 sampling bias。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | “OTel LLM attributes” | 2025 OpenTelemetry spec，用于 LLM span attributes（system、model、tokens） |
| Tail sampling | “Post-trace sample” | Collector 在 trace 完成后决定保留或丢弃（可以观察 errors） |
| PSI | “Population stability index” | 比较两个 distributions 的 drift metric；> 0.2 通常表示有意义漂移 |
| LLM-judge | “Eval as model” | 一个 LLM 按 rubric 给另一个 LLM 输出打分（faithfulness、toxicity、PII） |
| Tail-sampling policy | “Keep-rule” | 决定哪些 traces 持久化、哪些丢弃的规则；errored + sample-rate |
| Eval span | “Linked eval trace” | 携带 eval score 并链接到原始 LLM call span 的 child span |
| Cost per user | “Unit economics” | 在一个窗口内归因到 user_id 的美元成本；关键 product metric |

## Further Reading / 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — reference open-core observability platform
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 强 drift support 的 alternate reference
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — auto-instrumentation SDK family
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — ingest schema
- [Helicone](https://www.helicone.ai) — alternate hosted observability
- [Braintrust](https://www.braintrust.dev) — alternate eval-first platform
- [ClickHouse documentation](https://clickhouse.com/docs) — columnar span store
- [DeepEval](https://github.com/confident-ai/deepeval) — evaluator library
