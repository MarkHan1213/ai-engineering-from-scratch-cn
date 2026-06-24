# LLM Routing Layer — LiteLLM, OpenRouter, Portkey / LLM 路由层：LiteLLM、OpenRouter、Portkey

> provider lock-in 很贵。不同 tool-calling workload 适合不同模型。Routing gateways 提供一个 API surface、retries、failover、cost tracking 和 guardrails。2026 年三种 archetype 最主流：LiteLLM（open-source self-hosted）、OpenRouter（managed SaaS）、Portkey（production-grade，2026 年 3 月开源）。本课会给出 decision criteria，并走过一个 stdlib routing gateway。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, routing + failover + cost tracker)
**Prerequisites / 前置知识：** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 区分 self-hosted、managed 和 production-grade routing options。
- 实现 fallback chain，在 provider failures 时按定义好的 priority order retry。
- 跨 providers 追踪 per-request cost 和 token usage。
- 针对给定 production constraint，在 LiteLLM、OpenRouter、Portkey 之间做选择。

## The Problem / 问题

provider routing 重要的场景：

1. **Cost.** Claude Sonnet 的成本是 Haiku 的 3 倍。triage task 用 Haiku 足够；synthesis task 值得用 Sonnet。按 request route。

2. **Failover.** OpenAI 出现一小时故障。每个 request 都失败。你希望自动 fallback 到 Anthropic，而不 redeploy。

3. **Latency.** live chat UI 需要快的 time-to-first-token。batch summarizer 不需要。按 latency SLA route。

4. **Compliance.** EU users 必须留在 EU regions。按 region route。

5. **Experimentation.** 在同一 workload 上 A/B 两个 models。按 test bucket route。

为每个 integration 手写这些逻辑很重复。routing gateway 提供一个 OpenAI-compatible API，并处理其余部分。

## The Concept / 概念

### OpenAI-compatible proxy shape / OpenAI-compatible proxy 形状

大家都说 OpenAI-shape。routing gateway 暴露 `/v1/chat/completions`，接受 OpenAI schema，内部代理到 Anthropic / Gemini / Cohere / Ollama / 任何后端。client 不需要关心。

### Model aliases / 模型别名

代码不用写 `claude-3-5-sonnet-20251022`，而是写 `our_smart_model`。gateway 把 aliases 映射到真实 models。Anthropic 发布 Claude 4 时，你在 server-side 改 alias；代码完全不用动。

### Fallback chains / Fallback 链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

gateways 在 config 中定义它。retries 会计入 budget，避免 fallback cascades 把成本炸开。

### Semantic caching / 语义缓存

完全相同或近似相同的 prompts 命中 cache，不再打 provider。重复 agent loops 上的 savings 可达 30% 到 60%。key 基于 embedding；near-identical prompts 共用 cache slot。

### Guardrails / Guardrails

gateway-level：

- **PII redaction.** prompt 发送前做 regex 或 ML-based pass。
- **Policy violations.** 拒绝包含 prohibited content 的 prompts。
- **Output filters.** 清理 completion 中的泄漏。

Portkey 和 Kong 都提供 opinionated guardrails。LiteLLM 把它们作为 optional。

### Per-key rate limits / Per-key rate limits

一个 API key = 一个 team。per-key budgets 防止一个 team 消耗共享 quota。大多数 gateways 支持它。

### Self-hosted vs managed trade-offs / Self-hosted vs managed 取舍

| Factor | LiteLLM (self-hosted) | OpenRouter (managed) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Code | Open source, Python | Managed SaaS | Open source (Mar 2026) + managed |
| Setup | Deploy a proxy | Sign up | Either |
| Providers | 100+ | 300+ | 100+ |
| Billing | Your own keys | OpenRouter credits | Your own keys |
| Observability | OpenTelemetry | Dashboard | Full OTel + PII redaction |
| Best for | Teams that want full control | Rapid prototyping | Production with compliance |

当你有 SRE team 且需要 data sovereignty，LiteLLM 胜出。想要单一订阅且没有 infra，OpenRouter 胜出。需要开箱即用 guardrails 和 compliance，Portkey 胜出。

### Cost tracking / 成本追踪

每个 request 携带 `provider`、`model`、`input_tokens`、`output_tokens`。乘以 per-model per-token prices（来自 gateway 维护的 pricing sheet）。再按 per-user / per-team / per-project 聚合。

### MCP plus routing / MCP 加 routing

gateway 可以同时 route LLM calls 和 MCP sampling requests。sampling request 的 modelPreferences 偏好某个模型时，gateway 会翻译到正确 backend。这就是 Phase 13 · 17（MCP gateway）和本课 routing gateway 有时合并成一个服务的地方。

### Routing strategies / 路由策略

- **Static priority.** 列表第一个；error 时 fallback。
- **Load balancing.** Round-robin 或 weighted。
- **Cost-aware.** 选择满足 latency / quality 的最便宜模型。
- **Latency-aware.** 选择过去 N 分钟最快的模型。
- **Task-aware.** prompt classifier 把 coding 路由到一个模型，把 summarization 路由到另一个。

## Build It / 动手构建

本课会实现一个最小 routing gateway：接受 OpenAI-shaped request，把 model alias 展开成 priority-ordered backend list，按 5xx fallback，记录 token cost，并在发送前做 PII redaction。

## Use It / 应用它

`code/main.py` 用约 150 行实现 routing gateway：接受 OpenAI-shaped requests，翻译到 per-provider stubs，运行 priority fallback chain，追踪 per-request cost，并对 inputs 应用 PII redaction pass。用三个场景运行：normal request、primary-provider outage triggering fallback、PII leakage caught by redaction。

重点看：

- `ROUTES` dict：alias -> priority-ordered list of concrete providers。
- Fallback loop 在 5xx 上 retry。
- Cost tracker 把 token usage 乘以 per-model rates。
- PII redactor 在 forwarding 前清理 SSN-shaped patterns。

## Ship It / 交付它

本课产出 `outputs/skill-routing-config-designer.md`。给定 workload profile（latency、cost、compliance），这个 skill 会选择 LiteLLM / OpenRouter / Portkey 并产出 routing config。

## Exercises / 练习

1. 运行 `code/main.py`。触发 outage scenario；确认 fallback 落到第二个 provider，并且 cost attribution 正确。

2. 添加 semantic caching：prompt 的 SHA256 作为 lookup key；cache hit 立即返回。测量 repeated call 的 cost savings。

3. 添加 prompt classifier，把 "code ..." prompts 路由到偏 intelligence 的 alias，把 "summarize ..." prompts 路由到偏 speed 的 alias。

4. 设计 per-team budgets：每个 team 有 monthly spend cap；cap 命中后 gateway 拒绝 requests。选择 enforcement granularity（per-request 或 windowed）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey docs。说出每家提供、另外两家没有的一个 feature。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Routing gateway | “LLM proxy” | 位于多个 providers 前的一层统一 API surface |
| OpenAI-compatible | “Speaks the OpenAI schema” | 接受 `/v1/chat/completions` shape，并翻译到任意 backend |
| Model alias | “our_smart_model” | 代码中的名称，由 gateway 映射到具体模型 |
| Fallback chain | “Retry list” | failure 时按顺序尝试的 providers 列表 |
| Semantic caching | “Prompt-embedding cache” | key 是 prompt embedding；near-duplicates 共享 cache hit |
| Guardrails | “Input/output filters” | redact PII，reject policy violations |
| Per-key rate limit | “Team budget” | 绑定到 API key 的 quota |
| Cost tracking | “Per-request spend” | 聚合 token usage x model price |
| LiteLLM | “The open proxy” | 可 self-host 的 OSS routing gateway |
| OpenRouter | “The managed SaaS” | 基于 credits billing 的 hosted gateway |
| Portkey | “The production option” | 内建 guardrails 的 open-source + managed gateway |

## Further Reading / 延伸阅读

- [LiteLLM — docs](https://docs.litellm.ai/) — self-hosted routing gateway
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — managed routing SaaS
- [Portkey — docs](https://portkey.ai/docs) — 带 guardrails 的 production routing
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — decision guide
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — vendor survey
