# AI Gateways — LiteLLM, Portkey, Kong AI Gateway, Bifrost / AI Gateways：LiteLLM、Portkey、Kong AI Gateway、Bifrost

> Gateway 位于 apps 与 model providers 之间。核心功能是 provider routing、fallback、retries、rate limiting、secret references、observability、guardrails。2026 年市场分化：**LiteLLM** 是 MIT OSS，100+ providers，OpenAI-compatible，但在约 2000 RPS 附近崩溃（已发布 benchmark 中 8 GB memory、cascading failures）；最适合 Python、<500 RPS、dev/prototyping。**Portkey** 定位 control plane（guardrails、PII redaction、jailbreak detection、audit trails），2026 年 3 月开源为 Apache 2.0，latency overhead 20-40 ms，production tier $49/mo。**Kong AI Gateway** 构建在 Kong Gateway 上；Kong 自己在同样 12 CPUs 上的 benchmark：比 Portkey 快 228%，比 LiteLLM 快 859%；$100/model/month pricing（Plus tier 最多 5 个）；如果你已经用 Kong，它适合 enterprise。**Bifrost**（Maxim AI）提供 configurable backoff 的 automatic retries，并在 OpenAI 429 时 fallback 到 Anthropic。**Cloudflare / Vercel AI Gateways** 是 managed、zero-ops、basic retry。Data residency 决定 self-host；Portkey 和 Kong 位于 OSS + optional managed 的中间地带。

**类型：** 学习
**语言：** Python（stdlib, toy gateway-routing simulator）
**前置知识：** 第 17 阶段 · 01（Managed LLM Platforms）, 第 17 阶段 · 16（Model Routing）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 枚举六个 core gateway features（routing、fallback、retries、rate limits、secrets、observability、guardrails）。
- 把四个 2026 gateways（LiteLLM、Portkey、Kong AI、Bifrost）映射到 scale ceilings 和 use cases。
- 引用 Kong benchmark（228% vs Portkey，859% vs LiteLLM），并解释它为什么对 >500 RPS 重要。
- 根据 data residency 和 ops budget，在 self-hosted 与 managed 间选择。

## The Problem / 问题

你的产品调用 OpenAI、Anthropic 和一个 self-hosted Llama。每个 provider 都有不同 SDK、error model、rate limit 和 auth scheme。你想要 failover（如果 OpenAI 429，就试 Anthropic）、单一 credential store、统一 observability，以及 per-tenant rate limits。

在 app layer 重新发明这些，会把每个 service 绑定到每个 provider。Gateway layer 把它们合并为一个 process 和一个 API（通常 OpenAI-compatible），再向 providers fan out。

## The Concept / 概念

### Six core features / 六个核心功能

1. **Provider routing** — OpenAI、Anthropic、Gemini、self-hosted 等藏在一个 API 后。
2. **Fallback** — 遇到 429、5xx 或 quality failure 时换地方 retry。
3. **Retries** — exponential backoff、bounded attempts。
4. **Rate limits** — per-tenant、per-key、per-model。
5. **Secret references** — runtime 从 vault 拉 credentials（永不放在 app）。
6. **Observability** — OTel + GenAI attributes（Phase 17 · 13）+ cost attribution。
7. **Guardrails** — PII redaction、jailbreak detection、allowed-topics filters。

### LiteLLM — MIT OSS, Python / LiteLLM：MIT OSS，Python

- 100+ providers，OpenAI-compatible，router config，fallback，basic observability。
- Kong benchmark 中约 2000 RPS 崩溃；8 GB memory footprint，持续负载下 cascading failures。
- Best fit：Python app、<500 RPS、dev/staging gateways、experimental routing。
- 成本：OSS $0；cloud free tier 存在。

### Portkey — control plane positioning / Portkey：control plane 定位

- 2026 年 3 月起 Apache 2.0 OSS。Guardrails、PII redaction、jailbreak detection、audit trails。
- 每请求 latency overhead 20-40 ms。
- Production tier $49/mo，含 retention + SLA。
- Best fit：需要 guardrails + observability bundled 的 regulated industries。

### Kong AI Gateway — the scale play / Kong AI Gateway：规模化选择

- 构建在 Kong Gateway 上（成熟 API gateway 产品，lua+OpenResty）。
- Kong 自己在 12-CPU equivalent 上的 benchmark：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Pricing：$100/model/month，Plus tier 最多 5 个。
- Best fit：已经使用 Kong；>1000 RPS；愿意买 license。

### Bifrost (Maxim AI) / Bifrost（Maxim AI）

- Automatic retries with configurable backoff。
- OpenAI 429 时 fallback 到 Anthropic 是 canonical recipe。
- 较新 entrant；commercial。

### Cloudflare AI Gateway / Vercel AI Gateway

- Managed、zero-ops。Basic retry 和 observability。
- Best fit：跑在 Cloudflare/Vercel 上的 Edge-serving JavaScript apps。
- 在 guardrails 和 rate limits 上不如 Kong/Portkey。

### Self-hosted vs managed / Self-hosted 与 managed

Data residency 是强制因素。Healthcare 和 finance 默认 self-host（LiteLLM、Portkey OSS 或 Kong）。Consumer products 默认 managed（Cloudflare AI Gateway）或 middle-tier（Portkey managed）。Hybrid：regulated tenant 用 self-hosted，其他用 managed。

### Latency budget / 延迟预算

- LiteLLM：典型 5-15 ms overhead。
- Portkey：20-40 ms overhead。
- Kong：3-8 ms overhead。
- Cloudflare/Vercel：1-3 ms overhead（edge advantage）。

Gateway latency 会直接加到 TTFT。TTFT P99 < 100 ms SLA 时选 Kong 或 Cloudflare。P99 < 500 ms 时都可用。

### Rate-limit semantics matter / Rate-limit 语义很重要

简单 token-bucket 能撑到中等规模。Multi-tenant 需要 sliding-window + burst allowance + per-tenant tiering。LiteLLM 提供 token-bucket；Kong 提供 sliding-window；Portkey 提供 tiered。

### Gateway + observability + routing compose / Gateway、observability、routing 会组合成一层

Phase 17 · 13（observability）+ 16（model routing）+ 19（gateways）在生产中是同一层。要么选择一个覆盖三者的工具，要么谨慎拼接：多数 2026 deployments 会把 Helicone（observability）或 Portkey（guardrails）与 Kong（scale）组合，让角色分离。

### Numbers you should remember / 你应该记住的数字

- LiteLLM：约 2000 RPS 崩溃，8 GB memory。
- Portkey：20-40 ms overhead；2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong pricing：$100/model/month，Plus tier 最多 5 个。
- Cloudflare/Vercel：edge 上 1-3 ms overhead。

## Build It / 动手构建

用 `code/main.py` 写一个简化 gateway router：注入 429/5xx，配置 retry 与 fallback 顺序，观察 latency overhead 与 availability gain。

## Use It / 应用它

`code/main.py` 模拟 3 个 providers 上的 gateway routing 和 fallback，并注入 429/5xx。它报告 latency、retry rate 和 fallback hit rate。

## Ship It / 交付它

本课产出 `outputs/skill-gateway-picker.md`。给定 scale、ops posture、compliance 和 latency budget，它会选择 gateway。

## Exercises / 练习

1. 运行 `code/main.py`。配置 OpenAI→Anthropic→self-hosted fallback。在 5% provider error rate 下，expected hit rate 是多少？
2. 你的 SLA 是 300 ms baseline 上 TTFT P99 < 200 ms。哪些 gateways 留在预算内？
3. Healthcare 客户要求 self-hosted + PII redaction + audit。选择 Portkey OSS 还是 Kong？
4. 比较 LiteLLM vs Kong：RPS ceiling 到多少时团队应该迁移？
5. 为 multi-tenant SaaS 设计 rate-limit policy：free tier、trial tier、paid tier。用 token-bucket 还是 sliding-window？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Gateway | “API broker” | 位于 apps 和 providers 之间的 process |
| LiteLLM | “the MIT one” | Python OSS，100+ providers，2K RPS 左右崩溃 |
| Portkey | “guardrails gateway” | Control plane + observability，Apache 2.0 |
| Kong AI Gateway | “the scale one” | 构建在 Kong Gateway 上，benchmark 领先 |
| Bifrost | “Maxim's gateway” | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | “edge managed” | Edge-deployed managed gateway，zero-ops |
| PII redaction | “data scrub” | 发送给模型前做 regex + NER mask |
| Jailbreak detection | “prompt injection guard” | user input classifier |
| Audit trail | “regulated log” | 每次 LLM call 的不可变记录 |
| Token-bucket | “simple rate limit” | refill-based rate limiter |
| Sliding-window | “precise rate limit” | time-windowed rate limiter；更公平 |

## Further Reading / 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
