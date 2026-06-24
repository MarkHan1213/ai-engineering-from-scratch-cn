# FinOps for LLMs — Unit Economics and Multi-Tenant Attribution / LLM FinOps：单位经济与多租户归因

> 传统 FinOps 在 LLM spend 上会失效。成本来自 token 计费请求，而不是资源运行时长。传统标签模型不匹配：API call 是一条计费事件，不是可打标签的云资源。工程决策（prompt design、context window、output length）就是财务决策。2026 playbook 要从第一天开始 instrument 三个归因维度：per-user（`user_id`）用于 seat pricing 和 expansion，per-task（`task_id` + `route`）用于 product surface cost 和 prioritization，per-tenant（`tenant_id`）用于 unit economics 和 renewal。四个 token layers：prompt、tool、memory、response；放进一个 bucket 会隐藏 spend。Multi-tenant products 的 enforcement ladder：per tenant rate limits（expected peak 的 2-3x，清晰 429 + retry-after）；daily spend cap（contracted ceiling 的 1.5-3x，触发 rate tightening + alert）；spend z-score > 4 时 kill switches（auto-pause + page on-call）。Attribution patterns：tag-and-aggregate、telemetry-joiner（trace-ID → billing，准确度最高）、sampling-and-extrapolation、model-based allocation、event-sourced、real-time streaming。Unit metric：cost per resolved query、cost per generated artifact，不是 $/M tokens。Retroactive tagging 总会漏；在 request creation 时 instrument。

**类型：** 学习
**语言：** Python（stdlib, toy cost-attribution simulator with kill switch）
**前置知识：** 第 17 阶段 · 13（Observability）, 第 17 阶段 · 14（Caching）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释为什么传统 FinOps（tags + tiers）在 LLM spend 上失效，并说出三个新 attribution dimensions。
- 枚举四个 token layers（prompt、tool、memory、response），以及 single-bucket billing 为什么隐藏成本。
- 为 multi-tenant product 设计 enforcement ladder（rate → spend cap → kill switch）。
- 选择 unit metric（cost per resolved query / artifact），而不是 $/M tokens。

## The Problem / 问题

账单显示 $40,000。你不知道：
- 哪个 tenant 花了这些钱。
- 哪个 product feature 驱动了它。
- 是否有单个 user 滥用。
- 元凶是 prompt bloat、tool calls，还是 memory amplification。

Provider-side 的 tag-and-aggregate 适合 cloud resources（EC2、S3），因为 tags 会传播到 line items。LLM API calls 不会自动打 tag；你必须在 call site stamp user/task/tenant，并一路携带。Retroactive attribution 总会漏掉边缘情况。

## The Concept / 概念

### Three attribution dimensions / 三个归因维度

**Per-user**（`user_id`）：哪个用户造成了多少成本。驱动 seat pricing、expansion conversations，并识别 power users。

**Per-task**（`task_id` + `route`）：哪个 product surface 花多少钱。驱动 feature prioritization 和 kill-expensive-features decisions。

**Per-tenant**（`tenant_id`）：哪个客户盈利。驱动 unit economics、renewal pricing、tier thresholds。

第一天就在 call site instrument 三者。Retroactive 永远更差。

### Four token layers / 四个 token layers

| Layer | Example | Typical % of total |
|-------|---------|---------------------|
| Prompt | system + user input | 40-60% |
| Tool | tool-call results fed back | 20-40% (agent workloads) |
| Memory | prior conversation / retrieved docs | 10-30% |
| Response | model output | 10-30% |

把四者混进一个 bucket，会让优化变盲。在 attribution schema 中拆开。

### Enforcement ladder / Enforcement ladder

1. **Rate limit** per tenant。2-3x expected peak。返回带 `Retry-After` 的 429。Tenant 感到阻力；没有惊喜账单。

2. **Daily spend cap** per tenant。1.5-3x contracted ceiling。触发：收紧 rate limit + alert customer-success。

3. **Kill switch** on spend z-score > 4 relative to tenant baseline。Auto-pause tenant；page on-call；escalate to ops + CS。

### Attribution patterns / 归因模式

- **Tag-and-aggregate**：stamp metadata headers，后续聚合。简单，粗略。
- **Telemetry joiner**：通过 trace IDs 把 traces join 到 billing。准确度最高。成熟团队采用。
- **Sampling + extrapolation**：sample 5-10%，乘回来。粗略 spend 成本低；漏 tails。
- **Model-based allocation**：用 regression 推断 cost driver。适合没有 tags 的 legacy data。
- **Event-sourced**：cost 作为 stream 中的 events（Kafka / Kinesis）。Real-time。
- **Real-time streaming**：dashboard sub-second 更新。

### Cost per X is the unit metric / Cost per X 才是单位指标

$/M tokens 是 vendor 语言。产品指标是：

- Cost per resolved support ticket。
- Cost per generated article。
- Cost per successful agent task。
- Cost per user-session-minute。

把 cost 绑定到 product outcome。否则优化没有锚点。

### Cost attribution trace shape / 成本归因 trace 形状

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每次 call 都 emit。存进 data lake。按 dimension 聚合。Phase 17 · 13 observability stack 就承载这些。

### The compounded-savings stack / 复合节省栈

Stack：cache + batch + route + gateway。四者齐全时：
- Cache L2（Phase 17 · 14）：input 约便宜 10x。
- Batch（Phase 17 · 15）：50% off。
- Route to cheap model（Phase 17 · 16）：60% cost reduction。
- Gateway efficiency（Phase 17 · 19）：redundancy + retries。

Best-case stacked：约为 naive baseline 的 5-10%。多数团队启用 2-3 个 levers；很少全部叠加。

### Numbers you should remember / 你应该记住的数字

- Attribution dimensions：per-user、per-task、per-tenant。
- Four token layers：prompt、tool、memory、response。
- Kill switch：spend z-score > 4。
- Unit metric：cost per resolved query，不是 $/M tokens。
- Stacked optimizations：可能降到 baseline 的约 5-10%。

## Build It / 动手构建

用 `code/main.py` 生成多租户调用事件，给每次 call 标注 `user_id`、`tenant_id`、`task_id` 和 token layers，再触发 spend cap 与 kill switch。

## Use It / 应用它

`code/main.py` 模拟一个带三层 enforcement ladder 的 multi-tenant LLM service。它注入 abusive tenant，并展示 kill switch 触发。

## Ship It / 交付它

本课产出 `outputs/skill-finops-plan.md`。给定 product 和 scale，它会设计 attribution schema 和 enforcement ladder。

## Exercises / 练习

1. 运行 `code/main.py`。kill switch 在什么 z-score 触发？你如何选择 threshold？
2. 设计一个 per-tenant、per-task cost dashboard。先做哪 5 个 views？
3. 最大 tenant 的 unit economics 为负。按 customer impact 从低到高提出三个干预。
4. 为一个 support product 计算 cost per resolved ticket：3M tokens/ticket，约 800 tickets/day，GPT-5 cached rate。
5. 论证 retroactive tagging 是否可能可行。什么时候可以接受？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Per-user attribution | “user-level cost” | 每次 call 都 stamp `user_id` |
| Per-task attribution | “feature cost” | `task_id` + `route` 标识 product surface |
| Per-tenant attribution | “customer cost” | `tenant_id`；驱动 unit economics |
| Four token layers | “cost layers” | prompt + tool + memory + response |
| Rate limit | “429 guard” | Gateway 执行的 per-tenant ceiling |
| Daily spend cap | “daily ceiling” | Tenant-scoped budget with alert |
| Kill switch | “auto-pause” | Spend z-score > 4 时 auto-suspension |
| Cost per resolved | “product unit metric” | Cost 绑定 product outcome，而非 tokens |
| Telemetry joiner | “trace-to-billing” | 最高准确度 attribution pattern |
| Stacked optimization | “cache+batch+route+gateway” | 复合节省到 baseline 的约 5-10% |

## Further Reading / 延伸阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
