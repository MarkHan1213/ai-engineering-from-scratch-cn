# Prompt Caching and Semantic Caching Economics / Prompt Caching 与 Semantic Caching 经济学

> **Pricing snapshot dated 2026-04.** 下面的数字反映本课发布时抓取的 vendor rate cards；在下游引用前，请重新核对链接文档。

> Caching 发生在两层。L2（provider-level）prompt/prefix caching 为重复 prefix 复用 attention KV：Anthropic prompt-caching 文档宣称长 prompts 上最高 90% cost reduction 和 85% latency reduction；Claude 3.5 Sonnet 的 cache reads 是 $0.30/M，而 fresh 是 $3.00/M，TTL 5 分钟；1-hour TTL 选项有 2x write premium（docs.anthropic.com，2026-04）。OpenAI prompt caching 会自动应用于 ≥1024 tokens 的 prompts，cached input 价格约为 fresh 的 90% discount（platform.openai.com，2026-04）；具体 per-model cached rate 取决于 live rate card。L1（app-level）semantic caching 在 embedding similarity 命中时完全跳过 LLM。Vendor 所说 “95% accuracy” 指的是 match correctness，不是 hit rate；已报告生产 hit rates 从 open-ended chat 的 10% 到 structured FAQ 的 70% 不等；provider 都没有发布官方 baseline，因此把它们当成 community telemetry，而不是 guarantee。生产陷阱：parallelization 会杀死 caching（第一次 cache write 完成前并行发出的 N 个请求会让花费膨胀数倍）；dynamic content 放进 prefix 会完全阻止 cache hits。ProjectDiscovery 报告把 dynamic text 移出 cacheable prefix 后，hit rate 从 7% 提升到 74%（2025-11）。

**类型：** 学习
**语言：** Python（stdlib, toy two-layer cache simulator）
**前置知识：** 第 17 阶段 · 04（vLLM Serving Internals）, 第 17 阶段 · 06（SGLang RadixAttention）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 L2 prompt/prefix caching（provider 侧 KV reuse）和 L1 semantic caching（相似 prompt 命中时绕过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记，以及两个 TTL 选项（5-min vs 1-hour）和价格倍数。
- 给定 hit rate、prompt/response mix 和 token prices，计算 expected monthly savings。
- 说出让账单膨胀 5-10x 的 parallelization anti-pattern，以及让 hit rate 坍塌的 dynamic-content anti-pattern。

## The Problem / 问题

你给 RAG service 加了 prompt caching。账单没变。你测 hit rate：7%。你的 prompts 看起来静态，但其实不是：system prompt 包含精确到分钟的当前日期、request ID，以及为多样性随机重排的 examples。每个请求都写一个新 cache entry，读零次。

另外，你的 agent 对每个用户问题并行运行十个 tool calls。十个请求都在第一次 cache write 完成前到达 provider。十次写入，零次读取。你的账单是“启用 caching”预期成本的 5-10x。

Caching 是一种协议，不是一个 flag。两层，两种 failure modes。

## The Concept / 概念

### L2 — provider prompt/prefix caching / L2：provider prompt/prefix caching

Provider 存储一个 cacheable prefix 的 attention KV，并在下次匹配 prefix 的请求中复用。你一次性支付 write cost，之后 reads 几乎免费。

**Anthropic (Claude 3.5 / 3.7 / 4 series)**：request 中显式 `cache_control` marker。你标记哪些 blocks 可缓存。TTL：5-minute（write costs 1.25x base）或 1-hour（write costs 2x base）。Cache reads：Claude 3.5 Sonnet 上 $0.30/M，而 fresh 是 $3.00/M，便宜 10x（docs.anthropic.com，截至 2026-04）。不同模型（Opus/Haiku）价格不同；始终交叉检查 live pricing page。

**OpenAI**：对 ≥1024 tokens prompts 自动 caching（platform.openai.com，2026-04）。无显式 flag。当前 gpt-4o/gpt-5 rate cards 上，cached input 约比 fresh 便宜 10x。Docs 和 release notes 都不发布官方 hit-rate baseline；community reports 在精心 prompt design 下集中在 30–60%。监控 `usage.cached_tokens` 测自己的。

**Google (Gemini)**：通过显式 API 做 context caching；1M-token context 让 caching 更有价值。

**Self-hosted (vLLM, SGLang)**：Phase 17 · 06 讲 RadixAttention，同样模式，只是用自己的 compute。

### L1 — app-level semantic caching / L1：应用级 semantic caching

调用 LLM 前，先 hash prompt、embed 它，并查找相似 cached request（cosine similarity 高于阈值，通常 0.95+）。命中时返回 cached response。未命中时调用 LLM 并缓存结果。

Open-source：Redis Vector Similarity、GPTCache、Qdrant。Commercial：Portkey Cache、Helicone Cache。

Vendor accuracy claims 指返回 cached response 在语义上合适的比例，不是命中比例。生产 hit rates：

- Open-ended chat：10-15%。
- Structured FAQ / support：40-70%。
- Code questions：20-30%（小变化会杀命中）。
- Voice agents repeating prompts：50-80%（voice normalization fixed set）。

### The parallelization anti-pattern / parallelization 反模式

你的 agent 并行发起 10 个 tool calls。它们都有同一个 4K-token system prompt。Anthropic cache writes 按 request 发生；第一个 cache-write 大约在 provider 看到 prompt 后 300 ms 完成。请求 2-10 在同一毫秒窗口到达，每个都看到 cache miss。你支付 10 次 write premium，0 次 read discount。

修复：batch with sequential-first。先单独发请求 1，等它填充 cache 后再发 2-10。给第一个 tool call 增加 300 ms；节省 5-10x 账单。

### The dynamic content anti-pattern / dynamic content 反模式

你的 system prompt 看起来是：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都是唯一的。每个请求都写入。零命中。

修复：把真正静态的内容移到 cacheable prefix；把动态内容追加到 cache boundary 之后：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 这样做后，cache hit rate 从 7% 提升到 74%，并公开了结构。

### Stack batch + cache for overnight workloads / overnight workloads 叠加 batch + cache

Batch APIs（Phase 17 · 15）在 24-hour turnaround 下给 50% discount。Cached input 还能在其上再给约 10x。Overnight classification、labeling 和 report generation workloads 通过叠加，可以降到 synchronous-uncached cost 的约 10%。

### Numbers you should remember / 你应该记住的数字

Pricing points 来自 2026-04 的 vendor docs，几个月就可能漂移；依赖前要重查。

- Anthropic cached read：Claude 3.5 Sonnet 上 $0.30/M，约比 fresh input 便宜 10x（docs.anthropic.com）。
- Anthropic cache write premium：1.25x（5-min TTL）或 2x（1-hour TTL）。
- OpenAI auto-cache：适用于 prompts ≥1024 tokens；当前 rate cards 上 cached input 约为 fresh input 的 10%（platform.openai.com）。
- Semantic cache hit rate（community-reported）：open chat 约 10%；structured FAQ 最高约 70%。不是 vendor-documented baseline。
- ProjectDiscovery：通过把 dynamic 移出 prefix，hit rate 7% → 74%（project blog，2025-11）。
- Parallelization anti-pattern：N parallel requests miss the first cache write 时，典型报告账单膨胀 5–10x。

## Build It / 动手构建

在 `code/main.py` 中构造 L1 semantic cache 与 L2 prompt cache 的两层账本，切换 parallelization 与 dynamic prefix，观察同样流量下账单如何变化。

## Use It / 应用它

`code/main.py` 在 mixed workloads 上模拟 L1 + L2 caching。它报告 hit rates、bill，并展示 parallelization penalty。

## Ship It / 交付它

本课产出 `outputs/skill-cache-auditor.md`。给定 prompt template 和 traffic，它会审计 cacheability 并建议 restructure。

## Exercises / 练习

1. 运行 `code/main.py`。切换 parallelization flag。账单变化多少？
2. 你的 system prompt 里有日期。把它移出去。展示 before/after hit rate math。
3. 给定 request arrival rate，计算 1-hour TTL（2x write）与 5-minute TTL（1.25x write）的 break-even。
4. Semantic cache 在 0.95 threshold 下命中 20%。0.85 时命中 50%，但出现错误 cached responses。选择正确阈值并说明理由。
5. 每个用户问题 batch 10 个并行 sub-queries。重写为 cache-friendly，同时不增加 end-to-end latency。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| L2 prompt cache | “prefix cache” | Provider 存储重复 prefix 的 KV |
| `cache_control` | “Anthropic cache marker” | 标记 cacheable blocks 的显式属性 |
| Cache write premium | “write tax” | 第一次 miss-to-cache 的额外成本（1.25x 或 2x） |
| L1 semantic cache | “embedding cache” | 调用 LLM 前的 app-level hash-and-embed |
| GPTCache | “LLM caching lib” | 流行 OSS L1 cache library |
| Cache hit rate | “hits / total” | 从 cache 服务的请求比例 |
| Parallelization anti-pattern | “the N-write trap” | N 个并行请求让 cache miss N 次 |
| Dynamic content trap | “the time-in-prompt trap” | prefix 中的 dynamic bytes 杀死 hit rate |
| RadixAttention | “intra-replica cache” | SGLang 的 prefix-cache implementation |

## Further Reading / 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官方 `cache_control` semantics 和 TTLs。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — automatic caching behavior 和 eligibility。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
