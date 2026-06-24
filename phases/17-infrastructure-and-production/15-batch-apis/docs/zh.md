# Batch APIs — the 50% Discount as Industry Standard / Batch APIs：50% 折扣成为行业标准

> 每家主要 provider 都提供 async batch API：50% discount，约 24-hour turnaround。OpenAI、Anthropic、Google，以及多数 inference platforms（Fireworks batch tier、Together batch）都实现了同一种模式。把 batch 与 prompt caching 叠加，overnight pipelines 可以降到 synchronous-uncached cost 的约 10%。规则极其简单：只要不是 interactive，就应该放到 batch。Content generation pipelines、document classification、data extraction、report generation、bulk labeling、catalog tagging，只要能容忍 24-hour latency，没迁到 batch 就是在把钱留在桌上。2026 年生产模式是把每个新的 LLM workload 分到三条 lane：interactive（synchronous with caching）、semi-interactive（async queue with fallback）、batch（overnight，stack cached input）。那些假装 interactive 但其实能容忍分钟级延迟的 workloads 最浪费。

**类型：** 学习
**语言：** Python（stdlib, toy batch-vs-sync cost simulator）
**前置知识：** 第 17 阶段 · 14（Prompt & Semantic Caching）
**时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 说出三个 provider batch APIs（OpenAI、Anthropic、Google），以及共同的 50% discount + 24h turnaround guarantees。
- 对 overnight classification workload 计算 batch + cached-input 叠加后的成本，并与 synchronous-uncached baseline 比较。
- 把 workload 分到 interactive / semi-interactive / batch，并说明理由。
- 说出两个陷阱：partial interactivity（用户期望快于 24h）和 output-schema drift（每个 provider 的 batch file format 不同）。

## The Problem / 问题

你的团队有一个 nightly report generation pipeline。50,000 份 documents，逐份 summarize，再 cluster summaries，最后起草 executive brief。同步运行需要 4 小时，每晚 $2,000。你听说了 batch APIs。

Batch 给你 50% off。你又在共享 system prompt 上启用 prompt caching。叠加后，账单降到 $180/night，约为 baseline 的 9%。同一 pipeline，三项配置变化。

Batch 是 LLM cost toolkit 中最便宜、但最少被使用的杠杆。原因主要是组织性的：团队以为“real-time”，但真实 SLA 是“by morning”。本课的目标就是不要把 90% 账单留在桌上。

## The Concept / 概念

### The three batch APIs / 三个 batch APIs

**OpenAI Batch API**：上传 JSONL file，里面是 request 列表。承诺 24-hour turnaround（实际通常约 2-8 小时）。输入和输出 tokens 都 50% discount。`/v1/batches` endpoint。符合 cache 条件的 inputs 还能叠加 cached-input pricing。

**Anthropic Message Batches**：上传 JSONL。24-hour turnaround。50% discount。支持 `cache_control`：cache writes 显式，batch 内 reads 自动发生。

**Google Vertex AI Batch Prediction**：BigQuery 或 GCS input。Gemini 类似 50% discount。与 Vertex pipelines 集成。

### Semantic: asynchronous, not slow / 语义是异步，不是慢

Batch 意味着“我承诺 24 小时内返回”，不是“这会花 24 小时”。典型 P50 是 2-6 小时。Provider 会在 GPU inventory 利用不足的 off-peak windows 调度你的 batch。

### Stack with caching / 与 caching 叠加

一个 50k-document summarization，共享同一个 4K-token system prompt：

- Synchronous uncached：50000 × ($input × 4000 + $output × 200)，全价。
- Synchronous cached：system prompt 第一次写入 cache；剩余 49999 次获得 10x cheaper input。
- Batch cached：以上所有，再对读写都打 50% discount。

叠加：batch + cache = sync uncached bill 的约 10%。任何 overnight 运行且有 shared system prompt 的 workload 都应该这么做。

### Workload triage / Workload 分流

**Interactive** — 用户等待响应。TTFT 重要。Synchronous call with prompt caching。不能 batch。

**Semi-interactive** — 用户提交任务，几分钟后回来查看。Async queue，batch 不可用时 fallback to sync。比如中等规模 RAG indexing。

**Batch** — 用户期望“明早”或“下小时”看到结果。Content pipelines、classification at scale、offline analysis。永远 batch，永远叠加 caching。

常见错误：因为 pipeline 是 production，就把所有东西都归类为 interactive。Production 不是 latency spec，SLA 才是。

### The partial-interactivity trap / partial-interactivity 陷阱

有些功能看起来 interactive，但能容忍 5-10 分钟。例如：带“refresh”按钮的 nightly customer health report。用户点 refresh，等 10 分钟可以接受。团队却把它做成 synchronous。50 个并发 refresh 的成本是 batched-and-delivered-via-email 的 10x。

要问的问题是：“24 小时对这个用户意味着什么？”如果答案是“他们不会注意到”，就 batch。

### The output-schema trap / output-schema 陷阱

每个 provider 的 batch file format 都不同：

- OpenAI：JSONL，一行一个 request。
- Anthropic：JSONL，一行一个 message；response format 嵌入其中。
- Vertex：BigQuery table 或 GCS prefix with TFRecord。

跨 providers 写“一个 batch client”意味着每个 provider 都要 adapter code。宣传 multi-provider batch 的 gateways（Portkey、LiteLLM 某些 tiers）通常也只是 thin-wrap raw format。

### Numbers you should remember / 你应该记住的数字

- Provider 间 batch discount：input + output 固定 50%。
- Turnaround SLA：24 hours guaranteed，典型 P50 2-6 hours。
- Stacked batch + cached input：约为 sync uncached cost 的 10%。
- Workload triage rule：如果 24h latency 可接受，永远 batch。

## Build It / 动手构建

用 `code/main.py` 把文档量、共享 prompt 长度、输出长度和 token prices 参数化，算出 sync、sync+cache、batch、batch+cache 四条路径的账单。

## Use It / 应用它

`code/main.py` 会在 50k-document workload 上计算 sync、sync+cache、batch、batch+cache 的成本，并报告节省的 $ 和百分比。

## Ship It / 交付它

本课产出 `outputs/skill-batch-triager.md`。给定 workload characteristics，它会分流到 interactive/semi/batch，并估算 savings。

## Exercises / 练习

1. 运行 `code/main.py`。对一个 100k-doc pipeline，3K-token system prompt、500-token output，计算 full stack（batch + cache）相比 sync baseline 的 savings。
2. 选三个你熟悉真实产品中的功能。分别分到 interactive/semi/batch。
3. 用户抱怨报告花了 3 小时。这是 batch mis-triage，还是合理 interactive？写出 decision criterion。
4. 你的 batch API return SLA 是 24h，但 P99 是 20 小时。你如何对用户沟通？边缘情况里 downstream system 行为是什么？
5. 计算 break-even：shared-prefix length 到多少时，batch + cache 比在自有 reserved GPU 上 overnight 跑更便宜？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Batch API | “async discount” | 50% off，24h turnaround |
| JSONL | “batch format” | 一行一个 JSON request；OpenAI/Anthropic 标准 |
| Message Batches | “Anthropic batch” | Anthropic 的 batch API 产品名 |
| Batch prediction | “Vertex batch” | Vertex AI 的 batch API 产品 |
| Turnaround SLA | “24h promise” | 保证，不是典型值；典型 2-6h |
| Workload triage | “interactivity decision” | Interactive / semi / batch routing decision |
| Output schema | “response format” | 每个 provider 的 JSONL layout；不可移植 |
| Stacked discount | “batch + cache” | 两者都适用时约为 uncached sync bill 的 10% |

## Further Reading / 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL format 和 `/v1/batches` semantics。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch format 和 `cache_control` interaction。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch semantics。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
