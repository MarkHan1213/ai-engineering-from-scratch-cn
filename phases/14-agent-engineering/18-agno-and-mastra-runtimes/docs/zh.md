# Agno and Mastra: Production Runtimes / Agno 与 Mastra：生产 Runtime

> Agno（Python）和 Mastra（TypeScript）是 2026 年的一组 production-runtime 对照。Agno 目标是微秒级 Agent 实例化和 stateless FastAPI backends。Mastra 在 Vercel AI SDK 基座上提供 agents、tools、workflows、统一 model routing 和 composite storage。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python, TypeScript
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 识别 Agno 的性能目标，以及这些目标什么时候重要。
- 说出 Mastra 的三个原语：Agents、Tools、Workflows，以及支持的 server adapters。
- 解释为什么推荐使用 stateless session-scoped FastAPI backend 作为 Agno 生产路径。
- 根据技术栈选择 Agno vs Mastra（Python-first vs TypeScript-first）。

## The Problem / 问题

LangGraph、AutoGen、CrewAI 都偏 framework-heavy。想要 “just the agent loop, fast, in my runtime” 的团队会看 Agno（Python）或 Mastra（TypeScript）。两者都用一部分 framework-owned primitives 换来原始速度和对周边 stack 的更紧密适配。

## The Concept / 概念

### Agno / Agno

- Python runtime，前身是 Phi-data。
- “No graphs, chains, or convoluted patterns — just pure python.”
- 文档中的性能目标：约 2μs agent instantiation、约 3.75 KiB memory per agent、约 23 个 model providers。
- 生产路径：stateless session-scoped FastAPI backend。每个 request 启动新 Agent；session state 存在 DB 中。
- 原生 multimodal（text、image、audio、video、file）和 agentic RAG。

当你每秒有上千个短生命周期 Agents（chat fan-in、evaluation pipelines）时，速度目标重要。一个 Agent 跑 10 分钟时，实例化速度不重要。

### Mastra / Mastra

- TypeScript，构建在 Vercel AI SDK 上。
- 三个原语：**Agents**、**Tools**（Zod-typed）、**Workflows**。
- Unified Model Router：3300+ models，跨 94 providers（2026 年 3 月）。
- Composite storage：memory、workflows、observability 可分别接不同 backends；规模化 observability 推荐 ClickHouse。
- Apache 2.0，但源码中的 `ee/` 目录使用 source-available enterprise license。
- Server adapters 支持 Express、Hono、Fastify、Koa；first-class Next.js 和 Astro integration。
- 提供 Mastra Studio（localhost:4111）用于 debugging。
- 1.0（2026 年 1 月）时 22k+ GitHub stars、300k+ weekly npm downloads。

### Positioning / 定位

两者都不是要做 LangGraph。它们竞争的是：

- **Language fit。** Python-first 选 Agno；TypeScript-first 选 Mastra。
- **Runtime ergonomics。** Agno = near-zero overhead；Mastra = 与 Vercel ecosystem 集成。
- **Observability。** 二者都集成 Langfuse / Phoenix / Opik（Lesson 24），但 Mastra Studio 是 first-party。

### When to pick each / 什么时候选哪个

- **Agno**：Python backend、很多短生命周期 Agents、强性能要求、FastAPI shop。
- **Mastra**：TypeScript backend、Next.js / Vercel deploy、统一 multi-provider model routing、Zod-typed tools。
- **LangGraph**（Lesson 13）：durable state 和显式 graph reasoning 比原始速度更重要。
- **OpenAI / Claude Agent SDK**：想要 provider productized shape（Lessons 16-17）。

### Where this pattern goes wrong / 这个模式在哪里会出错

- **Perf-for-perf's-sake。** 因为 “2μs” 好听而选择 Agno，但 workload 每个 request 只有一次慢 Agent call。overhead 不是瓶颈。
- **Ecosystem lock-in。** Mastra 的 Vercel-flavored integration 在 Vercel 上是优点，离开 Vercel 可能变成缺点。
- **Enterprise license confusion。** Mastra 的 `ee/` 目录是 source-available，不是 Apache 2.0。计划 fork 前先读 license。

## Build It / 动手构建

本课主要是比较，没有一个代码 artifact 能公平覆盖两个框架。`code/main.py` 提供一个 side-by-side toy：最小 “run an agent, stream the output, persist session” flow 实现两遍，一遍 Agno-shaped，一遍 Mastra-shaped。

运行：

```
python3 code/main.py
```

你会看到两个结构不同但功能等价的 traces。

## Use It / 应用它

- **Agno**：需要速度和 FastAPI shape 的 Python backend。
- **Mastra**：有很多 providers 和 workflow primitives 的 TypeScript backend。
- 两者都提供 first-party observability hooks，并都能集成 Langfuse。

## Ship It / 交付它

`outputs/skill-runtime-picker.md` 会根据 stack、latency budget 和 operational shape，在 Agno、Mastra、LangGraph 或 provider SDK 之间选择。

## Exercises / 练习

1. 阅读 Agno docs。把 stdlib ReAct loop（Lesson 01）移植到 Agno。什么消失了？什么保留了？
2. 阅读 Mastra docs。把同一个 loop 移植到 Mastra。tool typing（Zod vs nothing）发生了什么变化？
3. Benchmark：测你的 stack 上 agent instantiation latency。Agno 的 2μs 对你的 workload 重要吗？
4. 设计迁移：如果你一直在 Python 中跑 CrewAI，切到 Agno 会破坏什么？
5. 阅读 Mastra 的 `ee/` license terms。哪些限制会影响开源 fork？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Agno | “Fast Python agents” | Stateless session-scoped agent runtime |
| Mastra | “TypeScript agents on Vercel AI SDK” | Agents + Tools + Workflows + Model Router |
| Unified Model Router | “Multi-provider access” | 跨 94 providers、3300+ models 的单客户端 |
| Composite storage | “Multiple backends” | Memory / workflows / observability 分别接不同 store |
| Mastra Studio | “Local debugger” | localhost:4111 UI，用于 introspecting agents |
| Source-available | “Not OSS” | 允许阅读源码但限制商业使用的 license |

## Further Reading / 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — performance targets、FastAPI integration
- [Mastra docs](https://mastra.ai/docs) — primitives、server adapters、Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — stateful-graph alternative
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra integrations 引用的 observability comparisons
