# Production Runtimes: Queue, Event, Cron / 生产运行时：Queue、Event、Cron

> 生产 Agent 有六种 runtime shapes：request-response、streaming、durable execution、queue-based background、event-driven、scheduled。先选 shape，再选 framework。Observability 在每一种 shape 中都是 load-bearing。

**类型：** 学习
**语言：** Python（stdlib）
**前置知识：** 第 14 阶段 · 13（LangGraph）, 第 14 阶段 · 22（Voice）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出六种 production runtime shapes，并把每一种匹配到 framework / product pattern。
- 解释为什么 durable execution（LangGraph）对 long-horizon tasks 很重要。
- 描述 event-driven runtime，以及 Claude Managed Agents 何时适合。
- 解释 observability-as-load-bearing 这个说法在 multi-step agents 中的含义。

## The Problem / 问题

生产 Agent 会以 Jupyter notebook 暴露不出来的方式失败：第 37 步网络超时、用户在语音通话中途挂断、cron job 因机器重启而死掉、background worker 内存耗尽。runtime shape 决定哪些 failure 是可恢复的。

## The Concept / 概念

### Request-response

- Synchronous HTTP。用户等待完成。
- 只适合短任务（<30s）。
- Stacks: Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- Observability: standard HTTP access logs + OTel spans。

### Streaming

- 用 SSE 或 WebSocket 做 progressive output。
- LiveKit 把这个扩展到 voice/video 的 WebRTC（Lesson 22）。
- Stacks: 任何支持 streaming 的 framework + 能处理 SSE/WS 的 frontend。
- Observability: per-chunk timing、first-token latency、tail latency。

### Durable execution

- 每一步之后 checkpoint state；失败后 auto-resumes。
- AutoGen v0.4 actor model 把 failures 隔离到单个 agent（Lesson 14）。
- LangGraph 的核心差异化能力（Lesson 13）。
- 当 step count 未知且 recovery cost 很高时必不可少。

### Queue-based / background

- Job 进入 queue，workers 拉取执行，结果通过 webhooks 或 pub/sub 回流。
- 对 long-horizon agents 必不可少（Anthropic 的 computer use announcement 中提到，每个 task 可能有 dozens-to-hundreds of steps）。
- Stacks: Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、custom。
- Observability: queue depth、per-job latency distribution、DLQ size。

### Event-driven

- Agents 订阅 triggers：new email、PR opened、cron fire。
- Claude Managed Agents 开箱覆盖这一类（Lesson 17）。
- CrewAI Flows（Lesson 15）用于组织 event-driven deterministic workflows。
- Observability: trigger source、event-to-start latency、agent latency。

### Scheduled

- 周期性运行的 cron-shaped agents。
- 与 durable execution 组合，让失败的 nightly run 在下一次 tick 恢复。
- Stacks: Kubernetes CronJob + durable framework；hosted（Render cron、Vercel cron）。

### 2026 deployment patterns / 2026 年部署模式

- **CrewAI Flows** 用于 event-driven production。
- **Agno** 用于 Python microservices 的 stateless FastAPI。
- **Mastra** server adapters（Express、Hono、Fastify、Koa）用于 embedding。
- **Pipecat Cloud / LiveKit Cloud** 用于 managed voice（Lesson 22）。
- **Claude Managed Agents** 用于 hosted long-running async。

### Observability is load-bearing / Observability 是承重结构

没有 OpenTelemetry GenAI spans（Lesson 23）加上 Langfuse/Phoenix/Opik backend（Lesson 24），你无法调试一个在第 40 步失败的 multi-step agent。这对生产不是可选项。它决定你是 “we debug fast”，还是 “we replay from scratch with more logging.”

### Where production runtimes fail / 生产运行时容易失败的地方

- **Wrong shape choice.** 给 5 分钟任务选 request-response。用户离开；workers 堆积；retries 叠加。
- **No DLQ.** Queue workers 没有 dead-letter。失败 jobs 直接消失。
- **Opaque background work.** Background agent 不导出 trace。直到用户报告前，失败不可见。
- **Skipping durable state.** 任何 > 30 秒且无法承受从头重跑的 run，都需要 durable execution。

## Build It / 动手构建

`code/main.py` 是一个 stdlib multi-shape demo：

- Request-response endpoint（plain function）。
- Streaming handler（generator）。
- 带 DLQ 的 queue-based worker。
- Event trigger registry。
- Cron-shaped scheduler。

运行：

```bash
python3 code/main.py
```

输出：五条 traces，展示同一个 task 在不同 shape 下的 behavior。相同 agent logic，不同 outer shells。Durable execution（第六种 shape）已经在 Lesson 13 通过 LangGraph checkpointing 专门覆盖，因此这里故意不重复。

## Use It / 应用它

- **Request-response** 用于 chat-style UX。
- **Streaming** 用于 progressive responses。
- **Durable** 用于 long-horizon tasks。
- **Queue** 用于 batch / async / long-running。
- **Event** 用于 agent reactivity。
- **Cron** 用于 housekeeping（memory consolidation、evals、cost reports）。

## Ship It / 交付它

`outputs/skill-runtime-shape.md` 会为一个 task 选择 runtime shape，并接入 observability requirements。

## Exercises / 练习

1. 把你的 Lesson 01 ReAct loop port 到你技术栈中的六种 shape。哪种 shape 适合哪种 product surface？
2. 给 queue-based demo 增加 DLQ。模拟 10% job failure；暴露 DLQ size。
3. 写一个 cron-triggered eval agent，每晚针对当天 top 20 traces 运行。
4. 实现带 backpressure 的 streaming：如果 client 很慢，就暂停 agent。这与 turn budget 如何相互作用？
5. 阅读 Claude Managed Agents docs。什么时候你会把 self-hosted long-horizon agent 迁到 managed？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Request-response | “Synchronous” | 用户等待；只适合短任务 |
| Streaming | “SSE / WS” | progressive output；UX 更好；每个 chunk 的 latency 可观测 |
| Durable execution | “Resume from failure” | checkpointed state；从最后一步 restart |
| Queue-based | “Background jobs” | producer / worker pool / DLQ |
| Event-driven | “Trigger-based” | Agent 对 external events 做反应 |
| DLQ | “Dead-letter queue” | failed jobs 的停放区 |
| Claude Managed Agents | “Hosted harness” | Anthropic-hosted long-running async with caching + compaction |

## Further Reading / 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — durable execution details
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted long-running async
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "dozens-to-hundreds of steps per task"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model fault isolation
