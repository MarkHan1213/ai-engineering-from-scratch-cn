# OpenAI Agents SDK: Handoffs, Guardrails, Tracing / OpenAI Agents SDK：Handoffs、Guardrails、Tracing

> OpenAI Agents SDK 是构建在 Responses API 之上的轻量 multi-agent framework。五个原语：Agent、Handoff、Guardrail、Session、Tracing。Handoffs 是命名为 `transfer_to_<agent>` 的工具。Guardrails 可以在 input 或 output 上触发。Tracing 默认开启。

**Type / 类型：** Learn + Build / 学习 + 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 说出 OpenAI Agents SDK 的五个原语。
- 解释 handoffs：为什么它们被建模为 tools，模型看到的 name shape 是什么，以及 context 如何传递。
- 区分 input guardrails、output guardrails 和 tool guardrails；解释 `run_in_parallel` 与 blocking mode 的区别。
- 实现一个 stdlib runtime，带 handoffs、guardrails 和 span-style tracing。

## The Problem / 问题

不能干净 delegate 的 Agent，最后会把所有东西都塞进一个 prompt。没有 guardrails 的 Agent 会泄露 PII、产出违反 policy 的输出，或陷入无限循环。OpenAI 的 SDK 把 multi-agent 可控化所需的三个原语固定下来。

## The Concept / 概念

### Five primitives / 五个原语

1. **Agent。** LLM + instructions + tools + handoffs。
2. **Handoff。** 委派给另一个 Agent。对模型呈现为名为 `transfer_to_<agent_name>` 的 tool。
3. **Guardrail。** 对 input（仅第一个 Agent）、output（仅最后一个 Agent）或 tool invocation（每个 function tool）做 validation。
4. **Session。** 自动跨 turns 保存 conversation history。
5. **Tracing。** 对 LLM generations、tool calls、handoffs、guardrails 内建 spans。

### Handoffs as tools / Handoff 作为工具

模型在工具列表中看到 `transfer_to_billing_agent`。调用它表示 runtime 需要：

1. 复制 conversation context（或通过 `nest_handoff_history` beta 折叠它）。
2. 用目标 Agent 的 instructions 初始化它。
3. 继续让目标 Agent 接管当前 run。

这是 supervisor pattern（Lesson 13 / Lesson 28）的产品化。

### Guardrails / Guardrails

三种类型：

- **Input guardrails。** 在第一个 Agent 的 input 上运行。任何 LLM call 之前拒绝 unsafe 或 out-of-scope request。
- **Output guardrails。** 在最后一个 Agent 的 output 上运行。捕获 PII leaks、policy violations、malformed responses。
- **Tool guardrails。** 在每个 function-tool 上运行。校验参数、检查权限、审计执行。

模式：

- **Parallel**（默认）。Guardrail LLM 和主 LLM 并行运行。tail latency 更低。如果触发，主 LLM 的工作被丢弃（浪费 token）。
- **Blocking**（`run_in_parallel=False`）。Guardrail LLM 先运行。如果触发，不会在主调用上浪费 token。

Tripwires 会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing / Tracing

默认开启。每个 LLM generation、tool call、handoff 和 guardrail 都发出 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可关闭。`add_trace_processor(processor)` 会把 spans 同步发送到你自己的 backend，并保留 OpenAI backend。

### Sessions / Sessions

`Session` 把 conversation history 存入 backend（SQLite、Redis、自定义）。`Runner.run(agent, input, session=session)` 会自动 load 并 append。

### Where this pattern goes wrong / 这个模式在哪里会出错

- **Handoff drift。** Agent A hand off 给 Agent B，Agent B 又 hand back 给 Agent A。增加 hop counter。
- **Guardrail bypass。** Tool guardrails 只在 function tools 上触发；built-in tools（file reader、web fetch）需要单独 policy。
- **Over-tracing。** spans 中含敏感内容。搭配 OTel GenAI content-capture rules（Lesson 23）：内容外存，span 只引用 ID。

## Build It / 动手构建

`code/main.py` 用 stdlib 实现 SDK 形状：

- `Agent`、`FunctionTool`、`Handoff`（作为具有 transfer semantics 的 function tool）。
- `Runner`，包含 input/output/tool guardrails、handoff dispatch 和 hop counter。
- 一个 simple span emitter，用来展示 trace shape。
- 一个 triage agent，根据用户 query hand off 到 billing 或 support；其中一条 input 会触发 guardrail。

运行：

```
python3 code/main.py
```

trace 会展示两次成功 handoff、一次 input guardrail trip，以及一棵和真实 SDK 类似的 span tree。

## Use It / 应用它

- **OpenAI Agents SDK** 用于 OpenAI-first products。
- **Claude Agent SDK**（Lesson 17）用于 Claude-first products。
- **LangGraph**（Lesson 13）用于你需要显式 state 和 durable resume 的场景。
- **Custom** 用于需要精确控制的场景（voice、multi-provider、federated deployments）。

## Ship It / 交付它

`outputs/skill-agents-sdk-scaffold.md` scaffold 一个 Agents SDK app，包含 triage agent、handoffs、input/output/tool guardrails、session store 和 trace processor。

## Exercises / 练习

1. 增加 handoff hop counter：超过 N 次 transfer 后拒绝。trace 行为。
2. 把 `nest_handoff_history` 实现为一个选项：transfer 前把 prior messages 折叠成一个 summary。
3. 写一个 blocking output guardrail。比较会触发的 prompts 和会通过的 prompts 的 latency。
4. 把 `add_trace_processor` 接到 JSON logger。每个 span 发出什么形状？
5. 阅读 SDK docs。把 stdlib toy 移植到 `openai-agents-python`。你建模错了什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Agent | “LLM + instructions” | SDK 中的 Agent type，拥有 tools 和 handoffs |
| Handoff | “Transfer” | 模型调用来 delegate 给另一个 Agent 的 tool |
| Guardrail | “Policy check” | input / output / tool invocation 上的 validation |
| Tripwire | “Guardrail trip” | guardrail 拒绝时抛出的异常 |
| Session | “History store” | 在 runs 之间持久化 conversation memory |
| Tracing | “Spans” | 内建覆盖 LLM + tool + handoff + guardrail 的可观测性 |
| Blocking guardrail | “Sequential check” | guardrail 先运行；触发时不浪费主调用 token |
| Parallel guardrail | “Concurrent check” | guardrail 并行运行；低延迟，但触发时浪费 token |

## Further Reading / 延伸阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — primitives、handoffs、guardrails、tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude-flavored counterpart
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 什么时候值得用 handoffs
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK spans 可映射到的标准
