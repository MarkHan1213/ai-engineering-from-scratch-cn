# Parallel Tool Calls and Streaming with Tools / 并行工具调用与工具流式输出

> 三个彼此独立的天气查询如果串行执行，就要走三次 round trip。并行运行后，总耗时会收缩到最慢的一次调用。现在每个 frontier provider 都能在单轮中发出多个 tool call。收益真实存在，plumbing 也更微妙。本课会讲清两半：parallel fan-out 和 streamed-argument reassembly，重点放在 id-correlation 这个容易踩坑的地方。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, thread pool + streaming harness)
**Prerequisites / 前置知识：** Phase 13 · 02 (function calling deep dive)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释为什么需要 `parallel_tool_calls: true`，以及什么时候应该关闭它。
- 在 parallel fan-out 期间，把 streamed argument chunks 关联到正确的 tool-call id。
- 在不提前 parse 的前提下，把 partial `arguments` strings 重新组装为完整 JSON。
- 跑一个三城市天气 benchmark，展示 sequential vs parallel latency。

## The Problem / 问题

没有 parallel calls 时，agent 回答“Bengaluru、Tokyo 和 Zurich 天气如何”会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM round trip，而且每次还要支付 executor latency。大约是理想 wall-clock time 的 4 倍。

有 parallel calls 时：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

只需要一次 LLM round trip。Executor time 取三者最大值，而不是总和。OpenAI、Anthropic、Gemini 上的生产 benchmark 都显示，fan-out workload 的 wall-clock 可减少 60% 到 70%。

代价是 correlation complexity。三个调用乱序完成时，你返回的结果必须带上 matching `tool_call_id`，模型才能对齐。结果流式到达时，你必须先把 partial argument fragments 组装成完整 JSON，再执行。Gemini 3 加入 unique ids，一部分原因就是解决现实中两个并行调用同一个工具时无法区分的问题。

## The Concept / 概念

### Enabling parallel / 启用并行

- **OpenAI.** `parallel_tool_calls: true` 默认开启。设为 `false` 强制串行。
- **Anthropic.** 通过 `disable_parallel_tool_use: false` 并行（Claude 3.5 及以上默认）。设为 `true` 串行。
- **Gemini.** 始终具备并行能力；`tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

当工具之间有顺序依赖（`create_file` 再 `write_file`）、一个调用输出会成为另一个调用输入，或 rate limiter 扛不住 fan-out 时，关闭并行。

### Id correlation / Id 关联

模型发出的每个调用都有一个 `id`。host 返回的每个结果都必须包含同一个 id。否则结果就会歧义。

- **OpenAI.** 每个 tool-role message 上的 `tool_call_id`。
- **Anthropic.** 每个 `tool_result` block 上的 `tool_use_id`。
- **Gemini.** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 按 name 匹配，同名并行调用会坏）。

### Running calls concurrently / 并发运行调用

host 会把每个调用的 executor 放到独立 thread、coroutine 或 remote worker 上运行。最简单的 harness 使用 thread pool；生产中通常用 asyncio 配合 `asyncio.gather` 或 structured concurrency。完成顺序不可预测，id 才是真正的标识符。

一个常见 bug 是按 call-list order 回复结果，而不是按 completion order。通常这也能工作，因为模型只关心 `tool_call_id`；但如果结果丢失或重复，乱序提交更难排查。更好的做法是按 completion order 回复，并显式携带 ids。

### Streaming tool calls / 流式工具调用

模型流式输出时，`arguments` 会分片到达。三个 parallel calls 的 chunk 会在同一条 wire stream 上交错。你需要为每个 id 准备一个 accumulator。

各 provider 的形状：

- **OpenAI.** 每个 chunk 是 `choices[0].delta.tool_calls[i].function.arguments`（partial string）。chunk 携带 `index`（call list 中的位置）。你按 index 累积，在 id 首次出现时读取它，并在 `finish_reason = "tool_calls"` 时 parse JSON。
- **Anthropic.** Stream events 是 `message_start`，随后每个 block 有一个 `content_block_start`，type 为 `tool_use`（包含 id、name、empty input）。`content_block_delta` events 携带 `input_json_delta` chunks。`content_block_stop` 关闭每个 block。
- **Gemini.** `streamFunctionCallArguments`（Gemini 3 及以上）发出的 chunk 带有 `functionCallId`，因此调用可以干净交错。Gemini 3 之前，streaming 一次返回一个完整调用。

### Partial JSON and the parse-early trap / Partial JSON 与过早 parse 陷阱

在 `arguments` 完整之前不能 parse。像 `{"city": "Beng` 这样的 partial JSON 不是合法 JSON，会抛错。正确的 gate 是 provider 的 end-of-call signal：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`，或 Gemini 的 stream-end event。只有那时才调用 `json.loads`。更健壮的方案是使用 incremental JSON parser，在结构完成时产出事件；OpenAI streaming guide 推荐这种方式，用于展示实时 "thinking" indicator。用 brace-counting 判断完整性并不可靠（quoted string 或 escaped content 中的 brace 会造成 false positive），最多只能作为非正式 debug heuristic。

### Out-of-order completion / 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

host reply 仍然必须引用 ids：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在 OpenAI 或 Anthropic 上，reply 中的顺序不影响正确性。Gemini 也接受任意顺序，只要 ids 匹配。

### Benchmark: sequential vs parallel / Benchmark：串行 vs 并行

`code/main.py` 中的 harness 用 400、600、800 ms latency 模拟三个 executor。Sequential 总耗时 1800 ms。Parallel 总耗时 max(400, 600, 800) = 800 ms。差异是常量项，不是比例项，所以工具数量越多，节省越明显。

现实 caveat：parallel calls 会给下游 API 施压。对 rate-limited service 做 10-way fan-out 会失败。Phase 13 · 17 会讲 gateway-level backpressure；retry semantics 会放到未来 phase。

### Streaming fan-out wall-clock / 流式 fan-out 的 wall-clock 优化

如果模型本身在 streaming，你可以在某个调用的 arguments 完成时立刻开始执行，而不是等所有调用都 finalize。这是 OpenAI 文档中提到的优化，但不是所有 SDK 都暴露。本课 harness 就这样做：模拟 stream 一旦产出完整 argument object，host 立即启动该调用。

## Build It / 动手构建

本课要构建两块机制：先用 thread pool 跑出 sequential 和 parallel 的可测量差异，再用 `StreamAccumulator` 处理同一条 stream 中交错出现的 `arguments` chunks。核心约束是：每个 buffer 必须按 id 隔离，且只能在 provider 发出完成信号后 parse。

## Use It / 应用它

`code/main.py` 分成两半。第一半用 `concurrent.futures.ThreadPoolExecutor` 顺序和并行运行三个 simulated weather calls，并打印 wall-clock time。第二半回放一个 fake streaming response：三个 parallel calls 的 `arguments` chunks 在同一条 stream 上交错，并由 `StreamAccumulator` 按 id 重新组装。不需要 LLM、不需要网络，只看 reassembly logic。

重点看：

- sequential timer 约为 1.8 秒。parallel timer 在同样 fake latencies 下约为 0.8 秒。
- accumulator 通过 per-id buffering 处理乱序到达的 chunks，并且只在每个 call 的 JSON 完成后 parse。
- executor 在某个 id 的 arguments finalize 后就会启动，而不是等所有 streams 结束。

## Ship It / 交付它

本课产出 `outputs/skill-parallel-call-safety-check.md`。给定一个 tool registry，这个 skill 会审计哪些工具可以安全并行，哪些有 ordering dependencies，哪些会压垮下游 rate limits，并返回带有 per-tool `parallel_safe` flags 的修订 registry。

## Exercises / 练习

1. 运行 `code/main.py` 并调整 simulated latencies。确认 parallel-to-sequential ratio 近似为 `max/sum`（真实运行会因 thread scheduling、serialization 和 harness overhead 略偏离理想值）。什么样的 latency distribution 下，并行不再重要？

2. 扩展 accumulator，处理“call was cancelled mid-stream”的情况：丢弃对应 buffer，并发出 `cancelled` event。哪家 provider 明确记录了这个情况？检查 Anthropic 的 `content_block_stop` semantics 和 OpenAI 的 `finish_reason: "length"` behavior。

3. 把 thread pool 替换为 `asyncio.gather`。分别 benchmark。只有 executor 做真实 I/O 时，你才应该看到 async 因 lower context-switch cost 带来的小幅收益。

4. 选择两个不应该并行的工具（例如 `create_file` 再 `write_file`）。给 registry 增加一个 `ordering_dependency` graph，并基于该 graph gate parallel fan-out。这是 dependency-aware scheduling 的最小机制，未来 agent-engineering phase 会形式化它。

5. 阅读 OpenAI 的 parallel-function-calling section 和 Anthropic 的 `disable_parallel_tool_use` docs。找出 Anthropic 建议关闭并行的一个真实工具类型。（提示：同一资源上的 consequential mutations。）

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Parallel tool calls | “一轮 fan-out” | 模型在单个 assistant message 中发出多个 tool call |
| `parallel_tool_calls` | “OpenAI 的 flag” | 启用或禁用 multi-call emission |
| `disable_parallel_tool_use` | “Anthropic 的反向开关” | opt-out flag；默认启用并行 |
| Tool call id | “关联句柄” | 每个调用的标识符，result message 必须回显它 |
| Accumulator | “Stream buffer” | 用于 partial `arguments` chunks 的 per-id string buffer |
| Out-of-order completion | “最快的先回来” | 并行调用完成顺序不可预测；ids 是粘合剂 |
| Dependency graph | “顺序约束” | 输出会流入其他工具输入的工具，不能并行 |
| Parse-early trap | “JSON.parse 爆了” | 尝试解析尚未完整的 `arguments` string |
| `streamFunctionCallArguments` | “Gemini 3 feature” | 带 unique id 的 streamed argument chunks |
| Completion-order reply | “不要等全部完成” | 结果一到就按 id 回复 |

## Further Reading / 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — default behavior 和 opt-out flag
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 与 result batching
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 的 id-correlated parallel calls
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI streams 的 chunked argument reassembly
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — 带 `input_json_delta` 的 `content_block_delta`
