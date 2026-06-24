# CrewAI: Role-Based Crews and Flows / CrewAI：基于角色的 Crews 与 Flows

> CrewAI 是 2026 年基于角色的 multi-agent framework。四个原语：Agent、Task、Crew、Process。两个顶层形态：Crews（自主、基于角色的协作）和 Flows（event-driven、deterministic）。文档说得很直接：“for any production-ready application, start with a Flow.”

**Type / 类型：** Learn + Build / 学习 + 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process）以及各自拥有的职责。
- 区分 Sequential、Hierarchical 和计划中的 Consensus process，并为不同 workload 选择一种。
- 区分 Crews（自主 role-based）和 Flows（event-driven deterministic），并解释文档的生产建议。
- 用 `@tool` decorator 和 `BaseTool` subclass 接工具；理解 structured outputs 与 free text 的取舍。
- 说出 CrewAI 的四种 memory types，以及每种什么时候值得付出成本。
- 实现一个 stdlib three-agent crew（researcher、writer、editor），产出一份 brief。
- 识别三个 CrewAI failure modes：prompt-bloat、manager-LLM tax、brittle handoffs。

## The Problem / 问题

团队采用 multi-agent framework 时，常撞上同一堵墙。“Autonomous collaboration” 在 demo 里很好看。但客户报 bug 时，你需要 deterministic replay。财务问一次 LLM-routed crew 每次 run 要花多少钱。on-call 需要知道凌晨 3 点是哪个 Agent 卡住了。

自由形式的 LLM-routed crews 很难清楚回答这些问题。纯 DAG 都能回答，但会失去 brainstorming Agent 需要的探索形状。

CrewAI 的拆分诚实地面对了这个取舍。Crews 适合协作、角色化、探索式工作。Flows 适合 event-driven、code-owned、auditable production。同一个框架，两种形态，按 surface 选择。

## The Concept / 概念

### Four primitives / 四个原语

CrewAI 的表面很小。记住下面这组，剩下都是配置。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。backstory 是承重件。它塑造语气、判断、何时停止。tools 是 Agent 可调用的函数（下面详述）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。可复用工作单元。`expected_output` 是 contract。`context` 列出上游 tasks，其输出会传入。`output_pydantic` 强制结构化形状。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选 `memory` + `verbose` + `manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（planned）。决定 run 的形状。

Agents 不直接看见彼此。Tasks 引用 Agents。Crew 排列 tasks。Process 决定谁选择下一个 task。这就是完整心智模型。

> **Validated against** CrewAI 0.86 (2026-05). Newer versions may rename or merge process types; check the [CrewAI Processes docs](https://docs.crewai.com/concepts/processes) before relying on a specific shape.

### Sequential vs Hierarchical vs Consensus / Sequential、Hierarchical 与 Consensus

- **Sequential。** Tasks 按声明顺序运行。Task N 的输出可以作为 `context` 传给 Task N+1。成本最低，最可预测。顺序固定时使用。
- **Hierarchical。** manager Agent（额外 LLM call）在 specialists 之间路由。CrewAI 会基于你的 `manager_llm` config 或默认值生成 manager。manager 每轮选择下一个 task，并可拒绝或改路由。当你有四个以上 specialists 且顺序真的依赖先前输出时使用。
- **Consensus。** 已规划，但当前 public API 未实现。文档为未来 voting-based process 保留了这个名称。今天不要依赖它。

Hierarchical 在每个 specialist call 之外还会额外增加 per-round manager LLM call。五步 run 的 token cost 可能翻三倍。只有确实需要 routing 时才付这笔钱。

### Crews vs Flows / Crews 与 Flows

这是 2026 年文档的主叙事。

- **Crew。** LLM-driven autonomy。框架在 runtime 选择形状。适合：research、brainstorming、first drafts，以及任何“路径本身就是答案一部分”的场景。难 replay，难 test，原型很快。
- **Flow。** 你拥有的 event-driven graph。`@start` 标记入口。`@listen(topic)` 标记在另一步发出 topic 后触发的步骤。每一步都是普通 Python（也可以在内部调用 Crew）。适合：production。可观察、可测试、deterministic。

2026 年文档的生产建议：从 Flow 开始。当 autonomy 值得成本时，把 Crews 作为 Flow steps 中的 `Crew.kickoff()` calls 折进去。Flow 给 audit trail，Crew 给 exploration。要组合，不要二选一。

### Tool integration / 工具集成

给 Agent 工具有三种方式。选择符合需求的最简单方式。

1. **`@tool` decorator。** 纯函数变成工具。signature 是 schema；docstring 是 LLM 看到的 description。适合一次性 helpers。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass。** 类形式工具，有显式 args schema、async support、retries。当工具有 state（client、cache）或需要 structured args 时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits。** CrewAI 提供 first-party adapters：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一个 import 就能接入。

Structured outputs 使用 Pydantic。在 Task 上传入 `output_pydantic=MyModel`。CrewAI 会用该模型校验 LLM response，并尝试 coerce 或 retry。要和紧凑的 `expected_output` 字符串配合。Free-text outputs 适合 drafts；structured outputs 才是下游 Flows 可以消费的形状。

### Memory hooks / Memory hooks

CrewAI 内置四种 memory types。它们可以组合：一个 Crew 可以同时启用四种。

> **Validated against** CrewAI 0.86 (2026-05). Recent releases route everything through a unified `Memory` system that wraps these four stores. The conceptual model below still holds, but the public class surface may collapse to a single `Memory` entry-point in newer versions; check [CrewAI memory docs](https://docs.crewai.com/concepts/memory) for the current API.

- **Short-term。** 单次 run 内的 conversation buffer。run 结束即清空。
- **Long-term。** 跨 run 持久化。存储在 vector DB（默认 Chroma，可替换）。按当前 task 的 similarity 检索。
- **Entity。** per-entity facts。“Customer X is on the enterprise plan.” 按 entity 而不是 similarity 做 key。跨 run 存活。
- **Contextual。** assembly-time retrieval。在 Agent 需要时拉取相关 memory，而不是提前全部加载。

通过 `memory=True` 或 per-type config 在 Crew 上启用。背后使用你配置的 embeddings provider（默认 OpenAI，可换成本地）。Memory 是 CrewAI 相对更轻框架有价值的地方之一；纯 LangGraph 需要你自己接每一种 store。

### When CrewAI fits / CrewAI 适合哪里

- 三到六个带命名角色的 Agents 和一个协作 workflow。Drafting、reviewing、planning、brainstorming。
- LLM 对下一步的判断本身有价值的 routing（Hierarchical）。
- 团队更愿意阅读 `role + goal + backstory`，而不是 graph definition 的场景。

### When CrewAI does not fit / CrewAI 不适合哪里

- 有严格顺序的 deterministic DAG。用 LangGraph（Lesson 13）。graph shape 才是正确抽象；CrewAI 的角色 framing 会增加摩擦。
- sub-second latency budgets。Hierarchical 会增加 round trips。即使 Sequential 也会序列化包含 backstories 和 prior outputs 的 prompts。
- Single-agent loops。跳过框架；Agent loop（Lesson 1）加 tool registry 更短。

Lesson 17（Agent Framework Tradeoffs）会用矩阵展开。简短说：CrewAI 位于 “collaborative role-based” 角落。

### Dependency shape / 依赖形态

不依赖 LangChain。Python 3.10 到 3.13。使用 `uv`。star count 见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（2026-05 snapshot）。AWS Bedrock integration 有文档；vendor benchmarks 声称在 QA workloads 上比 LangGraph 大幅提速，但方法（dataset、hardware、evaluation metric）未公开，因此把框架厂商数字只当方向性参考。

### Where this pattern goes wrong / 这个模式在哪里会出错

- **Prompt-bloat from backstories。** 每个 Agent 2000 字 backstory，五个 Agent 的 crew 在第一次 tool call 前就烧完 context budget。backstories 控制在 200 字以内。跨 agents 复用短语；不要重复五遍 house style。
- **Manager-LLM token tax。** Hierarchical process 在每个 specialist call 前增加 manager LLM call。五个 task crew 变成六次 LLM calls，而且 manager call 携带完整 task list 和 prior outputs。除非 routing 依赖 output，否则切到 Sequential。
- **Brittle handoffs。** Task N 的 `expected_output` 是 “an outline”。Task N+1 把它当 `context` 读取，并尝试解析三个 sections。LLM 产出了四个。下游 Agent 即兴发挥。给 Task N 加 `output_pydantic`，让 Task N+1 读 typed object，而不是 free text。
- **Crew-as-prod。** Free-form Crew 没有 Flow wrapper 就上生产。输出变异高；无法 replay；on-call 不能 diff bad run 和 good run。用 Flow 包住。

## Build It / 动手构建

`code/main.py` 实现两种形态的 stdlib 版本，以及一个 three-agent crew。

形状：

- `Agent`、`Task` dataclasses，对齐 CrewAI surface。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行 tasks，并把 outputs 作为 `context` 传递。
- `HierarchicalCrew.kickoff(topic)` 增加 manager Agent，每轮选择下一个 specialist，在 “done” 时停止。
- `Flow`，带 `@start` 和 `@listen(topic)` decorators、一个 tiny event loop 和 trace。
- `tool(name)` decorator，镜像 CrewAI 的 `@tool` 形状。
- `Memory`，含 `short_term`、`long_term`、`entity` stores；mock similarity 使用 numpy。
- Mock LLM responses 是基于 role 和 input prefix 的 hardcoded strings。无网络。确定性。

具体 demo：researcher、writer、editor crew 产出关于 “agent engineering 2026” 的 brief。Researcher 拉取（mocked）sources。Writer 起草。Editor 收紧。同一个 crew 也通过 Flow 跑一遍，以展示 deterministic shape。

运行：

```bash
python3 code/main.py
```

Trace 覆盖：sequential crew 如何通过 `context` 串接 outputs，hierarchical crew 的 manager picks（researcher、writer、editor，然后 “done”），flow 如何通过显式 topics（`researched`、`drafted`、`edited`）运行同样三步，tool calls 如何通过 `@tool` 路由，以及 long-term memory 如何跨两个 kickoffs 存活。

Crew trace 是流动的；manager 原则上可以改顺序。Flow trace 是固定的。这个选择就是本课重点。

## Use It / 应用它

- **CrewAI Flow** 用于生产。哪怕 Flow 只有一步调用 `Crew.kickoff()`。Flow 给你 audit boundary。
- **CrewAI Crew (Sequential)** 用于顺序清晰的协作工作，尤其 first drafts 和 review loops。
- **CrewAI Crew (Hierarchical)** 用于 routing 依赖 output 且你有四个以上 specialists 的场景。
- **LangGraph**（Lesson 13）用于显式 state machines、durable resume、strict ordering。
- **AutoGen v0.4**（Lesson 14）用于 actor-model concurrency 和 fault isolation。
- **OpenAI Agents SDK**（Lesson 16）用于 OpenAI-first products，带 handoffs 和 guardrails。
- **Claude Agent SDK**（Lesson 17）用于 Claude-first products，带 subagents 和 session store。

## Ship It / 交付它

`outputs/skill-crew-or-flow.md` 会为任务选择 Crew vs Flow，并 scaffold 最小实现。它会 hard reject Crew-without-backstory、Flow-without-explicit-topics、少于三个 specialists 的 Hierarchical。

## Pitfalls / 常见坑

- **Backstory as flavor。** 它会塑造输出。每个 Agent 测三个 variants；variance 很真实。选一个，冻结它。
- **Skipping `expected_output`。** 没有 per-task contract，下游 task 会拿到 LLM 随便产出的东西。Crew 能跑，audit 会失败。
- **Memory always-on。** long-term 每次 run 都写。vector DB 增长。retrieval 变噪。只在事实确实 persistent 的 tasks 上写。
- **Manager prompt drift。** Hierarchical 的 manager prompt 是隐式的。routing 奇怪时，打开 verbose mode，把它 dump 出来读。
- **Tool side effects in Crews。** Crew 可能调用工具比预期更多次。POST、DELETE、payment 应在 Flow step 中，永远不要放成 Crew tool。

## Exercises / 练习

1. 把 Sequential crew 转成 Flow。数一数 variability 降低了哪些 touchpoints。记录 readability 哪里下降了。
2. 给 crew 加 entity memory：关于某个 customer 的 facts 跨 kickoffs 持久化。验证 retrieval 拉取的是正确 entity。
3. 实现一个 Hierarchical process：manager 在 writer output 至少三段前拒绝路由到 editor。trace retry。
4. 给（mocked）web search 接一个 `BaseTool` subclass。比较它和 `@tool` decorator 版本的 trace shape。
5. 给 editor task 增加 `output_pydantic=Brief`，其中 `Brief` 含 `title`、`summary`、`sections`。让 writer task 先输出一次 malformed JSON；验证 CrewAI 的 retry behavior 在 trace 中体现。
6. 阅读 CrewAI docs intro。把 toy 移植到真实 `crewai` API。stdlib version 跳过了哪些 guarantees？
7. 把 AgentOps 或 Langfuse（Lesson 24）接到真实 run。stdlib version 缺了哪些 traces？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Agent | “Persona” | Role + goal + backstory + tools |
| Task | “Unit of work” | Description + expected output + assignee + optional structured output |
| Crew | “Agent team” | Agents + Tasks + Process 的容器 |
| Process | “Execution strategy” | Sequential / Hierarchical / Consensus (planned) |
| Flow | “Deterministic workflow” | Event-driven、code-owned、testable |
| Backstory | “Persona prompt” | 塑造 Agent 语气和判断的 prompt |
| `@tool` | “Function tool” | 把函数变成 Agent 可调用工具的 decorator |
| `BaseTool` | “Class tool” | 带 args schema、retries、async support 的 class-based tool |
| Entity memory | “Per-entity facts” | scoped 到 customer / account / issue 的 memory |
| Long-term memory | “Cross-run memory” | 在 kickoffs 之间存活的 vector-backed memory |
| Contextual memory | “Just-in-time retrieval” | Agent 需要时才拉取 memory |
| Manager LLM | “Router agent” | Hierarchical process 中选择下一个 task 的额外 LLM |
| `expected_output` | “Task contract” | 告诉 Agent（也告诉 audit）要返回什么形状的字符串 |

## Further Reading / 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): concepts and the recommended production path
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): event-driven shape, `@start`, `@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools): `@tool`, `BaseTool`, built-in toolkits
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): short-term, long-term, entity, contextual
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): when multi-agent helps and when it does not
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): the state-machine alternative
