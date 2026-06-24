# ReWOO and Plan-and-Execute: Decoupled Planning / ReWOO 与 Plan-and-Execute：解耦式规划

> ReAct 把思考和行动交错在同一条流里。ReWOO 把它们拆开：先一次性做出大计划，再执行。它能减少 5 倍 token，在 HotpotQA 上提升 4% accuracy，而且可以把 planner 蒸馏到 7B 模型。Plan-and-Execute 把这个模式泛化；Plan-and-Act 又把它扩展到网页导航。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 ReWOO 的 Planner / Worker / Solver 拆分为什么比 ReAct 的交错循环更省 token、更稳健。
- 实现一个 plan DAG、按依赖顺序执行的 executor，以及把 worker 输出组合起来的 solver，全部只用 stdlib。
- 借助 2026 年 Anthropic “five workflow patterns” 框架，判断任务应该用 plan-then-execute 还是交错式 ReAct。
- 识别什么时候需要 Plan-and-Act 的 synthetic plan data 来处理 long-horizon web 或 mobile tasks。

## The Problem / 问题

ReAct 的 thought-action-observation 交错循环简单且灵活，但每一次工具调用都必须携带完整先前上下文，包括每个 previous thought。token 使用量会随深度近似二次增长。更糟的是，工具在中途失败时，模型要从 error observation 里重新推导整个计划。

ReWOO (Xu et al., arXiv:2305.18323, May 2023) 抓住了这个问题，并做了一个取舍：先把整体计划定下来，并行获取 evidence，最后组合答案。一次 LLM 调用用于 plan，N 次工具调用用于 evidence（可并行），一次 LLM 调用用于 solve。代价是灵活性更低（计划是静态的），换来更好的 token efficiency 和更清晰的 failure modes。

## The Concept / 概念

### The three roles / 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 生成一个 DAG。每个节点说明工具名、参数，以及依赖哪些更早节点（例如 `#E1`、`#E2`）。Workers 按拓扑顺序执行节点。Solver 把所有 evidence 拼成最终答案。

### Why 5x fewer tokens / 为什么能少 5 倍 token

ReAct 的 prompt length 会随 step count 线性增长。到第 10 步时，prompt 包含 thought 1、action 1、observation 1、thought 2、action 2、observation 2，以此类推。每个中间步骤还会重复携带原始 prompt。

ReWOO 只付一次 planner prompt（较大）、N 个小 worker prompts（每个只是 tool call，没有 chain）和一次 solver prompt。论文在 HotpotQA 上测得约 5 倍 token 减少，同时绝对 accuracy 增加 4 个点。

### Why it is more robust / 为什么更稳健

如果 worker 3 在 ReAct 中失败，循环必须在流中对 error 继续推理。在 ReWOO 中，worker 3 返回一个 error string；solver 会把它和原始计划放在同一上下文里看，并可以优雅降级。失败定位是 per-node，而不是 per-step。

### Planner distillation / Planner 蒸馏

论文第二个结果是：由于 planner 看不到 observations，你可以用 175B teacher 产出的 planner outputs 微调一个 7B 模型。小模型负责 planning；推理时不再需要大模型。这已经很常见：很多 2026 生产 Agent 会用小 planner + 大 executor，或反过来。

### Plan-and-Execute (LangChain, 2023) / Plan-and-Execute（LangChain，2023）

LangChain 团队在 2023 年 8 月的文章里把 ReWOO 泛化成一个模式名：Plan-and-Execute。先由 planner 生成 step list，executor 执行每一步，可选的 replanner 在观察部分结果后修订计划。它比 ReWOO 更接近 ReAct（replanner 把 observations 带回 planning），但保留了 token savings。

### Plan-and-Act (Erdogan et al., arXiv:2503.09572, ICML 2025) / Plan-and-Act

Plan-and-Act 把这个模式扩展到 long-horizon web 和 mobile agents。关键贡献是 synthetic plan data：一个带标签的 trajectory generator 产生显式 plan 的训练数据。它用于微调 planner model，让模型在 WebArena 类任务上超过 30-50 步后仍能保持一致性；单条 ReAct trajectory 在这种场景中很容易丢失 coherence。

### When to pick which / 什么时候选哪种

| Pattern | When |
|---------|------|
| ReAct | 短任务、未知环境、需要响应式异常处理 |
| ReWOO | 结构化任务、工具已知、token 敏感、evidence 可并行 |
| Plan-and-Execute | 类似 ReWOO，但要在部分执行后 replanning |
| Plan-and-Act | Long-horizon (>30 steps)，web / mobile / computer-use |
| Tree of Thoughts | Search 值得付费（Lesson 04） |

Anthropic 2024 年 12 月的建议是：从最简单的开始。如果任务只是一次工具调用加 summary，不要构建 ReWOO。如果任务是 40 步研究任务，不要只跑 ReAct。

## Build It / 动手构建

`code/main.py` 实现了一个 toy ReWOO：

- `Planner`：一个脚本化策略，从 prompt 生成 plan DAG。
- `Worker`：通过 registry 派发每个节点的 tool call。
- `Solver`：脚本化组合器，读取 evidence 并产出最终答案。
- Dependency resolution：把 `#E1` 这样的引用替换为更早 worker 的输出。

demo 会回答 “What is the population of the capital of France, rounded to millions?”，使用两步计划：(1) 查询首都，(2) 查询人口，然后 solve。

运行：

```
python3 code/main.py
```

trace 会先展示完整计划，再展示 worker results，最后展示 solver composition。把 token count（我们打印的是粗略字符数）和 ReAct-style interleaved run 对比，在这种结构化任务上 ReWOO 会赢。

## Use It / 应用它

LangGraph 把 Plan-and-Execute 作为 recipe 提供（`create_react_agent` 用于 ReAct，自定义 graphs 用于 plan-execute）。CrewAI 的 Flows 直接编码这个模式：你先定义 tasks，Flow DAG 执行它们。Plan-and-Act 的 synthetic data 方法目前仍偏研究；runtime pattern（显式 plan DAG）已经通过 LangGraph 和 CrewAI Flows 进入生产。

## Ship It / 交付它

`outputs/skill-rewoo-planner.md` 会在给定工具目录时，从用户请求生成一个 ReWOO plan DAG。它会在交给 executor 前验证计划：无环、每个引用都能解析、每个工具都存在。

## Exercises / 练习

1. 并行化彼此独立的 worker execution。对于一个 6-node DAG、2 个 parallel groups，它能带来什么收益？
2. 增加一个 replanner node，在任何 worker 返回 error 时触发。让 ReWOO 变成 Plan-and-Execute 的最小改动是什么？
3. 用小模型（7B class）替换 `Planner`，把 `Solver` 留给 frontier model。比较端到端质量：这种拆分在哪里会失败？
4. 阅读 ReWOO 论文 Section 4 关于 planner distillation 的内容。概念性复现 175B -> 7B：需要什么训练数据？如何评分 plan quality？
5. 把 toy 移植成 Plan-and-Act 的 trajectory shape：plan 是 sequence，不是 DAG。取舍会发生什么变化？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ReWOO | “Reasoning without observations” | 先 plan，再并行获取 evidence，最后 solve；planning prompt 中没有 observations |
| Plan-and-Execute | “LangChain 的 plan-execute pattern” | ReWOO 加上执行后的可选 replanner node |
| Plan-and-Act | “Scaled plan-execute” | 显式 planner / executor 拆分，并用 synthetic plan training data 处理 long-horizon tasks |
| Evidence reference | “#E1, #E2, ...” | 派发时替换为先前 worker 输出的 plan-node placeholder |
| Planner distillation | “Small planner, big executor” | 用大 teacher 的 planner traces 微调小模型 |
| Token efficiency | “Fewer round trips” | 论文中 HotpotQA 相对 ReAct 少 5 倍 token |
| DAG executor | “Topological dispatcher” | 按依赖顺序执行 plan nodes；每一层可并行 |

## Further Reading / 延伸阅读

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — 标准论文
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — 带 synthetic plans 的扩展 planner-executor
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — framework recipe
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 选择能工作的最简单模式
