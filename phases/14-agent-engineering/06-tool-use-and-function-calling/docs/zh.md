# Tool Use and Function Calling / 工具使用与函数调用

> Toolformer (Schick et al., 2023) 开启了 self-supervised tool annotation。Berkeley Function Calling Leaderboard V4 (Patil et al., 2025) 则定义了 2026 年的门槛：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination。单轮调用已经基本解决；memory、dynamic decision-making 和 long-horizon tool chains 还没有。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 Toolformer 的 self-supervised training signal：只有当执行工具能降低 next-token loss 时，才保留该工具标注。
- 说出 BFCL V4 的五个评测类别，以及每个类别衡量什么。
- 实现一个 stdlib tool registry，包含 schema validation、argument coercion 和 execution sandboxing。
- 诊断 2026 年三个未解决问题：long-horizon tool chaining、dynamic decision-making 和 memory。

## The Problem / 问题

早期工具使用问的是：模型能不能预测正确的 function call？现代工具使用问的是：模型能不能跨 40 步串联工具，带着 memory，在部分可观测环境中，从工具失败里恢复，并且不幻觉并不存在的工具？

Toolformer 建立了基线：模型可以通过 self-supervision 学会何时调用工具。BFCL V4 定义了 2026 年的评测目标。两者之间的差距，就是生产 Agent 真实工作的空间。

## The Concept / 概念

### Toolformer (Schick et al., NeurIPS 2023) / Toolformer

核心想法：让模型给自己的预训练语料标注候选 API calls。对每个候选，实际执行它。只有当加入工具结果能降低下一个 token 的 loss 时，才保留这条标注。然后在过滤后的语料上微调。

覆盖的工具包括 calculator、QA system、search engines、translator、calendar。self-supervision 信号纯粹来自“工具是否帮助预测文本”，不需要人工标签。

规模结果是：tool use 随规模涌现。小模型会被 tool annotations 伤害，大模型会受益。这解释了为什么 2026 年 frontier models 内建了强工具能力，而多数 7B 模型仍需要显式 tool-use fine-tuning 才可靠。

### Berkeley Function Calling Leaderboard V4 (Patil et al., ICML 2025) / BFCL V4

BFCL 是 2026 年事实上的 function calling 评测。V4 构成：

- **Agentic (40%)**：完整 Agent trajectories，包括 memory、multi-turn、dynamic decisions。
- **Multi-Turn (30%)**：带工具链的交互式对话。
- **Live (10%)**：用户提交的真实 prompts（分布更难）。
- **Non-Live (10%)**：synthetic test cases。
- **Hallucination (10%)**：判断不应该调用工具的场景。

V3 引入了 state-based evaluation：工具序列执行后检查 API 的真实状态（例如“文件是否创建成功？”），而不是匹配 tool calls 的 AST。V4 增加了 web search、memory 和 format sensitivity 类别。

2026 年关键发现是：single-turn function calling 已接近解决。失败集中在 memory（跨 turn 携带上下文）、dynamic decision-making（基于先前结果选择工具）、long-horizon chains（20+ 步后漂移）和 hallucination detection（没有合适工具时拒绝调用）。

### Tool schema / 工具 schema

每个 provider 都有 schema。细节不同，但形状相同：

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者都接受 JSON Schema。Descriptions 是承重件，模型会读它来选择工具。错误工具选择的第一根因，通常就是糟糕的 tool descriptions。

### Argument validation / 参数校验

不要信任任何 tool call。需要校验：

1. **Type coercion。** 模型可能返回字符串 `"5"`，但 schema 要求 int。若无歧义可以转换，否则拒绝。
2. **Enum validation。** 如果 schema 规定 `status in {"open", "closed"}`，模型却输出 `"in_progress"`，要带描述性错误拒绝。
3. **Required fields。** 缺失 required field -> 立即把 error observation 回传给模型，而不是 crash。
4. **Format validation。** Dates、emails、URLs 要用具体 parser 校验，不要用 regex 糊弄。

每个 validation failure 都应该返回结构化 observation，让模型能用正确形状 retry。

### Parallel tool calls / 并行工具调用

现代 provider 支持一个 assistant turn 中的 parallel tool calls。循环形态：

1. 模型发出 3 个带不同 `tool_use_id` 的 tool calls。
2. runtime 执行它们（独立时可并行）。
3. 每个结果带着对应 `tool_use_id` 作为 `tool_result` block 回传。

工程规则：correlation IDs 是承重件。把它们搞混，就会把错误工具结果路由给错误 tool call。

### Sandboxing / 沙箱

工具执行就是 sandbox boundary。详见 Lesson 09。简短规则是：每个工具都应该声明 read/write surface、network access、timeout、memory cap。通用 `run_shell(cmd)` 是红旗；具体 `git_status()` 更安全。

```figure
tool-routing
```

## Build It / 动手构建

`code/main.py` 实现了一个生产形态的 tool registry：

- JSON Schema subset validator（只用 stdlib）。
- 带 description、input schema、timeout 和 executor 的 tool registration。
- Argument coercion 和 enum validation。
- 带 correlation IDs 的 parallel tool dispatch。
- 作为结构化字符串返回的 error observations。

运行：

```
python3 code/main.py
```

trace 会展示一个 mini agent 在一轮中调用三个工具，其中一个故意 malformed call 会被描述性错误拒绝，模型可以据此修正。

## Use It / 应用它

每家 provider 都有自己的 tool schema：Anthropic、OpenAI、Gemini、Bedrock。如果需要 multi-provider，使用 translation layer（OpenAI Agents SDK、Vercel AI SDK、LangChain tool adapter）。BFCL 是参考 benchmark；如果工具使用是产品核心，上线前要用它跑你的 Agent。

## Ship It / 交付它

`outputs/skill-tool-registry.md` 会为给定任务域生成 tool catalog、schema 和 registry，并包含 description-quality checks：每个工具的 description 是否告诉模型什么时候该用它？

## Exercises / 练习

1. 添加一个 “no-op” 工具，让模型能显式拒绝使用其他工具。在类似 BFCL 的 hallucination test 上测量。
2. 实现 int-as-string 和 float-as-string 的 argument coercion。coercion 从哪里开始会隐藏真实 bug？
3. 增加 per-tool timeout 和 circuit breaker（连续 3 次失败后 60 秒内拒绝该工具）。这会如何改变模型恢复方式？
4. 阅读 BFCL V4 描述。选择一个类别（例如 “multi-turn”），让你的 Agent 跑 10 个示例 prompts。报告 pass rate。
5. 把 stdlib validator 移植到 Pydantic 或 Zod。Pydantic / Zod 抓到了 toy 没抓到的什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Function calling | “Tool use” | 带校验 schema 的结构化 tool invocation |
| Toolformer | “Self-supervised tool annotation” | Schick 2023：只保留能降低 next-token loss 的 tool calls |
| BFCL | “Berkeley Function Calling Leaderboard” | 2026 benchmark：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination |
| Tool schema | “Function signature for the model” | name、description、JSON Schema arguments |
| tool_use_id | “Correlation ID” | 把 tool call 和 result 对上；parallel dispatch 必不可少 |
| Hallucination detection | “Know when not to call” | V4 类别：没有合适工具时拒绝调用 |
| Argument coercion | “String-to-int repair” | 对可预期 schema mismatch 做窄修复；有歧义就拒绝 |
| Sandboxing | “Tool execution boundary” | 每工具的 read/write surface、network、timeout、memory cap |

## Further Reading / 延伸阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — self-supervised tool annotation
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026 eval benchmark
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Agent SDK 中的生产 tool schema
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — function tool type 和 Guardrails
