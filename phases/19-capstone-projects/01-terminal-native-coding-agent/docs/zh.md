# Capstone 01 — Terminal-Native Coding Agent / 终端原生编码 Agent

> 到 2026 年，编码 Agent 的形态已经基本定型：TUI harness、有状态计划、沙箱化工具面，以及 plan、act、observe、recover 的循环。Claude Code、Cursor 3 和 OpenCode 从远处看都是同一类系统。本 capstone 要求你端到端构建一个这样的系统：从 CLI 输入到 pull request 输出，并在 SWE-bench Pro 上与 mini-swe-agent、Live-SWE-agent 对比。你会学到，难点不在一次模型调用，而在工具循环、沙箱，以及 50 轮运行里的成本上限。

**类型：** 综合项目
**语言：** TypeScript / Bun（harness）, Python（eval scripts）
**前置知识：** 第 11 阶段（LLM engineering）, 第 13 阶段（tools and protocols）, 第 14 阶段（agents）, 第 15 阶段（autonomous systems）, 第 17 阶段（infrastructure）
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间：** 35 小时

## Learning Objectives / 学习目标

- 设计一个清晰分层的 plan / act / observe / recover 编码 Agent harness
- 实现沙箱化工具调用、生命周期 hooks、成本预算和上下文压缩
- 用 MCP StreamableHTTP 暴露文件、搜索、运行和 git 工具面
- 在 SWE-bench Pro 子集上评估 pass@1、turns-per-task 和 $-per-task
- 交付一个能从任务描述生成 PR，并附带 trace bundle 的终端原生 Agent

## Problem / 问题

到 2026 年，编码 Agent 已经成为最主流的 AI 应用类型之一。Claude Code (Anthropic)、带 Composer 2 和 Agent Tabs 的 Cursor 3、Amp (Sourcegraph)、OpenCode（112k stars）、Factory Droids、Google Jules，都在发布同一种架构的变体：终端 harness、带权限的工具面、沙箱，以及围绕 frontier model 构建的 plan-act-observe loop。前沿成绩很窄，例如 Live-SWE-agent 用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%，但工程细节非常宽。多数失败并不是模型不会推理，而是工具循环不稳定、上下文被污染、token 成本失控，或文件系统操作具有破坏性。

你无法只从外部评估这种 Agent。你必须亲手构建一个，看它在第 47 轮因为 ripgrep 返回 8MB 匹配结果而崩掉，然后重新设计截断层。这正是本 capstone 的意义。

## Concept / 概念

harness 有四个核心表面。**Plan** 维护 TodoWrite 风格的状态对象，模型每一轮都会整体重写它。**Act** 分发工具调用（read、edit、run、search、git）。**Observe** 捕获 stdout / stderr / exit codes，截断后把摘要送回上下文。**Recover** 处理工具错误，避免撑爆上下文窗口或陷入无限循环。2026 年的形态还多了一个关键面：**hooks**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact` 是可配置扩展点，operator 可以在这里注入策略、遥测和 guardrails。

沙箱通常是 E2B 或 Daytona。每个任务在全新的 devcontainer 中运行，挂载一个可读写的 git worktree。harness 永远不直接触碰宿主文件系统。任务成功或失败后，worktree 都会被清理。成本控制分三层执行：每轮 token 上限、每个 session 的美元预算，以及硬性的轮数限制（通常是 50）。可观测性层使用带 GenAI semantic conventions 的 OpenTelemetry spans，并写入自托管 Langfuse。

## Architecture / 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## Stack / 技术栈

- Harness runtime: Bun 1.2 + Ink 5（React-in-terminal）
- Model access: OpenRouter unified API，接入 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（最难任务）
- Tool transport: Model Context Protocol StreamableHTTP（MCP 2026 revision）
- Sandbox: E2B sandboxes（JS SDK）或 Daytona devcontainers
- Code search: ripgrep subprocess，面向 17 种语言的预编译 tree-sitter parsers
- Isolation: 每个任务使用 `git worktree add`，成功 / 失败后清理
- Eval harness: SWE-bench Pro（verified subset）+ Terminal-Bench 2.0 + 你自己的 30-task holdout
- Observability: OpenTelemetry SDK with `gen_ai.*` semconv → self-hosted Langfuse
- PR posting: GitHub App，使用细粒度 token，scope 限制在目标 repo

## Build It / 动手构建

1. **TUI and command loop.** 用 Ink 搭一个 Bun 项目。接受 `agent run <repo> "<task>"`。输出三段式视图：plan pane（顶部）、tool-call stream（中部）、token budget（底部）。Ctrl-C 取消时，退出前触发 `SessionEnd` hook。

2. **Plan state.** 定义 typed TodoWrite schema（pending / in_progress / done items with notes）。模型每轮通过工具调用重写完整状态，不要让它增量 mutate。把 plan 持久化到 `.agent/state.json`，这样崩溃后可以恢复。

3. **Tool surface.** 定义六个工具：`read_file`、`edit_file`（带 diff preview）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带 timeout）、`git`（status / diff / commit / push）。通过 MCP StreamableHTTP 暴露，让 harness 与传输层解耦。每个工具返回截断后的输出（每次调用上限 4k tokens）。

4. **Sandbox wrapping.** 每个任务启动一个 E2B sandbox。用 `git worktree add -b agent/$TASK_ID` 创建新分支。所有工具调用都在 sandbox 内执行，宿主文件系统不可达。

5. **Hooks.** 实现 2026 年的八类 hook。至少接入四个用户自定义 hook：(a) `PreToolUse` destructive-command guard，阻止 worktree 外的 `rm -rf`；(b) `PostToolUse` token accounting；(c) `SessionStart` budget initialization；(d) `Stop` 写出 final trace bundle。

6. **Eval loop.** 克隆 SWE-bench Pro Python 的 30-issue 子集。让你的 harness 逐个运行。与 mini-swe-agent（最小 baseline）比较 pass@1、turns-per-task 和 $-per-task。把结果写入 `eval/results.jsonl`。

7. **Cost control.** 硬上限：50 turns、200k context、每任务 $5。`PreCompact` hook 在 150k mark 把旧轮次总结成 prior-state block，为新观察结果腾空间，同时不丢失计划。

8. **PR posting.** 成功时，最后一步是 `git push` + GitHub API 调用，用 plan 和 diff summary 作为正文打开 PR。

## Use It / 应用它

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## Ship It / 交付它

交付物 skill 位于 `outputs/skill-terminal-coding-agent.md`。给定 repo path 和 task description，它会在 sandbox 中运行完整的 plan-act-observe loop，并返回 PR URL 与 trace bundle。本 capstone 的评分标准：

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs baseline | 你的 harness 与 mini-swe-agent 在 30 个匹配 Python 任务上的对比 |
| 20 | Architecture clarity | Plan/act/observe separation、hook surface、tool schema，并按 Live-SWE-agent layout 审查 |
| 20 | Safety | Sandbox escape tests、permission prompts、destructive-command guard 通过 red-team |
| 20 | Observability | Trace completeness（100% tool calls 有 span）、每轮 token accounting |
| 15 | Developer UX | Cold-start < 2s，crash recovery 可恢复 plan，Ctrl-C 能干净取消 mid-tool |
| **100** | | |

## Exercises / 练习

1. 把 backing model 从 Claude Sonnet 4.7 换成运行在 vLLM 上的 Qwen3-Coder-30B。比较 pass@1 和 $-per-task。报告 open model 表现不佳的位置。

2. 添加一个 `reviewer` sub-agent，在 PR posting 前读取 diff，并可以请求 revision loop。测量 false-positive review 是否会把 SWE-bench pass rate 拉低到 single-agent baseline 以下（提示：通常会）。

3. 压测 sandbox：写一个尝试 `curl` 外部 URL 的任务，再写一个尝试写出 worktree 的任务。确认两者都被 PreToolUse hook 阻止，并记录这些尝试。

4. 用更小的模型（Haiku 4.5）实现 `PreCompact` summarization。测量 3x compaction 时 plan fidelity 损失多少。

5. 把 MCP StreamableHTTP transport 换成 stdio。benchmark cold-start 和 per-call latency，为 local-only 使用场景选出赢家。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Harness | “The agent loop” | 包围模型的代码：分发工具、维护 plan state、执行预算约束 |
| Hook | “Agent event listener” | 由 harness 在八类生命周期事件之一上运行的用户自定义脚本 |
| Worktree | “Git sandbox” | 位于独立路径的 linked git checkout，可丢弃且不影响主 clone |
| TodoWrite | “Plan state” | 模型每轮重写的 pending / in-progress / done typed list |
| StreamableHTTP | “MCP transport” | 2026 MCP revision：长连接 HTTP + 双向流式传输；替代 SSE |
| Token ceiling | “Context budget” | 每轮或每 session 的 input+output token 上限；触发 compaction 或 termination |
| pass@1 | “Single-attempt pass rate” | SWE-bench 任务首次运行即解决的比例，不重试、不窥探测试集 |

## Further Reading / 延伸阅读

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的 reference harness
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench harness 对比用的最小 baseline
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 达到 79.2% SWE-bench Verified
- [OpenCode](https://opencode.ai) — 开放 harness，112k stars
- [SWE-bench Pro leaderboard](https://www.swebench.com) — 本 capstone 目标评测
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、capability metadata
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — tool calls 和 token usage 的 span schema
