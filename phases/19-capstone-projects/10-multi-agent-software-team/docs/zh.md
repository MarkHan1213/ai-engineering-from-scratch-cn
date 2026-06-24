# Capstone 10 — Multi-Agent Software Engineering Team / 多 Agent 软件工程团队

> SWE-AF 的 factory architecture、MetaGPT 的 role-based prompting、AutoGen 0.4 的 typed actor graph、Cognition 的 Devin，以及 Factory 的 Droids，都在 2026 年收敛到同一种形态：architect 规划，N 个 coders 在并行 worktrees 中工作，reviewer gate，tester 验证。Parallel worktrees 把 wall-clock 转成 throughput。Shared state 和 handoff protocols 成为主要失败面。本 capstone 要构建这个团队，在 SWE-bench Pro 上评估，并报告哪些 handoffs 会坏、坏得多频繁。

**类型：** 综合项目
**语言：** Python / TypeScript（agents）, Shell（worktree scripts）
**前置知识：** 第 11 阶段（LLM engineering）, 第 13 阶段（tools）, 第 14 阶段（agents）, 第 15 阶段（autonomous）, 第 16 阶段（multi-agent）, 第 17 阶段（infrastructure）
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17
**时间：** 40 小时

## Learning Objectives / 学习目标

- 设计 role-typed multi-agent coding team，并明确 Architect、Coder、Reviewer、Tester 的契约
- 用 task board 和 A2A-typed messages 管理跨角色 handoff
- 通过 `git worktree` + Daytona sandbox 支撑并行实现与隔离验证
- 记录 handoff accounting 和 token amplification，并与 single-agent baseline 对比
- 在 SWE-bench Pro 上评估 pass@1、parallel speedup、review quality 和 coordination failures

## Problem / 问题

单 Agent coding harness 在大型任务上会碰到天花板。不是因为某个 Agent 弱，而是 200k-token context 放不下 architecture plan、四个并行 codebase slices、reviewer commentary 和 test output。Multi-agent factories 把问题拆开：architect 负责 plan，coders 在 parallel worktrees 中负责 implementation，reviewer gate，tester verify。SWE-AF 的 “factory” architecture、MetaGPT 的 roles、AutoGen 的 typed actor graph，本质上都描述同一个形态。

失败面在 handoff。Architect 计划了 coders 无法实现的内容。Coders 产出冲突 diffs。Reviewer 批准了幻觉修复。Tester 与仍在写代码的 coder 发生 race。你要构建一个这样的团队，跑 50 个 SWE-bench Pro issues，追踪每个 handoff，并发布 post-mortem。

## Concept / 概念

Roles 是 typed agents。**Architect**（Claude Opus 4.7）读取 issue，写 plan，并把它拆成有明确 interfaces 的 subtasks。**Coders**（Claude Sonnet 4.7，N 个并行实例，每个运行在 `git worktree` + Daytona sandbox 中）独立实现 subtasks。**Reviewer**（GPT-5.4）读取 merged diff，批准或请求具体修改。**Tester**（Gemini 2.5 Pro）在隔离环境中运行 test suite，并带 artifacts 报告 pass/fail。

通信通过共享 task board（file-backed 或 Redis）。每个 role 只消费自己被允许处理的 tasks。handoffs 是 A2A-protocol-typed messages。协调关注点包括：merge-conflict resolution（coordinator role 或 automatic three-way merge）、shared-state synchronization（coders 开始后 plan 冻结；replans 是单独事件）、reviewer gatekeeping（reviewer 不能批准自己写的改动或自己建议的改动）。

Token amplification 是隐藏成本。每个 role boundary 都会增加 summary prompts 和 handoff context。一个 40-turn single-agent run 可能变成四个角色合计 160 turns。rubric 明确衡量相对 single-agent baseline 的 token efficiency，因为真正的问题不是 “multi-agent 是否可行”，而是 “它是否按美元也更划算”。

## Architecture / 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## Stack / 技术栈

- Orchestration: LangGraph with shared state + per-agent sub-graphs
- Messaging: A2A protocol（Google 2025），用于 typed inter-agent messages
- Models: Opus 4.7（architect）、Sonnet 4.7（coders）、GPT-5.4（reviewer）、Gemini 2.5 Pro（tester）
- Worktree isolation: 每个 coder 使用 `git worktree add` + Daytona sandbox
- Merge coordinator: 自定义 three-way merge + LLM-mediated conflict resolution
- Eval: SWE-bench Pro（50 issues）、SWE-AF scenarios、HumanEval++ for unit tests
- Observability: Langfuse with role-tagged spans、per-agent token accounting
- Deployment: K8s，每个 role 是独立 Deployment，HPA 基于 backlog

## Build It / 动手构建

1. **Task board.** File-backed JSONL，包含 typed messages：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。Agents 按 tags 订阅。

2. **Architect.** 读取 GitHub issue，用 Opus 4.7 和 plan template 生成 plan，要求明确 subtask interfaces（files touched、public functions、test impact）。输出一个包含 subtasks DAG 的 `plan_request`。

3. **Coders.** N 个 parallel workers，每个从 board claim 一个 subtask。每个 worker 启动新的 `git worktree add` branch 和 Daytona sandbox。实现 subtask。输出 `diff_ready`，包含 patch + test deltas。

4. **Merge coordinator.** 所有 coders 完成后，把 N 个 branches three-way merge 到 staging branch。只有出现 file-level overlap 时，才使用 LLM-mediated conflict resolution。

5. **Reviewer.** GPT-5.4 读取 merged diff。不能批准自己 authored 的 diffs。输出 `approved`（no-op）或 `review_feedback`，其中具体 change requests 会路由回相关 coder。

6. **Tester.** Gemini 2.5 Pro 在干净 sandbox 中运行 test suite。捕获 artifacts。输出 `test_passed` 或带 stacktraces 的 `test_failed`。失败测试循环回拥有该 subtask 的 coder。

7. **Handoff accounting.** 每条跨 role boundary 的 message 都在 Langfuse 中写 span，记录 payload size 和 model used。计算 per-subtask token amplification（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **Eval.** 在 50 个 SWE-bench Pro issues 上运行。与 single-agent baseline（单个 Sonnet 4.7、单 worktree）比较 pass@1 和 $-per-solved-issue。

9. **Post-mortem.** 对每个 failed issue，识别坏掉的 handoff（plan too vague、merge conflict、reviewer false-approve、tester flake）。生成 handoff-failure histogram。

## Use It / 应用它

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## Ship It / 交付它

`outputs/skill-multi-agent-team.md` 是交付物。给定 issue URL 和 parallelism level，团队会产出 merge-ready PR，并附带 per-role token accounting。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | Matched 50-issue subset, pass@1 |
| 20 | Parallel speedup | 相对 single-agent baseline 的 wall-clock |
| 20 | Review quality | injected-bug probe 上的 false-approval rate |
| 20 | Token efficiency | 相对 single-agent 的 total tokens per solved issue |
| 15 | Coordination engineering | Merge-conflict resolution、handoff-failure histogram |
| **100** | | |

## Exercises / 练习

1. 在运行中向 diff 注入明显 bug（主逻辑前多一个 `return None`）。测量 reviewer 的 false-approve rate。调 reviewer prompt，直到 false-approval 低于 5%。

2. 缩减到两个 coders（architect + coder + reviewer + tester，coder 顺序处理两个 subtasks）。比较 wall-clock 和 pass rate。

3. 用 single-writer constraint 替换 merge coordinator（subtasks 触碰互不相交的 file sets）。测量 architect 的 planning burden。

4. 把 reviewer 从 GPT-5.4 换成 Claude Opus 4.7。测量 false-approval rate 和 token cost delta。

5. 添加第五个角色：documenter（Haiku 4.5）。review 后它生成 changelog entry。测量文档质量是否值得额外 token spend。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Parallel worktree | “Isolated branch” | `git worktree add` 为每个 coder 生成全新的 working tree |
| Task board | “Shared message bus” | Agents 订阅的 typed messages 存储，可用 file 或 Redis |
| Handoff | “Role boundary” | 从一个 role 的 context 跨到另一个 role 的任何 message |
| Token amplification | “Multi-agent overhead” | 同一任务中跨 roles 的 total tokens / single-agent tokens |
| A2A protocol | “Agent-to-agent” | Google 2025 typed inter-agent messages spec |
| Merge coordinator | “Integrator” | 运行 three-way merge 并调解 conflicts 的组件 |
| False approval | “Reviewer hallucination” | Reviewer 批准了带已知 bug 的 diff |

## Further Reading / 延伸阅读

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 2026 multi-agent factory reference
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — role-based multi-agent framework
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft typed actor framework
- [Cognition AI (Devin)](https://cognition.ai) — reference product
- [Factory Droids](https://www.factory.ai) — alternate reference product
- [Google A2A protocol](https://developers.google.com/agent-to-agent) — inter-agent messaging spec
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — isolation substrate
- [SWE-bench Pro](https://www.swebench.com) — evaluation target
