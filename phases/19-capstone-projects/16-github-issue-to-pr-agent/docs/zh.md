# Capstone 16 — GitHub Issue-to-PR Autonomous Agent / GitHub Issue 到 PR 的自主 Agent

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都在交付同一种 2026 产品形态：给 issue 打 label，然后得到 PR。Agent 在 cloud sandbox 中运行，验证测试通过，并带着 rationale 发布 review-ready PR。难点在于自动复现 repo 的 build environment、防止 credential leakage、执行 per-repo budgets，以及确保 Agent 不能 force-push。本 capstone 构建 self-hosted 版本，并在 cost 和 pass rate 上与 hosted alternatives 对比。

**类型：** 综合项目
**语言：** Python（agent）, TypeScript（GitHub App）, YAML（Actions）
**前置知识：** 第 11 阶段（LLM engineering）, 第 13 阶段（tools）, 第 14 阶段（agents）, 第 15 阶段（autonomous）, 第 17 阶段（infrastructure）
**Phases exercised:** P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## Learning Objectives / 学习目标

- 构建以 GitHub label / PR comment 为触发器的 async cloud coding Agent
- 实现 GitHub App、webhook dispatcher、cloud worker、sandbox 和 issue-to-PR loop
- 设计 environment inference、full CI verification、coverage delta gate 和 budget enforcement
- 用 GitHub App permissions、branch protection 和 worker allow-list 限制安全边界
- 与 hosted alternatives 对比 pass rate、cost、latency 和 PR quality

## Problem / 问题

async cloud coding agent 是与 interactive coding agent（capstone 01）不同的产品类别。UX 是一个 GitHub label。你给 issue 打上 `@agent fix this`，worker 在 cloud sandbox 中启动，clone repo，运行 tests，编辑文件，验证，然后打开一个 PR，并在正文里写清 Agent rationale。没有交互 loop，没有 terminal。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules、Factory Droids 都在收敛到这个形态。

工程挑战很具体：environment reproduction（Agent 必须在没有缓存 dev image 的情况下从零构建 repo）、flaky tests（必须重跑或隔离）、credential scoping（使用最小 fine-grained permissions 的 GitHub App）、per repo per day 的 budget enforcement，以及 no-force-push policy。本 capstone 测量相对 hosted alternatives 的 pass rate、cost 和 safety。

## Concept / 概念

触发器是 GitHub webhook（issue label 或 PR comment）。dispatcher 把工作入队到 ECS Fargate 或 Lambda。worker 把 repo 拉进 Daytona 或 E2B sandbox，并根据 repo 推断通用 Dockerfile（language、framework）。Agent 运行 mini-swe-agent 或 SWE-agent v2 loop，底层用 Claude Opus 4.7 或 GPT-5.4-Codex。它迭代：读代码、提出修复、应用 patch、运行 tests。

Verification 是 gate。PR 打开前，完整 CI 必须在 sandbox 中通过。coverage delta 会被计算；如果负向超过阈值，PR 仍可打开，但会加 `needs-review` label。Agent 把 rationale 放进 PR description，并开启一个 reviewer 可以用 `@agent` ping 的 thread 做 follow-ups。

Safety 通过两个 GitHub surface 限定：App 提供短期 installation token，包含 `workflows: read` 和窄 repo contents/PR scopes；branch protection（不是 app permissions）执行 “no direct writes to `main`” 和 “no force-push”，并且 app 永远不加入 bypass list。GitHub App 没有真正的 path-scoped read-only access 到 `.github/workflows`，所以 Agent 的 file edit allow-list 必须在 worker 上强制执行。per repo per day budget ceilings 在 dispatcher 执行（例如每个 repo 每天最多 5 个 PR、每个 PR $20）。

## Architecture / 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## Stack / 技术栈

- Trigger: GitHub App with fine-grained token；webhook receiver 使用 Lambda 或 Fly.io
- Worker: ECS Fargate task（或 GitHub Actions self-hosted runner）
- Sandbox: 每个任务一个 Daytona devcontainer 或 E2B sandbox
- Agent loop: mini-swe-agent baseline 或 SWE-agent v2 over Claude Opus 4.7 / GPT-5.4-Codex
- Retrieval: tree-sitter repo-map + ripgrep
- Verification: sandbox 内 full CI + coverage delta gate
- Observability: Langfuse，PR body 中链接 per-PR trace archive
- Budget: per-repo daily dollar ceiling；每个 repo 每天最多 PR 数

## Build It / 动手构建

1. **GitHub App.** Fine-grained installation token：issues read+write、pull_requests write、contents read+write、workflows read。Branch protection（唯一能做到这件事的 surface）强制 “no direct push to `main`” 和 “no force-push”；app 不在 bypass list 中。由于 GitHub App permissions 不支持 path-scoped，worker 需要在 proposed diff 上用 allow-list 强制 “no writes under `.github/workflows`”。

2. **Webhook receiver.** Lambda function 接收 issue label / PR comment webhooks。按 label `@agent fix this` 过滤。入队到 SQS。

3. **Dispatcher.** 从 SQS 弹出 tasks。执行 per-repo per-day budget。用 repo URL、issue body 和 fresh Daytona sandbox 启动 ECS Fargate task。

4. **Environment inference.** 检测 language（Python、Node、Go、Rust）和 package manager（uv、pnpm、go mod、cargo）。如果 repo 没有 Dockerfile，就动态生成一个。

5. **Agent loop.** mini-swe-agent 或 SWE-agent v2 with Claude Opus 4.7。Tools: ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬上限：$20 cost、30 min wall-clock、30 agent turns。

6. **Verification.** loop 结束后，在 sandbox 中运行完整测试套件。通过 jacoco / coverage.py 计算 coverage delta。如果 CI red：停止，不打开 PR。如果 coverage 下降超过 2%：打开 PR，并加 `needs-review` label。

7. **PR posting.** Push agent branch。通过 GitHub API 打开 PR，包含 title、rationale、diff summary、trace URL、cost、turns。

8. **Credential hygiene.** Worker 使用短期 GitHub App installation token。归档前清理 logs 中的 secrets。

9. **Eval.** 30 个不同难度的 seeded internal issues。测量 pass rate、PR quality（diff size、style、coverage）、cost、latency。与 Cursor Background Agents 和 AWS Remote SWE Agents 在同一 issues 上对比。

## Use It / 应用它

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## Ship It / 交付它

`outputs/skill-issue-to-pr.md` 是交付物。它包含 GitHub App + async cloud worker，把被标记的 issues 转成 review-ready PRs，并带 bounded cost 和 scoped credentials。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Pass rate on 30 issues | End-to-end success（CI green + coverage OK） |
| 20 | PR quality | Diff size、coverage delta、style conformance |
| 20 | Cost and latency per resolved issue | 每个 PR 的 $ 和 wall-clock |
| 20 | Safety | Scoped token、per-repo budget、no force-push、credential hygiene |
| 15 | Operator UX | Rationale comments、retry affordance、@-mention follow-up |
| **100** | | |

## Exercises / 练习

1. 添加 “fix flaky test” mode：label `@agent stabilize-flake TestX` 在 sandbox 中把该测试跑 50 次，并提出最小稳定化改动。

2. 在三个 shared issues 上与 Cursor Background Agents 比较 cost。报告哪些工具在哪些场景获胜。

3. 实现 budget dashboard：per-repo per-day cost、per-user cost。异常时 alert。

4. 构建 “dry-run” mode：不运行 CI，只打开 draft PR，让 reviewer 低成本检查 plan。

5. 添加 retention policy：超过 7 天未 merge 的 PR branches 自动删除。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | “Scoped bot identity” | 具备 fine-grained permissions 和短期 installation token 的 App |
| Async cloud agent | “Background agent” | 在 cloud sandbox 中运行的非交互 worker，不是 terminal |
| Environment inference | “Dockerfile synthesis” | 检测 language + package manager，并在缺失时生成 Dockerfile |
| Verification | “CI-in-sandbox” | 打开 PR 前在 worker 内运行完整测试套件 |
| Coverage delta | “Coverage preservation” | base 到 agent branch 的 test coverage % 变化 |
| Per-repo budget | “Daily ceiling” | dispatcher 执行的 dollar 和 PR-count cap |
| Rationale | “PR body explanation” | Agent 在 PR body 中写明改了什么和为什么；必需 |

## Further Reading / 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — canonical async cloud agent reference
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI reference
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — commercial alternative
- [OpenAI Codex (cloud)](https://openai.com/codex) — hosted competitor
- [Google Jules](https://jules.google) — Google hosted version
- [Factory Droids](https://www.factory.ai) — alternate commercial reference
- [GitHub App documentation](https://docs.github.com/en/apps) — scoped bot identity
- [Daytona cloud sandboxes](https://daytona.io) — reference sandbox
