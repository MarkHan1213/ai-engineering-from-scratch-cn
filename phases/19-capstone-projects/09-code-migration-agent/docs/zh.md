# Capstone 09 — Code Migration Agent (Repo-Level Language / Runtime Upgrade) / 代码迁移 Agent（仓库级语言 / 运行时升级）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 migrator 设定了 2026 年基准。Moderne 的 OpenRewrite 能在大规模场景中做确定性的 AST rewrites。Grit 用 codemod-style DSL 处理同类问题。生产模式会把两者结合：安全重写用 deterministic substrate，模糊案例用 Agent layer，每个分支在 sandbox 中 build，并在打开 PR 前让 test harness 变绿。本 capstone 要迁移 50 个真实 repo，并发布 pass rate 与 failure taxonomy。

**类型：** 综合项目
**语言：** Python（agent）, Java / Python（targets）, TypeScript（dashboard）
**前置知识：** 第 05 阶段（NLP）, 第 07 阶段（transformers）, 第 11 阶段（LLM engineering）, 第 13 阶段（tools）, 第 14 阶段（agents）, 第 15 阶段（autonomous）, 第 17 阶段（infrastructure）
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## Learning Objectives / 学习目标

- 把 repo-level migration 拆成 deterministic recipes 与 Agent 修复两层
- 在 Daytona sandbox 中运行 build、test、coverage gate 和预算控制
- 实现会分类失败、定向修改、重跑验证的 LangGraph Agent loop
- 构建 failure taxonomy dashboard，区分依赖、构建工具、语法、测试和预算问题
- 在 MigrationBench 子集上评估 pass rate、coverage delta、cost per repo 和 baseline 差异

## Problem / 问题

大规模代码迁移是 2026 年编码 Agent 最清晰的生产应用之一。ground truth 很明确（迁移后测试套件是否通过），收益很真实（Java-8 fleet migration 是 headcount-scale project），benchmark 也公开（MigrationBench 50-repo subset）。Moderne 的 OpenRewrite 处理确定性部分。Agent layer 处理所有 recipes 解决不了的内容：ambiguous rewrites、build-system drift、long-tail syntax、transitive dependency breakage。

你要构建一个 Agent，输入 Java 8 repo（或 Python 2 repo），输出 CI 变绿的 migrated branch。你会测量 pass rate、test-coverage preservation、cost per repo，并建立 failure taxonomy。与 deterministic-only baseline 的 side-by-side 会告诉你 Agent 的价值到底在哪里。

## Concept / 概念

pipeline 有两层。**deterministic substrate**（Java 用 OpenRewrite，Python 用 libcst）负责安全地完成大部分机械重写：imports、method signatures、null-safety edits、try-with-resources、deprecated API replacements。它快，并且 diff 可审计。**agent layer**（OpenAI Agents SDK 或 LangGraph，底层 Claude Opus 4.7 和 GPT-5.4-Codex）处理 recipes 无法覆盖的案例：build-file upgrades（Maven/Gradle/pyproject）、transitive dependency conflicts、test flakes、custom annotations。

每个 repo 都有一个预装目标 runtime 的 Daytona sandbox。Agent 迭代执行：run build、classify failures、apply fix、rerun。硬限制：每个 repo 30 分钟、$8、20 agent turns。如果全部测试通过且 coverage delta 非负，就打开 PR。否则，把 repo 归入 failure class 并附上证据。

failure taxonomy 是交付物。50 个 repo 里到底哪里坏了？Transitive deps？Custom annotations？Build tool version？与迁移无关的 test flakes？每个类别都要有 count 和 exemplar diff。未来 recipe authors 可以优先处理前三类。

## Architecture / 架构

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## Stack / 技术栈

- Deterministic substrate: OpenRewrite（Java）或 libcst（Python）
- Agent: OpenAI Agents SDK 或 LangGraph over Claude Opus 4.7 + GPT-5.4-Codex
- Sandbox: 每个分支一个 Daytona devcontainer，预装目标 runtime（Java 17 / Python 3.12）
- Build systems: Maven、Gradle、uv（Python）
- Benchmarks: Amazon MigrationBench 50-repo subset（Java 8 to 17）、Google App Engine Py2-to-Py3 repos
- Test harness: 并行 runner，Java coverage 用 Jacoco，Python coverage 用 coverage.py
- Observability: Langfuse + 每个 repo 的 trace bundle，包含每个 diff chunk
- Dashboard: failure-taxonomy dashboard，展示 per-class counts 和 exemplar diffs

## Build It / 动手构建

1. **Recipe pass.** 先运行 OpenRewrite（Java）或 libcst（Python）recipes。捕获 70-80% 的机械迁移。提交为 "recipe" commit。

2. **Build trial.** Daytona sandbox：安装目标 runtime，运行 build。如果 green，跳到 tests。如果 red，交给 Agent。

3. **Agent loop.** LangGraph tools: `run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`。Agent 分类失败（dep、syntax、test、build-tool），应用定向修复，然后重跑。

4. **Budget caps.** 每个 repo 30 分钟 wall-clock、$8 cost、20 agent turns。任何上限被触发，就停止并以 "budget_exhausted" 归档，同时保留当前 diff。

5. **Test + coverage gate.** build 变绿后运行测试套件。把 coverage 与 base repo 对比。如果 coverage 下降超过 2%，归入 "coverage_regression"。

6. **PR open.** 成功后 push branch，打开 PR，正文包含 diff、哪些 recipes 被应用、哪些 commits 由 Agent 生成。

7. **Failure taxonomy.** 对每个失败 repo 打标签：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建 dashboard。

8. **50-repo run.** 在 MigrationBench subset 上执行。报告 per-class pass rate、cost-per-repo、coverage-preservation，以及与 deterministic-only baseline 的对比。

## Use It / 应用它

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## Ship It / 交付它

`outputs/skill-migration-agent.md` 是交付物。给定一个 repo，它先执行 deterministic recipes，再运行 Agent loop，产出绿色 migrated branch；失败时把 repo 归入 taxonomy class。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | MigrationBench pass rate | 50-repo subset pass@1 |
| 20 | Test-coverage preservation | migrated branch 相对 base 的 mean coverage delta |
| 20 | Cost per migrated repo | passing runs 上的 $/repo |
| 20 | Agent / deterministic-tool integration | OpenRewrite 处理的 fixes 与 Agent authored fixes 的比例 |
| 15 | Failure analysis write-up | 带 exemplars 的 taxonomy completeness |
| **100** | | |

## Exercises / 练习

1. 只用 OpenRewrite（没有 Agent）运行 migrate pipeline。与完整 pipeline 比较 pass rate。识别哪些 case 只有 Agent 能补上差距。

2. 实现 "lint-clean" check：迁移后运行 style linter（Java 用 spotless，Python 用 ruff）。如果出现新 lint errors，则 PR 失败。测量 coverage preserved but style regressed 的比例。

3. 添加 "minimal-diff" optimizer：Agent branch 通过测试后，用第二轮裁掉不必要修改。报告 diff-size reduction。

4. 扩展到第三种迁移：Node 18 到 Node 22。复用 sandbox wrapping；把 recipe layer 换成自定义 codemod。

5. 把 time-to-first-green-build（TTFGB）作为 UX metric。目标：p50 低于 10 分钟。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Deterministic substrate | “Recipe engine” | OpenRewrite / libcst：带 safety guarantees 的 declarative AST rewrites |
| Codemod | “Code-modifying program” | 机械修改源代码的 rewrite rule |
| Build drift | “Tool version skew” | Maven / Gradle / uv 在 major versions 间的微妙行为变化 |
| Failure class | “Taxonomy bucket” | repo 未成功迁移的标签原因：dep、syntax、test、build-tool、budget |
| Coverage delta | “Coverage preservation” | base 到 migrated branch 的 test coverage % 变化 |
| Agent turn | “Tool-call round” | Agent loop 中一次 plan -> act -> observe cycle |
| Budget exhaustion | “Hit the ceiling” | repo 消耗完 30-min / $8 / 20-turn limit 但仍未通过 |

## Further Reading / 延伸阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — canonical 2026 benchmark
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — deterministic substrate reference
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe authoring
- [Grit.io](https://www.grit.io) — alternate codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK reference
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — alternate migration benchmark
- [libcst](https://github.com/Instagram/LibCST) — Python deterministic substrate
- [Daytona sandboxes](https://daytona.io) — per-branch sandbox reference
