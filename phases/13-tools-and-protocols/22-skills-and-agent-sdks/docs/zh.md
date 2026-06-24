# Skills and Agent SDKs — Anthropic Skills, AGENTS.md, OpenAI Apps SDK / Skills 与 Agent SDK：Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说明“有哪些工具”。Skills 说明“如何完成一个任务”。2026 年的 stack 会把两者叠起来。Anthropic Agent Skills（open standard，2025 年 12 月）以 SKILL.md 交付，并支持 progressive disclosure。OpenAI Apps SDK 是 MCP 加 widget metadata。AGENTS.md（现在存在于 60,000+ repos）位于 repo root，作为 project-level agent context。本课会说明三者各自覆盖什么，并构建一个能跨 agents 迁移的最小 SKILL.md + AGENTS.md bundle。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, SKILL.md parser and loader)
**Prerequisites / 前置知识：** Phase 13 · 07 (MCP server)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 区分三层：AGENTS.md（project context）、SKILL.md（reusable know-how）、MCP（tools）。
- 编写带 YAML frontmatter 和 progressive disclosure 的 SKILL.md。
- 以 filesystem-style 把 skills 加载进 agent runtime。
- 把 skill 与 MCP server、AGENTS.md 组合起来，让一个 package 可在 Claude Code、Cursor 和 Codex 中工作。

## The Problem / 问题

一名工程师把 release-notes-writing workflow 提炼成 multi-step prompt：“读取最新 merged PRs。按 area 分组。总结每个。按团队风格写 changelog entry。发到 Slack draft。”他们把它放进团队 Notion doc。

现在他们想在 Claude Code、Cursor 和 Codex CLI 中使用这个 workflow。每个 agent 加载 instructions 的方式都不同：Claude Code slash-commands、Cursor rules、Codex `.codex.md`。于是工程师复制了三份 workflow，并维护三份。

AGENTS.md 和 SKILL.md 一起解决这个问题：

- **AGENTS.md** 放在 repo root。每个兼容 agent 在 session start 读取它。“这个项目如何工作？有什么约定？哪些命令运行测试？”
- **SKILL.md** 是 portable bundle：YAML frontmatter（name、description）+ markdown body + optional resources。支持 skills 的 agents 会按 name on demand 加载。
- **MCP**（Phase 13 · 06-14）处理 skill 需要调用的工具。

三层，一个 portable artifact。

## The Concept / 概念

### AGENTS.md (agents.md) / AGENTS.md

2025 年末推出，到 2026 年 4 月已有 60,000+ repos 采用。repo root 一个文件。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

agents 在 session start 读取它，并据此校准自己在该项目中的行为。2026 年每个 coding agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md format / SKILL.md 格式

Anthropic Agent Skills（2025 年 12 月作为 open standard 发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明 skill identity。body 是 skill 加载时展示给模型的 prompt。

### Progressive disclosure / 渐进披露

Skills 可以引用 sub-resources，agent 只在需要时 fetch。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 写着 “see style-guide.md for the style rules.” agent 只有在 skill 正在运行时才拉取 style-guide.md。这样不会把模型可能不需要的细节塞进 prompt。

### Filesystem discovery / 文件系统发现

agent runtimes 会扫描已知目录中的 SKILL.md files：

- `~/.anthropic/skills/*/SKILL.md`
- Project `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载时按 folder name 和 frontmatter `name` 索引。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（cross-agent）都遵循这种模式。

### Anthropic Claude Agent SDK / Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）会在 session start 加载 skills，并把它们作为 runtime 内可调用的 “agents” 暴露。用户调用 skill 时，agent loop 会 dispatch 到该 skill。

### OpenAI Apps SDK / OpenAI Apps SDK

2025 年 10 月发布，直接构建在 MCP 上。它把 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到一个 developer surface。Apps SDK app 是：

- 一个 MCP server（tools、resources、prompts）。
- 加上 ChatGPT UI 的 widget metadata。
- 加上可选的 MCP Apps `ui://` resource，用于 interactive surfaces。

同一个协议，更丰富 UX。

### Cross-agent portability via SkillKit / 通过 SkillKit 跨 agent 迁移

SkillKit 和类似 cross-agent distribution layers 可以把单个 SKILL.md 翻译成 32+ AI agents（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的 native format。一个 source of truth，多个 consumers。

### The three-layer stack / 三层栈

| Layer | File | Loaded when | Purpose |
|-------|------|-------------|---------|
| AGENTS.md | repo root | session start | project-level conventions |
| SKILL.md | skills directory | skill invoked | reusable workflow |
| MCP server | external process | tools needed | callable actions |

三层可以组合：agent 在 session start 读取 AGENTS.md，用户调用 skill，skill instructions 包含 MCP tool calls，agent 通过 MCP client dispatch。

## Build It / 动手构建

本课会实现一个 stdlib SKILL.md parser 和 loader：扫描 `./skills/`，解析 YAML frontmatter 与 markdown body，并模拟 agent loop 按 name 调用 skill。重点是 portable package 的形状，而不是某个特定 agent 的专有格式。

## Use It / 应用它

`code/main.py` 提供一个 stdlib SKILL.md parser 和 loader。它发现 `./skills/` 下的 skills，解析 YAML frontmatter 加 markdown body，并产出按 skill name keyed 的 dict。随后它模拟 agent loop，通过 name 调用 `release-notes-writer`。

重点看：

- YAML frontmatter 用最小 stdlib parser 解析（没有 `pyyaml` dependency）。
- Skill body 原样保存；agent 在 invocation 时把它 prepend 到 system prompt。
- progressive disclosure 通过 `read_subresource` function 演示：只在需要时拉取 referenced files。

## Ship It / 交付它

本课产出 `outputs/skill-agent-bundle.md`。给定一个 workflow，这个 skill 会产出 combined SKILL.md + AGENTS.md + MCP-server-blueprint bundle，可跨 agents 迁移。

## Exercises / 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个 skill，确认 loader 能捡到。

2. 为这个课程 repo 写一个 AGENTS.md。包含 testing commands、style conventions 和 Phase 13 mental model。

3. 把团队内部文档中的 multi-step workflow 移植为 SKILL.md。验证它能在 Claude Code 加载。

4. 手工把该 skill 翻译成 Cursor 和 Codex 的 native rule formats。统计格式间差异；这就是 SkillKit 自动化的 translation surface。

5. 阅读 Anthropic Agent Skills blog post。找出 Claude Agent SDK 中一个本课 loader 没覆盖的 feature。（提示：agent sub-invocation。）

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| SKILL.md | “skill file” | agent runtime 加载的 YAML frontmatter + markdown body |
| AGENTS.md | “Repo-root agent context” | session start 时读取的 project-level conventions file |
| Progressive disclosure | “Lazy-load sub-resources” | skill body 引用只在需要时拉取的 files |
| Frontmatter | “YAML block at top” | `---` delimiters 中的 metadata（name、description） |
| Claude Agent SDK | “Anthropic's skill runtime” | `@anthropic-ai/claude-agent-sdk`，加载 skills 并 route |
| OpenAI Apps SDK | “MCP + widget meta” | 基于 MCP 加 ChatGPT UI hooks 的 OpenAI dev surface |
| Skill discovery | “Filesystem scan” | 遍历 known dirs 寻找 SKILL.md，并按 name 建索引 |
| Cross-agent portability | “One skill many agents” | 通过 SkillKit-style tools 把一个 SKILL.md 翻译到 32+ agents |
| Agent Skill | “Portable know-how” | MCP tool concept 之外的 reusable task template |
| Apps SDK | “MCP plus ChatGPT UI” | Connectors 和 Custom GPTs 在 MCP 上统一 |

## Further Reading / 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月 launch
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md format reference
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 面向 ChatGPT 的 MCP-based developer platform
- [agents.md](https://agents.md/) — AGENTS.md format 和 adoption list
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — official skill examples
