# Capstone — Build a Complete Tool Ecosystem / Capstone：构建完整工具生态系统

> Phase 13 已经教完每个部件。本 capstone 会把它们接成一个 production-shaped system：带 tools + resources + prompts + tasks + UI 的 MCP server、边界上的 OAuth 2.1、RBAC gateway、multi-server client、一次 A2A sub-agent call、进入 collector 的 OTel tracing、CI 中的 tool-poisoning detection，以及一个 AGENTS.md + SKILL.md bundle。结束后，你应该能为每个架构选择辩护。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, end-to-end ecosystem harness)
**Prerequisites / 前置知识：** Phase 13 · 01 through 22
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 组合一个 MCP server，暴露 tools、resources、prompts，以及带 `ui://` app 的 task。
- 用 OAuth 2.1 gateway 前置该 server，并强制 RBAC 与 pinned hashes。
- 编写 multi-server client，用 OTel GenAI attributes 做端到端 tracing。
- 把一部分 workload 委派给 A2A sub-agent，并验证 opacity 被保留。
- 用 AGENTS.md + SKILL.md 打包整个 stack，让其他 agents 也能驱动它。

## The Problem / 问题

交付 "research and report" system：

- 用户问：“summarize the three most-cited 2026 arXiv papers on agent protocols.”
- 系统：通过 MCP 搜索 arXiv；通过 A2A 把 paper summarization 委派给 specialized writer agent；聚合结果；把交互式 report 渲染为 MCP Apps `ui://` resource；把每一步记录到 OTel。

Phase 13 中的所有 primitives 都会出现。这不是 toy：Anthropic（Claude Research product）、OpenAI（GPTs with Apps SDK）和第三方在 2026 年交付的生产 research-assistant systems 都是这个形状。

## The Concept / 概念

### Architecture / 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### Trace hierarchy / Trace 层级

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

一个 trace id。每个 span 都带正确的 `gen_ai.*` attributes。

### Security posture / 安全姿态

- OAuth 2.1 + PKCE，并用 resource indicator 把 audience pin 到 gateway。
- gateway 持有 upstream credentials；用户永远看不到。
- RBAC：`alice` 有 `research:read`、`research:write`，可调用所有 tools。`bob` 只有 `research:read`，不能调用 `generate_report`。
- pinned description manifest：任何 tool hash 改变的 server 都会被丢弃。
- Rule of Two audit：没有工具同时组合 untrusted input、sensitive data 和 consequential action。

### Rendering / 渲染

最终 `generate_report` task 返回 content blocks 和一个 `ui://report/current` resource。client 的 host（Claude Desktop 等）在 sandbox iframe 中渲染 interactive dashboard。dashboard 包含排序后的 paper list、citation counts，以及一个按钮；用户点击任意 paper 时，它会调用 `host.callTool('summarize_paper', {arxiv_id})`。

### Packaging / 打包

整体以如下结构交付：

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

用户用 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` skill 驱动系统。

### What each Phase 13 lesson contributed / Phase 13 每课贡献了什么

| Lesson | What the capstone uses |
|--------|------------------------|
| 01-05 | Tool interface, provider-portability, parallel calls, schemas, linting |
| 06-10 | MCP primitives, server, client, transports, resources + prompts |
| 11-14 | Sampling, roots + elicitation, async tasks, `ui://` apps |
| 15-18 | Tool poisoning, OAuth 2.1, gateway + registry, production auth |
| 19 | A2A sub-agent delegation |
| 20 | OTel GenAI tracing |
| 21 | Routing gateway for the LLM layer |
| 22 | SKILL.md + AGENTS.md packaging |

## Build It / 动手构建

本课会把前面各课的 harness 串成一个 in-process demo：gateway handshake、auth simulation、tool discovery、task lifecycle、A2A delegation、`ui://` resource return、OTel spans 和 packaging artifacts。构建重点是边界之间的组合，而不是某个 primitive 的细节实现。

## Use It / 应用它

`code/main.py` 把前面课程的模式缝成一个可运行 demo。全部 stdlib、全部 in-process，便于从头读到尾。它为 research-and-report scenario 跑完整流程：handshake with gateway、OAuth 2.1 simulated、tools/list merged、`generate_report` as a task、A2A call to writer、ui:// resource returned、OTel spans emitted。

重点看：

- 每一跳共享同一个 trace id。
- gateway policy 会阻止第二个用户写入。
- Task lifecycle 从 working → completed，并同时返回 text 和 ui:// content。
- A2A call 的 internal state 对 orchestrator 保持 opaque。
- AGENTS.md 和 SKILL.md 是另一个 agent 复现 workflow 所需的唯一文件。

## Ship It / 交付它

本课产出 `outputs/skill-ecosystem-blueprint.md`。给定一个 product need（research、summarization、automation），这个 skill 会产出完整 architecture：使用哪些 MCP primitives、哪些 gateway controls、哪些 A2A calls、哪些 telemetry、哪些 packaging。

## Exercises / 练习

1. 运行 `code/main.py`。观察 single trace id 以及 spans 如何嵌套。数出 demo 触达了 Phase 13 中多少 primitives。

2. 扩展 demo：添加第二个 backend MCP server（例如 `bibliography`），确认 gateway 会把它的 tools 合并进同一个 namespace。

3. 用运行在 subprocess 上的真实 A2A writer agent 替换 fake A2A writer agent。使用 Lesson 19 harness。

4. 在 orchestrator 和 LLM 之间的 routing gateway 中添加 PII redaction step。确认 user query 中的 emails 被 scrub。

5. 为会维护这个系统的 teammate 写一份 AGENTS.md。阅读时间应低于五分钟，并给出他们在 Cursor 或 Codex 中驱动 capstone 所需的一切。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Capstone | “Phase-13 integration demo” | 使用每个 primitive 的 end-to-end system |
| Research and report | “The scenario” | search、summarize、render pattern |
| Ecosystem | “All the pieces together” | server + client + gateway + sub-agent + telemetry + package |
| Trace hierarchy | “Single trace id” | 每一跳的 span 共享 trace；parent-child 通过 span ids 表示 |
| Gateway-issued token | “Transitive auth” | client 只看到 gateway token；gateway 持有 upstream creds |
| Merged namespace | “All tools in one flat list” | gateway 上的 multi-server merge，collision 时 prefix |
| Opacity boundary | “A2A call hides internals” | sub-agent 的 reasoning 对 orchestrator 不可见 |
| Three-layer stack | “AGENTS.md + SKILL.md + MCP” | project context + workflow + tools |
| Defense-in-depth | “Multiple security layers” | pinned hashes、OAuth、RBAC、Rule of Two、audit log |
| Spec compliance matrix | “What we ship that the spec requires” | deliverables 到 2025-11-25 requirements 的 checklist mapping |

## Further Reading / 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — consolidated reference
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — protocol 走向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 reference
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — canonical tracing conventions
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — production agent runtime patterns
