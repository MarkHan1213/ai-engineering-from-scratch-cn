# Roots and Elicitation — Scoping and Mid-Flight User Input / Roots 与 Elicitation：作用域与执行中的用户输入

> 用户打开另一个项目时，hard-coded paths 立刻失效。用户表达不完整时，预填的 tool arguments 也会失效。Roots 把 server 限制到用户控制的一组 URIs；elicitation 则在 tool-call 中途暂停，通过 form 或 URL 向用户请求结构化输入。两个 client primitives，分别修复 MCP 中常见的两类失败。SEP-1036（URL-mode elicitation，2025-11-25）到 2026 H1 仍属 experimental，依赖前请检查 SDK versions。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, roots + elicitation demo)
**Prerequisites / 前置知识：** Phase 13 · 07 (MCP server)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 声明 `roots`，并响应 `notifications/roots/list_changed`。
- 把 server file operations 限制在 declared root set 内部的 URIs。
- 使用 `elicitation/create` 在 tool-call 中途向用户请求确认或结构化输入。
- 在 form-mode 和 URL-mode elicitation 之间做选择（后者仍 experimental；注意 drift-risk）。

## The Problem / 问题

notes MCP server 在生产里会遇到两个具体失败。

**Broken path assumption.** server 写死了 `~/notes`。用户在另一台机器上把 notes 放在 `~/Documents/Notes`，于是 tool call 要么悄悄失败（找不到文件），更糟时会写到错误位置。

**Missing argument the user would know.** 用户说 “delete the old TPS report note”。模型调用 `notes_delete(title: "TPS report")`，但有三条匹配 note，分别来自 2023、2024、2025。工具不能猜。返回 "ambiguous" 很烦；对三条都执行则是灾难。

Roots 修复第一个问题：client 在 `initialize` 声明 server 可访问的 URI 集合。Elicitation 修复第二个问题：server 暂停 tool call，发送 `elicitation/create` 请用户选择一个。

## The Concept / 概念

### Roots / Roots

client 在 `initialize` 中声明 root list：

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

server 随后可以调用 `roots/list`：

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

servers 必须把 roots 当作边界：root set 外的任何 file read 或 write 都要拒绝。这不是由 client 强制的（server 仍是用户信任的代码），但 spec-compliant servers 会遵守。

当用户添加或移除 root 时，client 发送 `notifications/roots/list_changed`。server 重新调用 `roots/list` 并更新边界。

### Why roots are a client primitive / 为什么 roots 是 client primitive

Roots 由 client 声明，因为它们代表用户的 consent model。用户告诉 Claude Desktop：“允许这个 notes server 访问这两个目录”。server 不能扩大这个 scope。

### Elicitation: the form-mode default / Elicitation：默认 form mode

`elicitation/create` 接收一个 form schema 和一句自然语言 prompt：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

client 渲染表单、收集用户回答，然后返回：

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

三个可能 action：`accept`（用户填写了）、`decline`（用户关闭了）、`cancel`（用户中止整个 tool call）。

Form schemas 是 flat 的，v1 不支持 nested objects。SDK 通常会拒绝超过一层的复杂 schema。

### Elicitation: URL mode (SEP-1036, experimental) / Elicitation：URL mode（SEP-1036，experimental）

2025-11-25 新增。server 发送 URL，而不是 schema：

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

client 在浏览器中打开 URL，等待完成，并在用户返回时给出 response。它适用于 OAuth flows、payment authorization 和 document signing 这类 form 不足以表达的场景。

Drift-risk note：SEP-1036 response shape 仍在收敛；有些 SDK 返回 callback URL，有些返回 completion token。生产使用 URL mode 前，请阅读你的 SDK release notes。

### When elicitation is the right tool / 何时适合使用 elicitation

- destructive actions 前的用户确认（destructive hint + elicitation）。
- Disambiguation（在 N 个匹配项中选一个）。
- First-run setup（API keys、directories、preferences）。
- OAuth-style flows（URL mode）。

### When elicitation is wrong / 何时不该使用 elicitation

- 填补模型本可用 prose 向用户追问的 required arguments。用普通 re-prompt，而不是 elicitation dialog。
- 高频调用。Elicitation 会打断 conversation；不要在 loop 里触发它。
- server 可以事后 validate 的内容。validate、返回 error，让模型用文本询问用户。

### Human-in-the-loop bridge / Human-in-the-loop 桥接

Elicitation 加 sampling 共同构成 MCP 的 "human-in-the-loop" 模型。server 的 agent loop 可以为用户输入（elicitation）或模型推理（sampling）暂停。Phase 13 · 11 讲了 sampling；本课讲 elicitation。把两者组合起来，就得到完整 mid-loop control。

## Build It / 动手构建

本课会扩展 notes server：从 client 读取 roots，所有文件操作前检查 URI 是否在 scope 内；当 `notes_delete` 匹配到多个候选时，通过 `elicitation/create` 请求用户选择；再用 URL-mode 模拟 first-run setup。

## Use It / 应用它

`code/main.py` 为 notes server 增加：

- `roots/list` response，server 会在 root-list-changed notifications 后重新查询。
- 一个 `notes_delete` tool，当多个 notes 匹配时用 `elicitation/create` 消歧。
- 一个 `notes_setup` tool，使用 URL-mode elicitation 打开模拟的 first-run config page。
- 一个 boundary check，拒绝 declared roots 外的 URI operations。

demo 运行三个场景：happy path（单个匹配）、disambiguation（三个匹配，触发 elicitation）、out-of-root-write（被拒绝）。

## Ship It / 交付它

本课产出 `outputs/skill-elicitation-form-designer.md`。给定一个可能需要用户确认或消歧的 tool，这个 skill 会设计 elicitation form schema 和 message template。

## Exercises / 练习

1. 运行 `code/main.py`。触发 disambiguation path；确认 simulated user answer 被路由回 tool。

2. 添加一个新工具 `notes_archive`，每次都需要 elicitation confirmation（destructive hint）。检查 UX：它和模型用文本重新询问相比如何？

3. 为 first-run OAuth flow 实现 URL-mode elicitation。标出 drift risk，并加入 SDK-version guard。

4. 扩展 `roots/list` handling：收到 notification 后，server 应该 atomically 重新读取，并 rescan 可能已经 out of scope 的 open file handles。

5. 阅读 GitHub 上 SEP-1036 issue discussion thread。找出一个会影响 server 如何处理 URL-mode callbacks 的 open question。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Root | “Consent boundary” | client 允许 server 访问的 URI |
| `roots/list` | “Server asks for scope” | client 返回当前 root set |
| `notifications/roots/list_changed` | “User changed scope” | client 通知 root set 已变化 |
| Elicitation | “Ask the user mid-call” | server 发起的结构化用户输入请求 |
| `elicitation/create` | “The method” | elicitation requests 的 JSON-RPC method |
| Form mode | “Schema-driven form” | 在 client UI 中渲染为表单的 flat JSON Schema |
| URL mode | “Browser redirect” | SEP-1036 experimental；打开 URL 并等待 |
| `accept` / `decline` / `cancel` | “User response outcomes” | server 需要处理的三个分支 |
| Disambiguation | “Pick one” | 当 tool 有 N 个候选项时的常见 elicitation 用例 |
| Flat form | “Top-level properties only” | elicitation schemas 不能嵌套 |

## Further Reading / 延伸阅读

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — canonical roots reference
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — canonical elicitation reference
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 additions walkthrough
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL-mode elicitation proposal（experimental, drift-risk）
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX walkthrough
