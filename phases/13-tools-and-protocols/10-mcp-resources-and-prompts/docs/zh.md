# MCP Resources and Prompts — Context Exposure Beyond Tools / MCP Resources 与 Prompts：工具之外的上下文暴露

> Tools 拿走了 MCP 90% 的注意力。另外两个 server primitives 解决的是不同问题。Resources 暴露可读数据；prompts 把可复用模板暴露成 slash-commands。很多 server 应该用 resources，而不是把读取包装成 tools；也应该用 prompts，而不是把 workflow 硬编码进 client prompts。本课会给出 decision rule，并走过 `resources/*` 与 `prompts/*` messages。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, resource + prompt handler)
**Prerequisites / 前置知识：** Phase 13 · 07 (MCP server)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 针对一个 domain，判断 capability 应该暴露为 tool、resource 还是 prompt。
- 实现 `resources/list`、`resources/read`、`resources/subscribe`，并处理 `notifications/resources/updated`。
- 用 argument templates 实现 `prompts/list` 和 `prompts/get`。
- 识别 host 什么时候把 prompts 呈现为 slash-commands，什么时候自动注入 context。

## The Problem / 问题

一个天真的 notes app MCP server 会把所有东西都暴露成 tools：`notes_read`、`notes_list`、`notes_search`。这会把每次数据访问都包装成由模型驱动的 tool call。后果：

- 每个可能受益于 context 的 query，模型都必须决定是否调用 `notes_read`。
- 只读内容无法被订阅，也无法流到 host 的 side panel。
- Client UIs（Claude Desktop 的 resource attachment panel、Cursor 的 "Include file" picker）无法展示这些数据。

正确拆分是：数据暴露为 resource，会修改或计算的动作暴露为 tool，可复用多步骤 workflow 暴露为 prompt。每个 primitive 都有自己的 UX affordance 和 access pattern。

## The Concept / 概念

### Tools vs resources vs prompts — the decision rule / Tools、resources、prompts 的决策规则

| Capability | Primitive |
|------------|-----------|
| User wants to search, filter, or transform data | tool |
| User wants the host to include this data as context | resource |
| User wants a templated workflow they can re-run | prompt |

准则：如果模型会在每个相关查询中受益于调用它，它就是 tool。如果用户会受益于把它附加到 conversation，它就是 resource。如果用户想复用的是一个完整 multi-step workflow，它就是 prompt。

### Resources / 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}`，返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址内容：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（custom scheme）
- `memory://session-2026-04-22/recent`（server-specific）

`contents[]` 同时支持 text 和 binary。Binary 使用 base64-encoded string 的 `blob`，再加 `mimeType`。

### Resource subscriptions / Resource 订阅

在 capabilities 中声明 `{resources: {subscribe: true}}`。Client 调用 `resources/subscribe {uri}`。resource 变化时，server 发送 `notifications/resources/updated {uri}`。Client 再重新读取。

用例：notes server 的 resources 是磁盘文件；file watcher 触发 update notifications；Claude Desktop 在文件被 host 外部修改时重新拉取进 context。

### Resource templates (2025-11-25 addition) / Resource templates（2025-11-25 新增）

`resourceTemplates` 允许暴露 parameterized URI pattern：`notes://{id}`，其中 `id` 是 completion target。client 可以在 resource picker 中自动补全 ids。

### Prompts / 提示模板

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}`，返回 `{description, messages: [{role, content}]}`。

prompt 是一个会填充成 message list 的模板，host 会把这组 messages 送给模型。例如，`code_review` prompt 接收 `file_path` argument，返回一个三消息序列：system message、带 file body 的 user message，以及带 reasoning template 的 assistant kickoff。

### Hosts and prompts / Host 如何呈现 prompts

Claude Desktop、VS Code 和 Cursor 会把 prompts 暴露为 chat UI 中的 slash-commands。用户输入 `/code_review`，再从表单中选择 arguments。server 的 prompt 是“用户 shortcut”和“发给模型的完整 prompt”之间的契约。

不是每个 client 都支持 prompts，必须检查 capability negotiation。server 即使声明了 prompt capability，client 如果不支持 prompts，用户也不会在界面里看到这些 slash commands。

### The "list changed" notification / “list changed” notification

resources 和 prompts 都会在集合变化时发出 `notifications/list_changed`。一个刚导入 20 条 notes 的 notes server 会发出 `notifications/resources/list_changed`；client 随后重新调用 `resources/list` 来拿到新增项。

### Content type conventions / Content type 约定

文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制：`image/png`、`application/pdf`，再加 `blob` field。
MCP Apps（Lesson 14）：`ui://` URI 中的 `text/html;profile=mcp-app`。

### Dynamic resources / 动态资源

resource URI 不一定对应静态文件。`notes://recent` 可以每次读取时返回最新五条 notes。`db://query/users/active` 可以执行 parameterized query。server 可以自由动态计算 content。

规则：如果 client 能按 URI cache，那么 URI 必须稳定。如果计算是 one-shot，URI 应该包含 timestamp 或 nonce，避免 client cache stale out。

### Subscriptions vs polling / 订阅 vs 轮询

支持 subscription 的 clients 通过 `notifications/resources/updated` 获得 server push。pre-subscription clients 或不支持它的 hosts 通过重新读取来 polling。两者都符合 spec。server 的 capability declaration 会告诉 client 它支持哪一种。

subscriptions 的成本：server 需要维护 per-session state（谁订阅了什么）。保持 subscribed set 有界；断开的 clients 应该超时。

### Prompts vs system prompts / Prompts 与 system prompts

MCP prompts 不是 system prompts。host 的 system prompt（自己的运行指令）和 MCP prompts（用户调用的 server-supplied templates）并存。行为良好的 client 不会让 server prompt 覆盖自己的 system prompt；它会进行 layering。

## Build It / 动手构建

本课会把 Lesson 07 的 notes server 扩展为一个更完整的 MCP server：notes 作为 resources，读写动作保留为 tools，`review_note` 作为 prompt。你会看到同一份 domain capability 该如何按 UX 和访问模式拆成三个 primitive。

## Use It / 应用它

`code/main.py` 在 Lesson 07 的 notes server 基础上增加：

- per-note resources（`notes://note-1` 等），支持 `resources/subscribe`。
- 一个 `review_note` prompt，可渲染成三消息模板。
- 一个 file-watcher simulation，在 note 修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` dynamic resource，始终返回最新五条 notes。

运行 demo，观察完整 flow。

## Ship It / 交付它

本课产出 `outputs/skill-primitive-splitter.md`。给定一个拟议 MCP server，这个 skill 会把每项 capability 分类为 tool / resource / prompt，并给出 rationale。

## Exercises / 练习

1. 运行 `code/main.py`。观察初始 resource list，然后触发 note edit，验证 `notifications/resources/updated` event 会发出。

2. 添加一个 `resources/list_changed` emitter：创建新 note 时发送 notification，让 clients 重新 discover。

3. 为 GitHub MCP server 设计三个 prompts：`summarize_pr`、`triage_issue`、`release_notes`。每个都带 argument schemas。prompt body 应该无需进一步编辑即可运行。

4. 选择 Lesson 07 server 中的一个已有 tool，判断它应该继续作为 tool，还是拆成 resource + tool pair。用一句话说明理由。

5. 阅读 spec 的 `server/resources` 和 `server/prompts` sections。找出 `resources/read` 中一个很少填充、但 spec 支持的字段。提示：看 resource content 上的 `_meta`。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Resource | “Exposed data” | host 可以读取的 URI-addressable content |
| Resource URI | “Pointer to data” | 带 scheme 的 identifier（`file://`、`notes://` 等） |
| `resources/subscribe` | “Watch for changes” | client opt-in 的 server-push updates，针对特定 URI |
| `notifications/resources/updated` | “Resource changed” | 告诉 client 某个 subscribed resource 有新内容 |
| Resource template | “Parameterized URI” | 带 completion hints 的 URI pattern，供 host picker 使用 |
| Prompt | “Slash-command template” | 带 argument slots 的 named multi-message template |
| Prompt arguments | “Template inputs” | host 在渲染前收集的 typed parameters |
| `prompts/get` | “Render template” | server 返回填充后的 message list |
| Content block | “Typed chunk” | `{type: text \| image \| resource \| ui_resource}` |
| Slash-command UX | “User shortcut” | host 把 prompts 呈现为以 `/` 开头的 commands |

## Further Reading / 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — resource URIs、subscriptions 和 templates
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — prompt templates 与 slash-command integration
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整 `resources/*` message reference
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整 `prompts/*` message reference
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 在官方文档基础上扩展的 community guide
