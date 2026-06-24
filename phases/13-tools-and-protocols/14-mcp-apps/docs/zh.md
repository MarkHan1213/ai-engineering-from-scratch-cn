# MCP Apps — Interactive UI Resources via `ui://` / MCP Apps：通过 `ui://` 提供交互式 UI 资源

> 纯文本 tool output 限制了 agent 能展示什么。MCP Apps（SEP-1724，2026 年 1 月 26 日正式发布）允许工具返回 sandboxed interactive HTML，并内联渲染在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中。Dashboards、forms、maps、3D scenes，都通过同一个扩展实现。本课会走过 `ui://` resource scheme、`text/html;profile=mcp-app` MIME、iframe-sandbox postMessage protocol，以及允许 server 渲染 HTML 带来的安全面。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, UI resource emitter), HTML (sample app)
**Prerequisites / 前置知识：** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 从 tool call 返回 `ui://` resource，并设置正确 MIME 和 metadata。
- 用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联 UI。
- 实现 iframe sandbox postMessage JSON-RPC，让 UI 与 host 通信。
- 应用 CSP 和 permissions-policy defaults，防御 UI-originated attacks。

## The Problem / 问题

2025 年的 `visualize_timeline` tool 只能返回 “Here are 14 notes organized chronologically: ...”。这只是一段文字。用户真正想要的是可交互 timeline。MCP Apps 之前，选项是 client-specific widget APIs（Claude artifacts、OpenAI Custom GPT HTML），或者完全没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日发布）标准化了这份契约。tool result 包含一个 `resource`，其 URI 为 `ui://...`，MIME 为 `text/html;profile=mcp-app`。host 在带 limited CSP 的 sandboxed iframe 中渲染它；除非显式授予，否则没有 network access。iframe 内部 UI 通过一个很小的 postMessage JSON-RPC dialect 向 host 发消息。

所有兼容 client（Claude Desktop、ChatGPT、Goose、VS Code）都会用同样方式渲染同一个 `ui://` resource。一个 server、一个 HTML bundle、通用 UI。

## The Concept / 概念

### The `ui://` resource scheme / `ui://` resource scheme

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

host 随后对 `ui://notes/timeline` URI 调用 `resources/read`，拿到：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe sandbox / Iframe 沙箱

host 会在 sandboxed `<iframe>` 中渲染 HTML，带有：

- `sandbox="allow-scripts allow-same-origin"`（或按 server declaration 更严格）。
- 通过 response headers 应用 server-declared CSP。
- 没有来自 host origin 的 cookies 或 localStorage。
- network access 限于 CSP 中的 `connectSrc`。

### postMessage protocol / postMessage 协议

iframe 通过 `window.postMessage` 与 host 通信。一个很小的 JSON-RPC 2.0 dialect：

Always pin `targetOrigin` to the peer's exact origin, and on the receiving side validate `event.origin` against an allowlist before processing any payload. Never use `"*"` for either side of this channel — the body carries tool calls and resource reads.

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI 可调用的 host-side methods：

- `host.callTool(name, arguments)` — 调用 server tool。
- `host.readResource(uri)` — 读取 MCP resource。
- `host.getPrompt(name, arguments)` — 获取 prompt template。
- `host.close()` — 关闭 UI。

每次调用仍然经过 MCP protocol，并继承 server 的 permissions。

### Permissions / 权限

`_meta.ui.permissions` list 请求额外能力：

- `camera` — 访问用户摄像头（用于 scan-a-document UIs）。
- `microphone` — 语音输入。
- `geolocation` — 位置。
- `network:*` — 比单独 `connectSrc` 更宽的网络访问。

每个 permission 都会成为用户在 UI 渲染前看到的 prompt。

### Security risks / 安全风险

iframe 中的 HTML 仍然是 HTML。新增攻击面：

- **Prompt-injection via UI.** 恶意 server UI 可以显示看起来像 system message 的文本来欺骗用户。host rendering 应明显区分 server UI 与 host UI。
- **Exfiltration via `connectSrc`.** 如果 CSP 允许 `connect-src: *`，UI 可以把数据发到任何地方。默认应严格。
- **Clickjacking.** UI 覆盖 host chrome。Hosts 必须阻止 z-index manipulation，并强制 opacity rules。
- **Steal focus.** UI 获取键盘焦点并捕获下一条消息。Hosts 必须拦截。

Phase 13 · 15 会在 MCP security 中深入这些问题；本课先引入它们。

### `ui/initialize` handshake / `ui/initialize` 握手

iframe 加载后，会通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

host 返回 capabilities 和 session token。UI 在之后每次 host call 中使用 session token。

### AppRenderer / AppFrame SDK primitives / AppRenderer / AppFrame SDK 原语

ext-apps SDK 暴露两个 convenience primitives：

- `AppRenderer`（server side）— 包装 React / Vue / Solid component，并发出带正确 MIME 和 metadata 的 `ui://` resource。
- `AppFrame`（client side）— 接收 resource、挂载 iframe，并调停 postMessage。

你可以使用它们，也可以手写 HTML 和 JSON-RPC。

### Ecosystem status / 生态状态

MCP Apps 于 2026 年 1 月 26 日发布。2026 年 4 月的 client support：

- **Claude Desktop.** 2026 年 1 月起 full support。
- **ChatGPT.** 通过 Apps SDK full support（底层是同一 MCP Apps protocol）。
- **Cursor.** Beta；通过 settings 启用。
- **VS Code.** 仅 Insider builds。
- **Goose.** Full support。
- **Zed, Windsurf.** Roadmapped。

生产中的 servers：dashboards、map visualizations、data tables、chart builders、sandbox IDE previews。

## Build It / 动手构建

本课会让 notes server 的 `visualize_timeline` tool 返回一个 `ui://notes/timeline` resource，并让 `resources/read` 返回一个完整 HTML bundle。示例不需要构建系统，重点是 MIME、`_meta.ui`、CSP、permissions 和 postMessage contract。

## Use It / 应用它

`code/main.py` 扩展 notes server，加入一个 `visualize_timeline` tool：它返回 `ui://notes/timeline` resource，并为该 URI 提供 `resources/read` handler，返回一个小而完整的 HTML bundle，里面包含 SVG timeline。HTML 用 stdlib template 生成，不需要 build system。由于 stdlib 不能驱动浏览器，postMessage 在 JS 注释中示意。

重点看：

- tool response 上的 `_meta.ui` 携带 resourceUri、CSP、permissions。
- HTML 不需要 network access；所有 data 都 inline。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（在这个 stdlib demo 中是 documented but inert）。

## Ship It / 交付它

本课产出 `outputs/skill-mcp-apps-spec.md`。给定一个适合 interactive UI 的 tool，这个 skill 会产出完整 MCP Apps contract：`ui://` URI、CSP、permissions、postMessage entrypoints 和 security checklist。

## Exercises / 练习

1. 运行 `code/main.py` 并检查输出的 HTML。直接在浏览器中打开 HTML，验证 SVG 会渲染。然后草拟 UI 用于调用 `host.callTool("notes_update", ...)` 的 postMessage contract。

2. 收紧 CSP：移除 `'unsafe-inline'`，改用 nonce-based script policy。HTML generation code 需要做什么变化？

3. 添加第二个 UI resource `ui://notes/editor`，用表单原地编辑 note。用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的 attack surface。恶意 server 可能在哪里注入内容？iframe sandbox 能防御什么，不能防御什么？

5. 阅读 SEP-1724 spec，找出 MCP Apps SDK 中这个 toy implementation 没使用的一个 capability。（提示：component-level state sync。）

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MCP Apps | “Interactive UI resources” | 2026-01-26 发布的 SEP-1724 extension |
| `ui://` | “App URI scheme” | UI bundles 的 resource scheme |
| `text/html;profile=mcp-app` | “The MIME” | MCP App HTML 的 Content-type |
| Iframe sandbox | “Render container” | 通过 CSP 和 permissions 对 UI 做 browser sandboxing |
| postMessage JSON-RPC | “UI-to-host wire” | 用于 host calls 的 tiny JSON-RPC-over-postMessage dialect |
| `_meta.ui` | “Tool-UI binding” | 把 tool result 关联到 UI resource 的 metadata |
| CSP | “Content-Security-Policy” | 声明 scripts、network、styles 的 allowed sources |
| AppRenderer | “Server SDK primitive” | 把 framework component 转成 `ui://` resource |
| AppFrame | “Client SDK primitive” | iframe mount helper，负责调停 postMessage |
| `ui/initialize` | “Handshake” | UI 发给 host 的第一条 postMessage |

## Further Reading / 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — reference implementation 和 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — formal spec document
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — high-level documentation
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月 launch post
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc-style SDK reference
