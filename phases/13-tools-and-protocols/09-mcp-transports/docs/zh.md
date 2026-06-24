# MCP Transports — stdio vs Streamable HTTP vs SSE Migration / MCP 传输：stdio、Streamable HTTP 与 SSE 迁移

> stdio 只适合本机，离开本机就不成立。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE transport 已 deprecated，并将在 2026 年中移除。选错 transport 会带来迁移成本；选对 transport，则能得到可远程托管、具备 session continuity 和 DNS-rebinding protection 的 MCP server。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, Streamable HTTP endpoint skeleton)
**Prerequisites / 前置知识：** Phase 13 · 07, 08 (MCP server and client)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 按部署形态（local vs remote、single-process vs fleet）在 stdio 和 Streamable HTTP 之间选择。
- 实现 Streamable HTTP single-endpoint pattern：POST 用于 requests，GET 用于 session stream。
- 强制 `Origin` validation 和 session-id semantics，以抵御 DNS-rebinding。
- 在 2026 年中移除 deadline 前，将 legacy HTTP+SSE server 迁移到 Streamable HTTP。

## The Problem / 问题

第一个 MCP remote transport（2024-11）是 HTTP+SSE：两个 endpoints，一个接 client 的 POST，另一个 Server-Sent-Events channel 负责 server-to-client stream。它能工作，也很笨重：每个 session 两个 endpoints，一些 CDN 前面的 cache 会坏掉，并且强依赖 long-lived SSE connections，而某些 WAF 会积极终止这类连接。

2025-03-26 spec 用 Streamable HTTP 替换了它：一个 endpoint，POST 处理 client requests，GET 建立 session stream，两者共享 `Mcp-Session-Id` header。从那之后新建或迁移的 server 都使用 Streamable HTTP。旧 SSE mode 正在 deprecated：Atlassian Rovo 于 2026 年 6 月 30 日移除；Keboola 于 2026 年 4 月 1 日移除；其余多数 enterprise servers 到 2026 年底完成。

stdio 仍然对 local servers 很重要。Claude Desktop、VS Code 和所有 IDE-shaped client 都通过 stdio 启动 servers。正确 mental model 是：stdio 用于“这台机器”，Streamable HTTP 用于“网络上”。不要交叉使用。

## The Concept / 概念

### stdio / 标准输入输出

- Child-process transport。client 启动 server，通过 stdin/stdout 通信。
- 每行一个 JSON object。Newline-delimited。
- 没有 session id；process identity 就是 session。
- 不需要 auth（child 继承 parent 的 trust boundary）。
- 永远不要用于 remote servers；否则你需要 SSH 或 socat tunnel，而这时应该直接用 Streamable HTTP。

### Streamable HTTP / 可流式 HTTP

单 endpoint `/mcp`（或任意 path）。支持三种 HTTP methods：

- **POST /mcp.** client 发送 JSON-RPC message。server 返回单个 JSON response，或一个包含一个或多个 responses 的 SSE stream（对 batched responses 和与该 request 相关的 notifications 有用）。
- **GET /mcp.** client 打开 long-lived SSE channel。server 用它发送 server-to-client requests（sampling、notifications、elicitation）。
- **DELETE /mcp.** client 显式终止 session。

Sessions 由 `Mcp-Session-Id` header 标识：server 在首次 response 上设置它，client 在所有后续 request 上回显。Session ids 必须是 cryptographically random（128+ bits）；出于安全考虑，client-chosen ids 会被拒绝。

### Single endpoint vs two / 单 endpoint vs 双 endpoint

旧 spec 的 two-endpoint mode 在 2026 年仍然可调用，spec 将其标记为 "legacy compatible"。但所有新 server 都应该使用 single-endpoint。官方 SDK 输出 single-endpoint；只有在对接尚未迁移的 remote 时才使用 legacy mode。

### `Origin` validation and DNS-rebinding / `Origin` 校验与 DNS-rebinding

浏览器今天不是 MCP clients，但攻击者可以构造网页，诱导浏览器向 `localhost:1234/mcp` 发送 POST，而用户的本地 MCP server 正监听在那里。如果 server 不检查 `Origin`，浏览器同源策略救不了它，因为 `Origin: http://evil.com` 对跨源请求是合法的。

2025-11-25 spec 要求 server 拒绝 `Origin` 不在 allowlist 中的请求。allowlist 通常包含 MCP client host（`https://claude.ai`、`vscode-webview://*`）和用于 local UIs 的 localhost variants。

### Session id lifecycle / Session id 生命周期

1. Client 首次 request 不带 `Mcp-Session-Id`。
2. Server 分配 random id，并在 response header 上设置 `Mcp-Session-Id`。
3. Client 在所有后续 requests 和 `GET /mcp` stream 上回显该 header。
4. Server 可以 revoke session；client 在后续 requests 上看到 404，必须重新 initialize。
5. Client 可以显式 DELETE session，完成 clean shutdown。

### Keepalive and reconnect / Keepalive 与重连

SSE connections 会断。client 用同一个 `Mcp-Session-Id` 重新 GET 来重建连接。Server 必须把 outage 期间错过的 events 排队（在合理窗口内），并通过 client 回显的 `last-event-id` header 重放。

Phase 13 · 13 会讲 Tasks，它们能让 long-running work 即使在 full-session reconnect 后仍然存活。

### Backwards compatibility probe / 向后兼容探测

一个同时支持旧新 server 的 client 可以这样做：

1. POST 到 `/mcp`。
2. 如果 response 是带 JSON 或 SSE 的 `200 OK`，这是 Streamable HTTP。
3. 如果 response 是 `200 OK`，`Content-Type: text/event-stream`，且 `Location` header 指向 secondary endpoint，这是 legacy HTTP+SSE；跟随 `Location`。

### Cloudflare, ngrok, and hosting / Cloudflare、ngrok 与托管

2026 年生产 remote MCP servers 常运行在 Cloudflare Workers（配合 MCP Agents SDK）、Vercel Functions，或容器化 Node/Python 上。关键点：hosting 必须支持用于 SSE GET 的 long-lived HTTP connections。Vercel free tier 上限 10 秒，不适合。Cloudflare Workers 支持 indefinite streams。

### Gateway composition / Gateway 组合

当你用 gateway 前置多个 MCP servers（Phase 13 · 17）时，gateway 是一个单一 Streamable HTTP endpoint，会重写 session ids 并 multiplex upstream。Tools 在 gateway layer 合并；client 看到的是一个 logical server。

### Transport failure modes / 传输失败模式

- **stdio SIGPIPE.** Child process 在 mid-write 时死亡会触发 SIGPIPE；servers 应干净退出。clients 应检测 EOF 并标记 session dead。
- **HTTP 502 / 504.** Cloudflare、nginx 等 proxy 会在 upstream failure 时发出这些。Streamable HTTP clients 应短暂 backoff 后重试一次。
- **SSE connection drop.** TCP RST、proxy timeout 或 client network change 会关闭 stream。client 携带 `Mcp-Session-Id` 和可选 `last-event-id` 重连以恢复。
- **Session revocation.** Server invalidates session id；client 下一次 request 看到 404。client 必须重新 handshake。
- **Clock skew.** client 的 Resource-TTL 计算和 server 偏离。client 应以 server timestamps 为准。

### When to bypass Streamable HTTP / 什么时候绕过 Streamable HTTP

一些企业会把 MCP servers 部署在内部网络的 gRPC 或 message-queue transports 后面。这不是标准做法，MCP spec 没有正式定义这些。Gateways 可以对 MCP clients 暴露 Streamable HTTP 表面，同时内部使用 gRPC。保持 external surface spec-compliant；translation 由 gateway 负责。

## Build It / 动手构建

本课会用 stdlib `http.server` 写一个最小 Streamable HTTP endpoint：实现 POST、GET、DELETE 的结构，生成并校验 session id，拒绝未通过 `Origin` allowlist 的请求。重点是协议边界，而不是 Web framework。

## Use It / 应用它

`code/main.py` 使用 `http.server`（stdlib）实现一个最小 Streamable HTTP endpoint。它处理 `/mcp` 上的 POST、GET 和 DELETE，在第一次 response 上设置 `Mcp-Session-Id`，校验 `Origin`，并拒绝非 allowlisted origins。handler 复用 Lesson 07 notes server 的 dispatch logic。

重点看：

- POST handler 读取 JSON-RPC body，dispatch，并写 JSON response（single-response variant；SSE variant 结构类似）。
- `Origin` check 会拒绝默认的 `http://evil.example` probe，但接受 `http://localhost`。
- Session ids 是 random 128-bit hex strings；server 在内存中保留 per-session state。

## Ship It / 交付它

本课产出 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（legacy）MCP server，这个 skill 会生成迁移到 Streamable HTTP 的计划，覆盖 session-id continuity、Origin checks 和 backwards-compatible probe support。

## Exercises / 练习

1. 运行 `code/main.py`。用 `curl` POST 一个 `initialize`，观察 `Mcp-Session-Id` response header。第二次 POST 时回显该 header，验证 session continuity。

2. 添加一个打开 SSE stream 的 GET handler。每五秒发送一个 `notifications/progress` event。用同一个 session id 重新 GET，确认 server 接受重连。

3. 实现 `last-event-id` replay logic。重连时，重放自该 id 以来生成的所有 events。

4. 扩展 `Origin` validation，支持 wildcard pattern（`https://*.example.com`），并确认它接受 `https://app.example.com`，拒绝 `https://evil.example.com.attacker.net`。

5. 从 official registry 中选一个 legacy HTTP+SSE server（有好几个），草拟迁移：endpoint handling、session id generation 和 header semantics 分别要改什么。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| stdio transport | “Local child process” | stdin/stdout 上的 JSON-RPC，newline-delimited |
| Streamable HTTP | “remote transport” | single-endpoint POST + GET + optional SSE，2025-03-26 spec |
| HTTP+SSE | “Legacy” | 2026 年中移除的 two-endpoint model |
| `Mcp-Session-Id` | “Session header” | server 分配的 random id，后续每个 request 都回显 |
| `Origin` allowlist | “DNS-rebinding defense” | 拒绝 Origin 未获批准的请求 |
| Single endpoint | “One URL” | `/mcp` 处理所有 session operations 的 POST / GET / DELETE |
| `last-event-id` | “SSE replay” | 用于恢复断开的 stream 且不丢 events 的 header |
| Backwards-compat probe | “Old vs new detection” | client response-shape check，自动选择 transport |
| Long-lived HTTP | “SSE streaming” | server 在一个 TCP connection 上推送数分钟或数小时 events |
| Session revocation | “Force re-init” | server invalidates session id；client 必须重新 handshake |

## Further Reading / 延伸阅读

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 与 Streamable HTTP 的 canonical reference
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的 revision
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers-hosted Streamable HTTP patterns
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — deployment shapes 对比
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体迁移 deadline 示例
