# MCP Security II — OAuth 2.1, Resource Indicators, Incremental Scopes / MCP 安全 II：OAuth 2.1、Resource Indicators 与增量 Scope

> Remote MCP servers 需要 authorization，而不只是 authentication。2025-11-25 spec 对齐 OAuth 2.1 + PKCE + resource indicators（RFC 8707）+ protected-resource metadata（RFC 9728）。SEP-835 在 403 WWW-Authenticate 上加入 incremental scope consent 和 step-up authorization。本课会把 step-up flow 实现成 state machine，让你看清每一跳。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, OAuth state machine simulator)
**Prerequisites / 前置知识：** Phase 13 · 09 (transports), Phase 13 · 15 (security I)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 resource server 和 authorization server 的职责。
- 走通 PKCE-protected OAuth 2.1 authorization code flow。
- 使用 `resource`（RFC 8707）和 protected-resource metadata（RFC 9728）防止 confused-deputy attacks。
- 实现 step-up authorization：server 用带 WWW-Authenticate 的 403 请求更高 scope；client 重新提示用户 consent 并 retry。

## The Problem / 问题

早期 MCP（2025 前）远程 server 常用 ad-hoc API keys，甚至没有 auth。2025-11-25 spec 用完整 OAuth 2.1 profile 补上这个缺口。

三个真实需求：

- **Ordinary remote servers.** 用户安装一个访问自己 Notion / GitHub / Gmail 的 remote MCP server。OAuth 2.1 + PKCE 是正确形状。
- **Scope escalation.** 一个 notes server 先获得 `notes:read`，之后某个具体动作需要 `notes:write`。无需重做整个 flow，step-up（SEP-835）只请求额外 scope。
- **Confused deputy prevention.** client 持有一个 audience-scoped 给 Server A 的 token。恶意 Server A 试图把 token 呈给 Server B。Resource indicators（RFC 8707）会把 token pin 到预期 audience。

OAuth 2.1 本身并不新。新的是 MCP profile：指定 required flows（只允许 authorization code + PKCE；默认不允许 implicit，也不允许 client credentials）、每个 token request 都强制 resource indicators，并发布 protected-resource metadata，让 clients 知道该去哪。

## The Concept / 概念

### Roles / 角色

- **Client.** MCP client（Claude Desktop、Cursor 等）。
- **Resource server.** MCP server（notes、GitHub、Postgres 等）。
- **Authorization server.** 发 token 的服务。可以和 resource server 是同一服务，也可以是独立 IdP（Auth0、Keycloak、Cognito）。

在 MCP profile 中，resource 和 authorization servers 可以是同一个 host，但应该用 URLs 区分。

### Authorization code + PKCE / Authorization code + PKCE

流程：

1. Client 生成 `code_verifier`（random）和 `code_challenge`（SHA256）。
2. Client 把用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户 consent。Authorization server 重定向到 `redirect_uri?code=...`。
4. Client POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. Authorization server 校验 verifier 的 hash 是否匹配 stored challenge，并签发 access token。
6. Client 在每个 request 上用 `Authorization: Bearer ...` 调用 resource server。

PKCE 防止 authorization-code interception attacks。Resource indicators 防止 token 在别处有效。

### Protected-resource metadata (RFC 9728) / Protected-resource metadata（RFC 9728）

resource server 发布 `.well-known/oauth-protected-resource` document：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

client 从 resource server 发现 authorization server。这样可以减少配置：client 只需要 resource URL。

### Resource indicators (RFC 8707) / Resource indicators（RFC 8707）

token request 中的 `resource` parameter 会 pin token 的 intended audience。签发出的 token 包含 `aud: "https://notes.example.com"`。另一个 MCP server 收到该 token 时会检查 `aud` 并拒绝。

### Scope model / Scope 模型

Scopes 是空格分隔的 strings。常见 MCP conventions：

- `notes:read`, `notes:write`, `notes:delete`
- `admin:*` for admin capabilities（谨慎使用）
- `profile:read` for identity

scope selection 应遵循 least-privilege：只请求当前需要的，需要更多时再 step up。

### Step-up authorization (SEP-835) / Step-up authorization（SEP-835）

用户授予 `notes:read`。稍后要求 agent 删除 note。server 响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

client 看到 insufficient_scope error，向用户弹出 additional scope consent dialog，为它执行一个 mini OAuth flow，然后用新 token 重试 request。

### Token audience validation / Token audience 校验

每个 request 都检查 `token.aud == self.resource_url`。不匹配 = 401。这会阻止 cross-server token reuse。

### Short-lived tokens and rotation / 短期 token 与轮换

Access tokens 应该 short-lived（默认 1 小时）。Refresh tokens 每次刷新都轮换。client 在后台处理 silent refresh。

### No token passthrough / 禁止 token 透传

Sampling servers（Phase 13 · 11）不能把 client token 透传到其他服务。sampling request 就是边界。

### Confused deputy prevention / Confused deputy 防护

Token 绑定到 `aud`。Client 绑定到 `client_id`。每个 request 都按两者校验。spec 明确禁止 pre-MCP remote tool ecosystems 中常见的旧式 "pass-the-token" pattern。

### Client ID discovery / Client ID 发现

每个 MCP client 在固定 URL 发布自己的 metadata。Authorization servers 可以获取 client metadata document，发现 redirect URIs 和 contact info。这移除了 manual client registration。

### Gateways and OAuth / Gateways 与 OAuth

Phase 13 · 17 会展示 enterprise gateway 如何处理 OAuth：gateway 持有 upstream servers 的 credentials，给 client 的 token 由 gateway 签发，upstream tokens 不离开 gateway。这会翻转 trust model：用户只和 gateway 认证一次；gateway 处理 N 个 server authorizations。

## Build It / 动手构建

本课会用内存 state machine 模拟完整 OAuth 2.1 step-up flow：从 protected-resource metadata 发现，到 PKCE code flow，再到 `insufficient_scope` 后重新 consent 并 retry。重点是协议状态，而不是 HTTP server。

## Use It / 应用它

`code/main.py` 把完整 OAuth 2.1 step-up flow 模拟成 state machine。它实现：

- PKCE code-verifier / challenge generation。
- 带 resource indicator 的 authorization code flow。
- Protected-resource metadata endpoint。
- 带 audience check 的 token validation。
- `insufficient_scope` 上的 step-up。

本课没有 HTTP server；state machine 在内存中运行，便于追踪每一跳。Phase 13 · 17 的 gateway lesson 会把它接到实际 transport。

## Ship It / 交付它

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带工具的 remote MCP server，这个 skill 会设计 scope set、pinning rules 和 step-up policy。

## Exercises / 练习

1. 运行 `code/main.py`。追踪 two-scope step-up flow。记录 step-up 时重复了哪些 hop。

2. 添加 refresh-token rotation：每次 refresh 都签发新 refresh token 并使旧 token 失效。模拟被盗 refresh token 在 rotation 后使用，确认失败。

3. 用 stdlib `http.server` 把 protected-resource metadata endpoint 实现成真实 HTTP response。镜像 Lesson 09 的 /mcp endpoint。

4. 为 GitHub MCP server 设计 scope hierarchy：read repo、write PR、approve PR、merge PR、admin。每层之间使用 step-up。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中一个 MCP 用法与 RFC 示例不同的字段。（提示：它涉及 `scopes_supported`。）

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| OAuth 2.1 | “Modern OAuth” | 整合 RFC，强制 PKCE 并禁止 implicit flow |
| PKCE | “Proof-of-possession” | code verifier + challenge，抵御 authorization-code interception |
| Resource indicator | “Token audience” | RFC 8707 `resource` parameter，把 token pin 到一个 server |
| Protected-resource metadata | “Discovery doc” | RFC 9728 `.well-known/oauth-protected-resource` |
| Step-up authorization | “Incremental consent” | SEP-835 flow，按需追加 scopes |
| `insufficient_scope` | “403 with WWW-Authenticate” | server 信号，要求为更大 scope 重新 consent |
| Confused deputy | “Token reuse across services” | trusted holder 不恰当地转发 token 的攻击 |
| Short-lived token | “Access token TTL” | 很快过期的 bearer；refresh token 负责续期 |
| Scope hierarchy | “Least privilege stack” | 分级 scope set，每层之间 step-up |
| Client ID metadata | “Client discovery doc” | client 发布自身 OAuth metadata 的 URL |

## Further Reading / 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — canonical MCP OAuth profile
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 changes walkthrough
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience-pinning RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — discovery-document RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — practical step-up-flow walkthrough
