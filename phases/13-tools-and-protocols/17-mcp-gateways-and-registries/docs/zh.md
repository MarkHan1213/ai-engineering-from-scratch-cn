# MCP Gateways and Registries — Enterprise Control Planes / MCP Gateway 与 Registry：企业控制平面

> 企业不能让每个开发者随意安装随机 MCP servers。gateway 会集中 auth、RBAC、audit、rate limiting、caching 和 tool-poisoning detection，然后把 merged tool surface 暴露为单个 MCP endpoint。Official MCP Registry（Anthropic + GitHub + PulseMCP + Microsoft，namespace-verified）是 canonical upstream。本课会说明 gateway 位于哪里，走过一个最小实现，并扫一遍 2026 vendor landscape。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, minimal gateway)
**Prerequisites / 前置知识：** Phase 13 · 15 (tool poisoning), Phase 13 · 16 (OAuth 2.1)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 MCP gateway 位于哪里（MCP clients 与多个 backend MCP servers 之间）。
- 实现五项 gateway responsibilities：auth、RBAC、audit、rate limit、policy。
- 在 gateway layer 强制 pinned-tool-hash manifest。
- 区分 Official MCP Registry 和 metaregistries（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## The Problem / 问题

一家 Fortune 500 有 30 个 approved MCP servers、5000 名开发者、compliance 和 audit requirements，以及一个希望集中 policy 的 security team。让每个开发者在 IDE 中安装任意 server 不现实。

gateway pattern：

1. gateway 作为单个 Streamable HTTP endpoint 运行，供开发者连接。
2. gateway 持有每个 backend MCP server 的 credentials。
3. 每个开发者 request 都通过 gateway 自己的 OAuth 做 authentication 和 scoping。
4. gateway 把调用路由到 backend server，并应用 policy。
5. 所有 calls 记录到 audit。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway 都在 2025-2026 年发布了 gateways 或 gateway features。

与此同时，Official MCP Registry 作为 canonical upstream 发布：curated、namespace-verified、reverse-DNS-named servers，gateway 可从中拉取。Metaregistries（Glama、MCPMarket、MCP.so、Smithery、LobeHub）则从多个来源聚合 servers。

## The Concept / 概念

### Five gateway responsibilities / 五项 gateway 职责

1. **Auth.** 用 OAuth 2.1 识别 developer，并映射到 user roles。
2. **RBAC.** per-user policy：哪些 servers、哪些 tools、哪些 scopes。
3. **Audit.** 记录每次 call 的 who、what、when、result。
4. **Rate limit.** per-user / per-tool / per-server caps，防止滥用。
5. **Policy.** 拒绝 poisoned descriptions，执行 Rule of Two，redact PII。

### Gateway as a single endpoint / Gateway 作为单 endpoint

对开发者来说，gateway 看起来像一个 MCP server。内部它会路由到 N 个 backends。Session ids（Phase 13 · 09）在边界被重写。

### Credential vaulting / 凭证托管

开发者永远看不到 backend tokens。gateway 持有它们，或代理到 identity provider。一个在 gateway 上拥有 `notes:read` 的开发者，可以在 policy 绑定下，通过 gateway 的 backend credentials 传递访问 notes MCP server，但不能越过 policy。

### Tool-hash pinning at the gateway / Gateway 层的 tool-hash pinning

gateway 持有一份 approved tool descriptions manifest（SHA256 hashes）。discovery 时，它获取每个 backend 的 `tools/list`，把 hash 与 manifest 对比，并移除 description 已变更的工具。这是 Phase 13 · 15 的 rug-pull defense 的集中化版本。

### Policy-as-code / Policy-as-code

高级 gateway 会用 OPA/Rego、Kyverno 或 Styra 表达 policy。诸如 “user `alice` may call `github.open_pr` only on repos in org `acme`” 这样的规则会被声明式编码。简单 gateway 使用手写 Python。两种形状都有效。

### Session-aware routing / 感知 session 的路由

当用户 session 混合多个 servers 时，gateway 会 multiplex：developer 的单个 MCP session 内部持有 N 个 backend sessions，每个 server 一个。任何 backend 的 notifications 都会经 gateway 路由回 developer session。

### Namespace merging / 命名空间合并

gateways 会合并所有 backends 的 tool namespaces，通常 collision 时加 prefix。`github.open_pr`、`notes.search`。这让 routing 不歧义。

### Registries / Registries

- **Official MCP Registry (`registry.modelcontextprotocol.io`).** 在 Anthropic、GitHub、PulseMCP、Microsoft stewardship 下发布。Namespace-verified（reverse-DNS: `io.github.user/server`）。会做基本质量预过滤。
- **Glama.** 搜索导向的 metaregistry，聚合多个来源。
- **MCPMarket.** 偏商业目录，包含 vendor listings。
- **MCP.so.** 社区目录，开放提交。
- **Smithery.** package-manager-style installation flow。
- **LobeHub.** 集成在 LobeChat app 中的 UI registry。

enterprise gateways 默认从 Official Registry 拉取，允许 admin-curated metaregistry additions，并拒绝任何 unpinned 内容。

### Reverse-DNS naming / Reverse-DNS 命名

Official Registry 要求 public servers 使用 reverse-DNS names：`io.github.alice/notes`。Namespaces 可以防止 squatting，并让 trust delegation 更清楚。

### Vendor survey, April 2026 / Vendor survey，2026 年 4 月

| Vendor | Strength |
|--------|----------|
| Cloudflare MCP Portals | Edge-hosted; OAuth integrated; free tier |
| Kong AI Gateway | K8s-native; fine-grained policy; logs to OpenTelemetry |
| IBM ContextForge | Enterprise IAM; compliance; audit export |
| TrueFoundry | DevOps-leaning; metrics-first |
| MintMCP | Developer-platform oriented |
| Envoy AI Gateway | Open-source; customizable filters |

Phase 17（production infrastructure）会更深入 gateway operations。

## Build It / 动手构建

本课会实现一个约 150 行的 minimal gateway：fake Bearer token authentication、per-user RBAC、两个 backend MCP servers 的 routing、append-only audit log、token-bucket rate limit，以及 pinned manifest hash 检查。

## Use It / 应用它

`code/main.py` 提供一个约 150 行的 minimal gateway：通过 fake Bearer token 认证用户，持有 per-user RBAC policy，把 requests 路由到两个 backend MCP servers，写入每次调用到 audit log，执行 rate limit，并拒绝任何 backend tool description hash 与 pinned manifest 不匹配的工具。

重点看：

- `RBAC` dict 以 `user_id` 为 key，值是允许的 `server_tool` entries。
- `AUDIT_LOG` 是 append-only events list。
- Rate limit 使用 per user token bucket。
- Pinned manifest 是 `server::tool -> hash` 的 dict。

## Ship It / 交付它

本课产出 `outputs/skill-gateway-bootstrap.md`。给定一个 enterprise MCP plan（users、backends、compliance），这个 skill 会产出 gateway configuration spec。

## Exercises / 练习

1. 运行 `code/main.py`。用 allowed user 调用一次；再用 disallowed user；再做一次 rate-limit-exceeded burst。验证三条 flow。

2. 添加一个在结果返回 client 前 redacts PII 的 policy。对 SSN-shaped strings 使用 simple regex pass；记录缺口（emails、phone numbers）。

3. 扩展 audit log，发出 OpenTelemetry GenAI spans。Phase 13 · 20 会覆盖精确 attributes。

4. 为一个 50 人开发团队设计 RBAC policy，五个 backends：notes、github、postgres、jira、slack。谁获得每个 backend 的 read-only？谁能 write？

5. 从头到尾阅读 Cloudflare enterprise MCP post。找出 Cloudflare 提供但这个 stdlib gateway 没有的一个 feature。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Gateway | “MCP proxy” | 位于 clients 和 backends 之间的 centralized server |
| Credential vaulting | “Backend tokens stay server-side” | 开发者永远看不到 upstream tokens |
| Session-aware routing | “Multi-backend session” | gateway 为每个 developer session multiplex N 个 backend sessions |
| Tool-hash pinning | “Approved manifest” | 每个 approved tool description 的 SHA256；集中阻断 rug-pulls |
| RBAC | “Per-user policy” | 针对 tools 和 servers 的 role-based access control |
| Policy-as-code | “Declarative rules” | 在 gateway 执行的 OPA/Rego、Kyverno、Styra policies |
| Audit log | “Who, what, when” | 用于 compliance 的 append-only event log |
| Rate limit | “Per-user token bucket” | 防止滥用的 per-minute caps |
| Official MCP Registry | “Canonical upstream” | `registry.modelcontextprotocol.io`，namespace-verified |
| Reverse-DNS naming | “Registry namespace” | `io.github.user/server` convention |

## Further Reading / 延伸阅读

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — canonical upstream，namespace-verified
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — 带 OAuth 和 policy 的 gateway pattern
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — open-source reference gateway
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — feature comparison article
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — IBM 的 enterprise gateway
