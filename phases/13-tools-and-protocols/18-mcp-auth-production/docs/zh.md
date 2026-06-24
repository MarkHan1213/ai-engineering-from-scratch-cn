# MCP Auth in Production — Enrollment, JWKS Refresh, Audience-Pinned Tokens / 生产中的 MCP Auth：注册、JWKS 刷新与 Audience-Pinned Tokens

> Lesson 16 在内存中搭起了 OAuth 2.1 state machine。到 2026 年，任何面向真实组织交付的 MCP server 都必须接入生产级 auth：能扩展到无限 client population 的 client enrollment（优先 Client ID Metadata Documents，dynamic client registration 作为 backward-compatible fallback）、authorization-server metadata discovery（RFC 8414 *or* OpenID Connect Discovery）、不会在凌晨三点 key rotation 时打断 token validation 的 JWKS cache refresh，以及拒绝 cross-resource replay 的 audience-pinned tokens。本课用三个角色建模完整认证链路：authorization server、resource server（MCP server）和 client，让你从 discovery 到 validated tool call 逐跳追踪。
>
> **Spec note (2025-11-25):** 2025 年 11 月 MCP authorization spec 把 Dynamic Client Registration 从 `SHOULD` 降为 `MAY`，并把 **Client ID Metadata Documents (CIMD)** 设为推荐默认 enrollment mechanism。本课按 spec 的优先顺序同时讲两者；代码保留 DCR walkthrough，因为它能在一个进程中完全自包含。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Time / 时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 通过 RFC 8414 metadata 发现 authorization server，并验证 contract。
- 实现 RFC 7591 dynamic client registration，让 MCP clients 无需 admin intervention 即可 enroll。
- 按计划 cache 和 refresh JWKS keys，使 signature verification 能经受 key roll-over。
- 使用 RFC 8707 resource indicators 把 tokens pin 到单个 MCP resource，并拒绝 confused-deputy reuse。
- 干净分离三种角色：authorization server、resource server、client，让每个角色只执行属于自己的检查。
- 阅读 IdP capability matrix，并在 IdP 无法满足 MCP auth profile 时拒绝部署。

## The Problem / 问题

Lesson 16 simulator 在内存中运行 OAuth 2.1。生产环境有三个 memory-only simulator 看不到的 operational gaps。

第一个缺口是 enrollment。真实组织会运行数百个 MCP servers 和数千个 MCP clients。operator 不会手工把每个 Cursor 用户注册成 OAuth client。2025-11-25 spec 给 client 一个优先级顺序：如果已有 pre-registered `client_id` 就用它；否则使用 **Client ID Metadata Document**（client 用一个自己控制的 HTTPS URL 标识自身，authorization server *pull* metadata）；否则退回 **RFC 7591 dynamic client registration**（client *push* 一个 `POST /register` 并当场收到 `client_id`）；再不行才提示用户。CIMD 是推荐默认方案，因为它完全移除 per-server registration，同时保留 DNS-rooted trust model；DCR 保留用于 backwards compatibility。两者的入口都来自 authorization server metadata：CIMD 看 `client_id_metadata_document_supported`，DCR 看 `registration_endpoint`。

第二个缺口是 key rotation。JWT validation 依赖 authorization server 发布在 JSON Web Key Set（JWKS）中的 signing keys。authorization server 会按计划轮换这些 key（常见是 hourly，在 incident response 中可能更快）。如果 MCP server 只在启动时 fetch 一次 JWKS，rotation window 到来前都能工作，之后每个 request 都失败，直到重启。生产做法是把 JWKS 接成带 refresh job 的 cached value：在旧 keys 过期前覆盖 cache；同时在 cache miss 时做一次 fall-back fetch，以处理 token 由比 cache 更新的 key 签发的情况。

第三个缺口是 audience binding。Lesson 16 引入 RFC 8707 resource indicators。在生产里，这个 indicator 会变成每个 request 的硬 claim check。MCP server 比较 `token.aud` 和自己的 canonical resource URL，不匹配就 HTTP 401。它是防止 upstream MCP server（或持有某 server token 的恶意 client）把 token replay 到同一 trust mesh 中另一个 server 的唯一协议层防线。

本课把每个 gap 映射到具体 surface。metadata document 是 HTTP endpoint。JWKS cache refresh 是 scheduled job 加 key-value cache。JWT validation 是 resource server 在 dispatch 任意 tool 前运行的 routine。保持三个角色分离：authorization server 负责签发和 rotate keys，resource server 负责 cache 和 validate，client 负责 discover 和 enroll。

## The Concept / 概念

### RFC 8414 — OAuth Authorization Server Metadata / RFC 8414：OAuth Authorization Server Metadata

`/.well-known/oauth-authorization-server` 上的 document 描述 client 需要的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

给定 MCP resource URL 后，client 做 chained discovery：先通过 RFC 9728 的 `oauth-protected-resource`（resource server document）找到 issuer，再通过 RFC 8414 的 `oauth-authorization-server` 找到所有 endpoint。client 不再 hard-code authorization URL。

把某个 IdP 信任为 MCP IdP 前要验证的 contract：

- `code_challenge_methods_supported` 包含 `S256`（PKCE per RFC 7636）。spec 明确说：如果该字段 **absent**，authorization server 不支持 PKCE，client **MUST** refuse to proceed。
- `grant_types_supported` 包含 `authorization_code`，并拒绝 `password` 和 `implicit`。
- 至少公布一种 enrollment path：`client_id_metadata_document_supported: true`（CIMD，preferred）**or** `registration_endpoint`（RFC 7591 DCR，fallback）。任一满足 contract；不再 hard-require DCR。
- `response_types_supported` 对 OAuth 2.1 应为 `["code"]`。

如果缺少 `S256`，MCP server 拒绝部署到这个 IdP；PKCE 没有 degraded mode。如果 *neither* enrollment path 被公布，且你没有 pre-registered `client_id`，也无法 enroll；这是 deployment manifest 错了，不是代码问题。

### RFC 9728 (recap) — Protected Resource Metadata / RFC 9728 回顾：Protected Resource Metadata

Lesson 16 已讲过 RFC 9728。生产中的差异是：client 只从这里查找 *这个* MCP server 信任哪些 authorization servers。一个 MCP server 可以接受多个 IdP 的 tokens（员工一个，合作伙伴一个）。RFC 9728 声明这个集合；RFC 8414 记录每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents (the recommended default) / Client ID Metadata Documents（推荐默认）

CIMD 把 registration 从 *push* 反转为 *pull*。client 不再请求 authorization server mint 一个 `client_id`，而是使用自己控制的 HTTPS URL **作为** `client_id`。该 URL 解析为 JSON metadata document；authorization server 在 OAuth flow 中按需 fetch 它。trust rooted in DNS：如果 server operator 信任 `app.example.com`，就信任来自 `https://app.example.com/client.json` 的 client。无需 registration round-trip，不消耗 `client_id` namespace，也无需维护 per-server state。

client 托管的 metadata document：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

document 中的 `client_id` value **MUST** 等于它被 serve 的 URL（authorization server 会验证；不匹配则拒绝）。authorization server 通过 RFC 8414 metadata 中的 `client_id_metadata_document_supported: true` 宣告支持。

spec 对两点安全事实说得很直白：

- **SSRF.** authorization server 会 fetch attacker-supplied URL。它必须防御 server-side request forgery（不能 fetch internal/admin endpoints）。
- **localhost impersonation.** 仅靠 CIMD 不能阻止本地攻击者冒用合法 client metadata URL 并绑定任意 `localhost` redirect。authorization server **MUST** 在 consent 时清晰展示 redirect URI hostname，并且 **SHOULD** 对 `localhost`-only redirects 给出警告。

由于 CIMD 不需要 server-side state，也就不需要像 DCR 那样部署 registrar。client 侧是 read-only：把 metadata document 放在静态 HTTPS endpoint，让 authorization server 拉取。

### RFC 7591 — Dynamic Client Registration (fallback / backwards compatibility) / RFC 7591：Dynamic Client Registration（fallback / 向后兼容）

DCR 现在是 `MAY`，保留给 pre-2025-11-25 deployments 和尚未支持 CIMD 的 IdPs。没有它（且没有 CIMD 或 pre-registration）时，每个 MCP client（Cursor、Claude Desktop、自定义 agent）都需要和 IdP admin 做 out-of-band exchange。有 DCR 时，client 会 post：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

server 用 `client_id` 和用于后续更新的 `registration_access_token` 响应：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

对运行在用户设备上的 MCP clients，`token_endpoint_auth_method: none` 是正确默认值。它们只拿 `client_id`，没有可被 exfiltrate 的 `client_secret`。PKCE 提供 public clients 所需的 proof-of-possession。

三个生产陷阱：

- registration endpoint 必须按 source IP rate-limit。否则 hostile actor 可以脚本化数百万 fake registrations，耗尽 `client_id` namespace。registrar 处理 request 前先跑 rate-limit check。
- 一些 enterprise IdPs 要求 `software_statement`（为 client 背书的 signed JWT）。本课 mock 跳过它；生产应加 verification step，拒绝除 localhost redirect URIs 之外的 unsigned registrations。
- `registration_access_token` 必须以 hash 存储，而不是 plaintext。这个 token 被偷就意味着攻击者可以改写 client redirect URIs。

### RFC 8707 (recap) — Resource Indicators / RFC 8707 回顾：Resource Indicators

Lesson 16 已建立形状。生产规则：每个 token request 都包含 `resource=<canonical-mcp-url>`，MCP server 在每次 call 上验证 `token.aud` 是否匹配自身 resource URL。canonical URI 是该 server 的 *most specific* identifier：scheme 和 host 小写、无 fragment，通常无 trailing slash。path component **不会** 按规则剥离；当它用于标识单个 MCP server 时，spec 会保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是合法 canonical URIs。每个 server 选一个，并把 `aud` 精确 pin 到它。（本课 mock 为简洁使用 `https://notes.example.com` 这种 bare-host audience；如果一个 deployment 在同一 origin 下 co-host 多个 MCP servers，就通过 path 区分。）

### RFC 7636 (recap) — PKCE / RFC 7636 回顾：PKCE

OAuth 2.1 强制 PKCE。本课 authorization-code flow 总是携带 `code_challenge` 和 `code_verifier`。server 会拒绝任何没有 verifier，或 verifier hash 不匹配 stored challenge 的 token request。

### MCP Spec 2025-11-25 Auth Profile / MCP Spec 2025-11-25 Auth Profile

MCP spec（2025-11-25）对 MCP server 的 authorization layer 要做什么非常精确：

- 实现 RFC 9728 protected-resource metadata，并通过 401 上的 `WWW-Authenticate: Bearer resource_metadata="..."` header **或** well-known URI `/.well-known/oauth-protected-resource` 暴露它的位置（SEP-985 让 header 可选，并提供 well-known fallback）。metadata 的 `authorization_servers` field **MUST** 至少列出一个 server。
- 每个 request 都只能通过 `Authorization: Bearer ...` 接受 tokens：绝不放 query string，绝不只在 session start 校验一次。
- 每个 request 都验证 `aud`、`iss`、`exp` 和 required scopes。server **MUST** 验证 token 确实是为它签发的（audience）；缺失或不匹配的 `aud` 会被拒绝，绝不当作 wildcard。
- 在 401/403 上返回 `WWW-Authenticate: Bearer`，携带 `error=...`、`resource_metadata="<PRM-URL>"` parameter（metadata document 的 URL，*不是* bare resource），并在 `insufficient_scope`（403）时携带 `scope="..."`。注意：parameter 是 `resource_metadata`，它是 discovery pointer；challenge 中没有 `resource` parameter。
- Authorization-server discovery 接受 RFC 8414 OAuth metadata **或** OpenID Connect Discovery 1.0；clients 必须按 priority order 尝试两个 well-known suffixes。
- client（不是 server）防御 **mix-up attacks**：redirect 前记录 expected `issuer`，并在 redeem code 前验证 `iss` authorization-response parameter（RFC 9207）。PKCE 单独不能阻止 mix-up，因为 client 会把 `code_verifier` 交给它被引导到的任意 token endpoint。

OAuth 2.1 draft 是 substrate；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是 surface；MCP spec 是 profile。

### IdP capability matrix / IdP 能力矩阵

不是每个 IdP 都支持完整 MCP profile。下表记录截至 2025-11-25 spec 的事实能力声明。它是 *deployment gate*，不是推荐。

CIMD 在 2025-11-25 spec 中发布，而底层 OAuth draft 到 2025 年 10 月才被 adopted，所以 vendor support 仍在到来。下面的 "CIMD" 应理解为“当前状态，请在你的 tenant 验证”，不是永久结论。

| IdP category | AS metadata (8414/OIDC) | CIMD | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | Notes |
|---|---|---|---|---|---|---|
| Self-hosted (Keycloak) | yes | emerging | yes | yes (since 24.x) | yes | 本课 MCP profile 的 reference IdP；DCR path 端到端完整，CIMD 跟进新 spec。 |
| Enterprise SSO (Microsoft Entra ID) | yes | emerging | yes (premium tiers) | yes | yes | DCR 可用性随 tenant tier 不同；部署前在目标 tenant 验证。 |
| Enterprise SSO (Okta) | yes | emerging | yes (Okta CIC / Auth0) | yes | yes | Auth0（now Okta CIC）可用 DCR；classic Okta orgs 需要 admin pre-registration。 |
| Social login IdPs (generic) | varies | no | rarely | rarely | yes | 多数 social IdPs 把 clients 当静态 partners；没有 self-service enrollment。只把它们用作 identity source，在上层叠加自己的 MCP-aware authorization server。 |
| Custom / homegrown | depends | depends | depends | depends | depends | 自研就交付完整 profile，并优先 CIMD。跳过 PKCE 或 audience binding 会破坏 MCP auth contract。 |

deployment manifest 的 refusal rule：如果选定 IdP 没在 `code_challenge_methods_supported` 中列出 `S256`，MCP server 拒绝启动；PKCE 没有 degraded mode。Enrollment 是较软的 gate：你需要 *一个* 可用路径（pre-registered `client_id`、`client_id_metadata_document_supported: true` 或 `registration_endpoint`）。DCR 缺失本身不再触发拒绝，因为 CIMD 或 pre-registration 可以覆盖。

### JWKS refresh pattern (rotate at the AS, refresh at the resource server) / JWKS 刷新模式（AS rotate，resource server refresh）

保持两个动词分离，因为混淆它们是真实生产 bug：

- **Rotate** 是 *authorization server* 做的：mint 一个新的 signing key，把它发布到 JWKS，稍后 retire 旧 key。resource server 与此无关，也不能做这件事；它不持有 IdP private keys。
- **Refresh** 是 *resource server* 做的：重新 `GET` 已发布的 JWKS 到自己的 cache。这是 resource server 唯一会做的 JWKS 动作。

生产失败模式是 stale cache。解决方案是 scheduled refresh job 加 key-value cache。resource server 跑一个固定 interval 的 job（cron、timer 或 runtime 自带机制），fetch `<issuer>/.well-known/jwks.json` 并覆盖 `cache[issuer] = {keys, fetched_at}`。validator 从 cache 读取。若某个 token 的 `kid` 不在 cache 中，就触发 **一次** synchronous refresh 作为 fallback，然后重新检查。这同时处理两种情况：计划刷新，以及 key-overlap window 中，新 key 签的 token 在下次计划 refresh 前到达。

fallback **必须是 re-fetch，绝不能是 rotate**。如果你把 cache-miss path 接到 rotate-and-mint，会同时坏两件事：(1) mint 新 key 会产生一个仍然不匹配 token 的 `kid`，lookup 还是失败；(2) 攻击者用随机 `kid` tokens 轰炸时，会强迫系统无限创建 keys，造成自我 DoS。re-fetch 是 idempotent，所以 bogus `kid` 最多浪费一次 fetch。

cache shape：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

稳态下通常有两个 keys。authorization server 轮换时，会在 retire 前一个 key（`k_2026_03`）之前先引入下一个 key（`k_2026_04`），所以用旧 key 签发的 tokens 在过期前仍有效。cache 保存 union；validator 按 `kid` 选择。

### The validation routine / 校验例程

MCP server 在 dispatch 任意 tool 前运行 validation。`code/main.py` 使用的形状：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS cache 中解析 signing key（miss 时刷新一次），验证 signature，然后检查 `iss` 是否在 allow-list 中、`aud` 是否等于这个 server 的 canonical resource、`exp` 和 required scope；第一次失败时返回 `WWW-Authenticate` challenge。把它保持为 resource server 上的单一 routine，意味着每个 entry point（每个 tool call、每种 transport）都会经过相同检查；不存在绕过 validation 直接到达 tool 的路径。

### Audience-replay walkthrough (access-token privilege restriction) / Audience replay 演练（access-token privilege restriction）

Server A（`notes.example.com`）和 Server B（`tasks.example.com`）都注册到同一个 authorization server。Server A 被攻陷。攻击者拿到用户的 notes token，并把它 replay 到 Server B。

Server B 的 validator：

1. Decode JWT，通过 `kid` fetch JWKS，verify signature。
2. 检查 `iss` 是否在它的 protected-resource metadata 的 `authorization_servers` 中。（通过，同一个 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败，token 的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

audience claim 是 protocol layer 防御这个攻击的唯一机制。为了性能跳过它，是最常见生产错误；validator 必须在每个 request 上运行，而不是只在 session start。spec 把这称为 **access-token privilege restriction**：MCP server `MUST` reject 任何没有把它列入 audience 的 token。

> **Naming note.** spec 把 *confused deputy* 保留给一个相关但不同的问题：MCP server 作为第三方 API 的 OAuth **proxy**，使用 static client ID，并在没有 per-client user consent 的情况下转发 token。audience binding 修复上面的 replay；confused-deputy 的修复是 per-client consent **plus** 永远不要把 inbound token 透传到 upstream APIs（MCP server `MUST` 获取自己的 separate upstream token）。

### Mix-up attacks (a client-side defense the server cannot provide) / Mix-up attacks（server 无法提供的 client-side 防御）

client 生命周期中会与许多 authorization servers 对话。恶意 AS 可以试图让 client 把 honest AS 的 authorization code 兑换到攻击者 token endpoint。audience binding 在这里没用，因为攻击发生在 token 存在之前。防御位于 client（RFC 9207）：

1. redirect 前，client 从已验证 AS metadata 记录 expected `issuer`。
2. 在 authorization response 上，client 比较返回的 `iss` parameter 与记录的 issuer（简单字符串比较，不做 normalization），然后才发送 code。
3. mismatch（或 AS 宣称 `authorization_response_iss_parameter_supported` 但 `iss` 缺失）→ reject，并且不要展示 `error` fields。

PKCE 单独不能阻止 mix-up，因为 client 会把 `code_verifier` 交给它被引导到的任意 token endpoint。这就是为什么 spec 会把 issuer 与 PKCE verifier、`state` 一起按 request 记录。

### Failure modes / 失败模式

- **Stale JWKS.** AS 轮换 key 后，validator 拒绝 valid tokens。修复是上面的 cron-refresh + cache-miss-refetch。永远不要没有 refresh job 地 cache JWKS。
- **Rotate-as-fall-back.** 把 cache-miss path 接到 rotate-and-mint 而不是 re-fetch 是真实 bug：它永远产不出 missing `kid`，还会把 attacker-controlled `kid` values 变成 key-creation DoS。fallback 必须是 idempotent `refresh-jwks`。
- **Missing `aud` claim.** 一些 IdPs 默认不发 `aud`，除非 token request 中出现 `resource`。validator 必须拒绝缺失 `aud` 的 tokens，不能把 absence 当 wildcard。
- **Mix-up via missing `iss` check.** client 如果不把 RFC 9207 `iss` authorization-response parameter 与 redirect 前记录的 issuer 比较，就可能被引导到攻击者 token endpoint 兑换 honest AS 的 code。这是 client-side failure；resource server 无法补偿。
- **Scope upgrade race.** 同一用户两个 concurrent step-up flows 可能都成功，并产生两个 scopes 不同的 access tokens。validator 必须使用 request 上呈现的 token，而不是查“该用户当前 scope”；后者会制造 TOCTOU window。
- **Registration token theft.** 泄露的 `registration_access_token` 允许攻击者改写 redirect URIs。at rest 时 hash；每次 update 要求 client 呈 cleartext；怀疑泄露时 rotate。
- **`iss` not pinned.** validator 接受任何 `iss`，攻击者就能自建 authorization server，为 target audience 注册 client 并签发 tokens。protected-resource metadata 的 `authorization_servers` list 就是 allow-list；必须执行。

## Build It / 动手构建

本课会用 stdlib Python 建模三个角色：`AuthorizationServer` 负责 metadata、registration、token issuance 和 key rotation；`ResourceServer` 负责 protected-resource metadata、JWKS cache 与 token validation；`Client` 负责 discovery、enrollment、PKCE flow 和 tool call。核心是把每项检查放在正确角色上。

## Use It / 应用它

`code/main.py` 用 stdlib Python 和三个角色走完整生产 flow：`AuthorizationServer`、`ResourceServer` 和 `Client`。流程：

1. Authorization server 在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 metadata。
2. MCP client 调用 metadata endpoint，检查 enrollment options（CIMD 的 `client_id_metadata_document_supported`、DCR 的 `registration_endpoint`）和 `S256` PKCE support。
3. walkthrough 采用 DCR fallback path：client POST 到 `/register`（RFC 7591）并收到 `client_id`。（CIMD client 会呈现自己的 HTTPS `client_id` URL，并跳过这一步。）
4. MCP client 用 `resource` indicator（RFC 8707）运行 PKCE-protected authorization code flow（RFC 7636）。
5. MCP client 用 `Authorization: Bearer ...` 调用 MCP server 上的 tool。
6. MCP server 运行 `validate`，从 JWKS cache 解析 signing key。
7. IdP rotate 一个 key；scheduled refresh 把 JWKS 重新拉进 cache。
8. 下一次 call 使用 refreshed keys 无需重启即可 validate，旧 token 在 overlap window 中仍能 validate。
9. 对另一个 MCP resource 的 audience-replay attempt 得到 401，带 `audience mismatch` 和 `resource_metadata` pointer。

这里的 JWT 使用 HS256 和 shared secret（为了只依赖 stdlib）。生产使用 RS256 或 EdDSA 加上前述 JWKS pattern；validation logic 其余部分相同。由于 IdP 和 resource server 位于同一进程，`refresh_jwks` 直接读取 authorization server 的 key list；在线上它是对 `jwks_uri` 的 HTTP `GET`。

## Ship It / 交付它

本课产出 `outputs/skill-mcp-auth.md`。给定 MCP server config 和 IdP capability set，这个 skill 会输出要搭建的 auth surface：protected-resource metadata、enrollment path（CIMD、pre-registration 或 DCR fallback）、JWKS refresh schedule、scope mapping，以及 IdP 不支持完整 RFC profile 时的 refusal rules。

## Exercises / 练习

1. 运行 `code/main.py`。追踪 flow。注意 IdP 如何在 step 6 rotate key，scheduled `refresh_jwks` 如何重新拉取 published set，以及 old token（overlap window）和 fresh token 如何无需重启都能 validate。

2. 向 protected-resource metadata 的 `authorization_servers` list 添加一个新 IdP。签发一个由新 IdP 签名的 token，确认 validator 接受它。再签发一个由未列出的 IdP 签名的 token，确认 validator 拒绝并返回 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`。

3. 给 `register_client` 添加 rate-limit check，且在 registrar 接收 request 前执行。使用一个按 IP keyed 的小 dict 保存 per source IP token-bucket。

4. 阅读 RFC 7591，找出本课 `/register` handler 没有 validate 的两个字段。添加 validation。（提示：`software_statement` 和 `redirect_uris` URI scheme。）

5. 添加 Client ID Metadata Document path。serve 一个 `client_id` 等于自身 URL 的 `client.json`，让 authorization server fetch 并验证它（若 `client_id` ≠ URL 则 reject）。确认 CIMD client 不调用 `register_client` 也能 enroll。

6. 证明 DoS fix。给 validator 发送一个带 random `kid` 的 token，确认 `refresh_jwks` 最多运行一次，并且 authorization server 的 key count 不增长。然后故意把 fallback 改接 rotate-and-mint，观察每个 bogus token 都会让 key count 上升；之后恢复 re-fetch。

7. 实现 mix-up section 中的 client-side RFC 9207 `iss` check：authorization request 前记录 expected issuer，然后拒绝 `iss` 不匹配的 authorization response。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ASM | “OAuth metadata document” | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | “Client metadata URL” | Client ID Metadata Document：用作 `client_id` 的 HTTPS URL；AS 拉取 JSON。自 2025-11-25 起为推荐默认 |
| DCR | “Self-service client registration” | RFC 7591 `POST /register` flow；在 2025-11-25 降为 `MAY` fallback |
| JWKS | “Public keys for JWT validation” | JSON Web Key Set，从 `jwks_uri` fetch，以 `kid` 索引 |
| Rotate vs refresh | “Updating the keys” | *Rotate* = AS mint/retire signing keys；*refresh* = resource server 重新 fetch published set。resource server 只会 refresh |
| Resource indicator | “Audience parameter” | RFC 8707 `resource` parameter，把 token pin 到一个 server |
| `aud` claim | “Audience” | validator 与 canonical resource URL 比较的 JWT claim |
| Audience replay | “Token replay” | 给 Server A 的 token 被呈给 Server B；由 audience validation 防御（spec: access-token privilege restriction） |
| Confused deputy | “Proxy token misuse” | 静态 client ID 的 MCP proxy 未经 per-client consent 转发 token；不同于 audience replay |
| Mix-up attack | “Wrong token endpoint” | client 被引导到攻击者 endpoint 兑换 honest AS 的 code；通过 RFC 9207 `iss` 在 client-side 防御 |
| `iss` allow-list | “Trusted authorization servers” | protected-resource metadata 的 `authorization_servers` 中命名的集合 |
| `resource_metadata` | “Where to find the PRM doc” | 401/403 上 `WWW-Authenticate` parameter，指向 RFC 9728 metadata URL |
| Public client | “Native or browser client” | 没有 `client_secret` 的 OAuth client；由 PKCE 补偿 |
| `WWW-Authenticate` | “401/403 response header” | 携带驱动 client recovery 的 `Bearer error=...` directives |

## Further Reading / 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 本课实现的 MCP auth profile
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025-11-25 变化（CIMD、XAA、DCR demotion）
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — CIMD-over-DCR rationale
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — discovery contract
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（fallback path）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — public-client proof-of-possession
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience pinning
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — resource server discovery
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 防御 mix-up attacks 的 `iss` parameter
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — consolidated OAuth substrate
