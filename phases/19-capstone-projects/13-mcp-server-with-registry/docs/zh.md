# Capstone 13 — MCP Server with Registry and Governance / 带注册中心与治理的 MCP Server

> 到 2026 年，Model Context Protocol 不再是未来方向，而是默认的 tool-use spec。Anthropic、OpenAI、Google 和所有主流 IDE 都提供 MCP clients。Pinterest 公开了内部 MCP servers 生态。AAIF Registry 把 capability metadata 固化在 `.well-known`。AWS ECS 发布 reference stateless deployment。Block 的 goose-agent 把同一协议放进 hosted assistant。2026 年生产形态是：StreamableHTTP transport、OAuth 2.1 scopes、OPA policy gating，以及让平台团队 discover、validate、enable servers 的 registry。把这整套端到端构建出来。

**类型：** 综合项目
**语言：** Python（server, via FastMCP） 或 TypeScript（@modelcontextprotocol/sdk）, Go（registry service）
**前置知识：** 第 11 阶段（LLM engineering）, 第 13 阶段（tools and MCP）, 第 14 阶段（agents）, 第 17 阶段（infrastructure）, 第 18 阶段（safety）
**Phases exercised:** P11 · P13 · P14 · P17 · P18
**时间：** 25 小时

## Learning Objectives / 学习目标

- 构建支持 StreamableHTTP 的 production-grade MCP server
- 实现 OAuth 2.1 per-tool scopes、OPA / Rego policy gate 和 destructive tool approval flow
- 设计 `.well-known/mcp-capabilities` capability manifest 与 registry service
- 建立 per-tenant audit log、PII redaction、load test 和 conformance tests
- 交付可发现、可验证、可治理的内部工具 MCP 生态入口

## Problem / 问题

MCP 已经成为 tool-use 的通用语。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 以及所有 managed agents 都消费 MCP servers。生产挑战不在 authoring servers（FastMCP 让这件事很简单），而在企业级部署：per-tenant OAuth scopes、destructive tools 上的 OPA policy、StreamableHTTP stateless scaling、用于 discovery 的 registry，以及每次 tool call 的 audit logs。Pinterest 的内部 MCP 生态和 AAIF Registry spec 设定了 2026 年标准。

你将构建一个暴露 10 个内部工具的 MCP server（Postgres read-only、S3 listing、Jira、Linear、Datadog 等）、一个用于平台发现的 registry UI，以及一个 destructive tools 的 human-approval gate。load test 要证明 StreamableHTTP 可以水平扩展。audit trail 要能通过企业安全审查。

## Concept / 概念

MCP 2026 revision 将 StreamableHTTP 定为默认 transport。与早期 stdio-and-SSE 形态不同，StreamableHTTP 默认 stateless：单个 HTTP endpoint 接受 JSON-RPC requests，流式返回 responses，并支持 notifications 的 long-lived connections。stateless 意味着可以放在 load balancer 后面水平扩展。

授权使用 OAuth 2.1，并按工具定义 scopes。token 携带 `jira:read`、`s3:list`、`postgres:query:readonly` 这类 scopes。MCP server 在 tool-call time 检查 scopes，而不只是在 session start 检查。对于高风险工具，如果 token 没有在最近 N 分钟内提升为 `approved:by:human` scope，server 会拒绝调用；这个 elevation 来自 Slack review card。

registry 是独立服务。每个 MCP server 暴露 `.well-known/mcp-capabilities` document，包含 tool manifest、transport URL 和 auth requirements。registry 定期 poll、validate、index。平台团队通过 registry UI 查看可用工具、所需 scopes、owner team。

## Architecture / 架构

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## Stack / 技术栈

- Server framework: FastMCP (Python) 或 `@modelcontextprotocol/sdk` (TypeScript)
- Transport: StreamableHTTP over HTTPS（stateless）
- Auth: OAuth 2.1，workload identity 使用 SPIFFE / SPIRE
- Policy: OPA / Rego rules per tool；每个请求调用 policy decision service
- Registry: self-hosted，消费 `.well-known/mcp-capabilities` manifests
- Human approval: destructive tools 使用 Slack interactive message
- Deployment: AWS ECS Fargate 或 Fly.io；每个 tenant 一个 server，或共享 server 加 tenant scoping
- Audit: per-tenant bucket 中的 structured JSONL，带 per-call lineage

## Build It / 动手构建

1. **Tool surface.** 暴露 10 个内部工具：Postgres read-only query、S3 list objects、Jira search/fetch、Linear search/fetch、Datadog metric query、PagerDuty on-call lookup、GitHub read-only、Notion search、Slack search、Salesforce read。每个工具都有 typed schema 和 scope label。

2. **FastMCP server.** 挂载工具。配置 StreamableHTTP transport。添加 OAuth token introspection 和 scope enforcement middleware。

3. **OPA policy.** 每个工具一条 Rego policy：哪些 scopes 允许调用、哪些 PII redaction 要应用、payload-size caps 是多少。每次 tool call 都调用 decision service。

4. **Registry service.** 独立 Go 或 TS service，从 registered servers poll `.well-known/mcp-capabilities`，用 JSON Schema validate，并提供 list / search / validate / enable-disable UI。

5. **Capability manifest.** 每个 server 暴露 `.well-known/mcp-capabilities`，内容包括：tool list、auth requirements、transport URL、owner team、SLO。

6. **Destructive tool separation.** 会改变状态的工具（Jira create、Linear create、Postgres write）放在第二个 MCP server，使用更严格 auth flow：tokens 必须在 15 分钟内通过 Slack card 提升过 `approved:by:human` scope。

7. **Audit log.** 每个 tenant append-only JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前用 Presidio 做 PII redaction。

8. **Load test.** 100 concurrent clients on StreamableHTTP。通过增加第二个 replica 演示 horizontal scaling；展示 load balancer 无需 session stickiness 也能重新分配负载。

9. **Conformance tests.** 对两个 servers 运行 official MCP conformance suite。通过所有 mandatory sections。

## Use It / 应用它

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## Ship It / 交付它

`outputs/skill-mcp-server.md` 描述交付物：一个 production-grade MCP server + registry + audit layer，用于带 OAuth 2.1 scopes 和 OPA gating 的内部工具。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Spec conformance | StreamableHTTP + capability manifest 通过 MCP conformance tests |
| 20 | Security | Scope enforcement、每个工具的 OPA coverage、secret hygiene |
| 20 | Observability | 带 PII redaction 的 per-tool-call audit log |
| 20 | Scale | 100-client load test horizontal scale demonstration |
| 15 | Registry UX | Discover / validate / enable-disable workflow |
| **100** | | |

## Exercises / 练习

1. 添加新工具（Confluence search）。不改 core server，通过 registry validation flow 发布它。

2. 写一条 OPA policy，用来 redacts Postgres query results 中列名为 `email`、`ssn` 或 `phone` 的字段。用 probe query 演练。

3. Benchmark StreamableHTTP vs stdio 的本地 latency。报告 per-call p50/p95。

4. 实现 per-tenant quota：每个 tenant 每个 tool 每分钟最多 N calls。通过第二条 OPA rule 执行。

5. 运行 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 的 MCP conformance suite，并修复所有失败。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| StreamableHTTP | “2026 MCP transport” | Stateless HTTP + streaming；替代 networked servers 上的 SSE + stdio |
| Capability manifest | “Well-known doc” | `.well-known/mcp-capabilities`，包含 tool list、auth、transport URL |
| OPA / Rego | “Policy engine” | Open Policy Agent，用外部规则授权 tool calls |
| Scope elevation | “Approved-by-human” | 通过 Slack approval 临时授予的短期 scope，destructive tools 必需 |
| Registry | “Tool discovery” | 从 capability manifests 索引 MCP servers 的服务 |
| Workload identity | “SPIFFE / SPIRE” | 为 OAuth token issuance 提供 cryptographic service identity |
| Conformance suite | “Spec tests” | 针对 StreamableHTTP + tool manifest correctness 的官方 MCP test battery |

## Further Reading / 延伸阅读

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、capability metadata、registry
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) — 2026 registry spec
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — reference production deployment
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — reference internal deployment
- [Block `goose` MCP usage](https://block.github.io/goose/) — reference agent consumption pattern
- [FastMCP](https://github.com/jlowin/fastmcp) — Python server framework
- [Open Policy Agent](https://www.openpolicyagent.org/) — policy engine reference
- [SPIFFE / SPIRE](https://spiffe.io) — workload identity reference
