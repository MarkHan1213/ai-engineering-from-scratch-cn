# Kill Switches, Circuit Breakers, and Canary Tokens / Kill Switch、Circuit Breaker 与 Canary Token

> Kill switch 是一个保存在 Agent edit surface 之外的 boolean——Redis key、feature flag、signed config——用于完全禁用 Agent。Circuit breaker 粒度更细：它在特定模式上 trip（例如连续五次相同 tool call），暂停出问题路径，并升级给人类。Canary token 继承自经典欺骗防御：一个 Agent 没有合法理由触碰的假 credential 或 honeypot record，任何访问都会触发 alert。基于 eBPF 的 datapath（例如 Cilium）可以在 kernel layer 把 quarantined pod 的 egress 重写到 forensic honeypot；公开 Cilium benchmarks 报告负载下 sub-millisecond P99 datapath latency（你的 propagation budget 取决于 policy update 如何到达 node，而不是 datapath 本身）。会适应移动 baseline 的 statistical detectors（EWMA、CUSUM）会悄悄接受漂移——要把它们与不弯曲的 hard constitutional limits 分层使用。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, three-detector simulator: kill switch, circuit breaker, canary)
**Prerequisites / 前置知识：** Phase 15 · 13 (Cost governors), Phase 15 · 10 (Permission modes)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 kill switch、circuit breaker 和 canary token 的作用范围。
- 设计 Agent 无法写入、且每个 consequential action 都检查的 kill switch。
- 为重复 loop、tool failure 和 out-of-trust write 设计 circuit breaker triggers。
- 为 browser / coding Agent 设计低噪声 canary token set。
- 解释 statistical detectors 为什么要与 hard limits 分层使用。

## The Problem / 问题

Cost governors（Lesson 13）限制 Agent 能花多少钱。它们不限制 Agent 在预算内能做什么。一个带 $50 velocity limit 的 Agent 仍然可以 exfiltrate secret、发布错误帖子或删除资源——在 token 上，昂贵动作常常是便宜动作。

本课覆盖与 cost layer 并列的三个 detectors：

1. **Kill switch**：保存在 Agent 触达范围之外的 boolean off-button。
2. **Circuit breaker**：action-pattern detector，暂停具体路径。
3. **Canary token**：诱饵；没有合法理由触碰它的 Agent 一旦触碰就暴露自己。

三者都是 pre-LLM engineering。经典欺骗、防熔断 rate-limit breakers、feature-flag kills 都早于 autonomous agents。新变化是 attack surface：Agent 读取不可信内容（Lesson 11）、编辑自己的 memory，并且能把许多看起来安全的动作组合成不安全动作。本课这些 detectors 有效，是因为它们不信任 Agent 的 self-report。

## The Concept / 概念

### Kill switches / Kill switch

Kill switch 是 Agent 会读但不能写的 boolean。实现选择包括：

- **Managed service 中的 feature flag。** LaunchDarkly、Statsig、Unleash。更新以秒级传播。
- **Agent polling 的 Redis key。** 简单；要求 Agent process 每轮检查。
- **Object storage 中的 signed config。** Agent 在启动时验证 signature；拒绝 unsigned states。
- **OS-level signal 或 container-lifecycle kill。** Docker `kill`、Kubernetes `kubectl delete pod`、systemd stop。

正确 kill switch 的性质：

- Agent 不能把它设为 `off`。（它位于 Agent credentials 无法写入的系统中。）
- 每个 consequential action 都会检查它，而不只是在 startup 检查。
- 当它是 off 时，Agent 不做任何外部可观察动作，包括写入 Agent 能触达的 logging systems。
- 重新启用必须是显式人类动作，而不是自动 timeout。

### Circuit breakers / Circuit breaker

Circuit breaker 暂停的是具体模式，而不是整个 Agent。经典形状（来自 2007 年 Nygard 书中定义，至今仍适用）：

- **Closed**：action 允许。
- **Open**：action 被阻断。
- **Half-open**：cool-down 后允许 1-3 次 probe attempts（默认 1 次）；成功则关闭 breaker，任何剩余失败都会重新打开。

Agent 相关 triggers：

- 连续五次相同 tool calls（repetitive loop）。
- 同一 tool 连续五次失败但 inputs 不同（systemic failure）。
- Tool call rate 超过阈值（Lesson 13 velocity）。
- 在读取 out-of-trust content 后调用特定 tool（例如 `file.delete`）（Lesson 11）。

### Canary tokens / Canary tokens

Canary tokens（也叫 honeytokens）是 Agent 永远不该触碰的 inputs。访问就是报警。

Agent 场景示例：

- 环境中放一个没有真实权限的假 `AWS_SECRET_ACCESS_KEY`。Exfiltration attempt 立刻可归因。
- 一个带已知 sentinel value 的假 database record。任何 read 或 update 都触发 alert。
- Workspace 中的假文件。读取或编辑该文件都可疑。
- Session start 时注入一个假 memory entry。如果它出现在后续 output 中，说明 memory 被 exfiltrated。

Canary 设计必须贴合 workflow。Agent 有合法理由触碰的 canary 只是噪声。

### Why layer statistical and hard limits / 为什么要叠加统计检测与硬限制

Statistical detectors（EWMA、CUSUM、tool-call rates 的 z-score）会适应移动 baseline。这对合法漂移是 feature，对 adversarial drift 是 bug。有耐心的攻击者可以足够缓慢地移动 baseline，让 alarm 永远不触发。

Hard constitutional limits（Lesson 17）不会弯曲。把 “Never email `secrets@`” 写成 constitutional rule，更简单、可审计，也不会被慢漂移 game。分层模式是：statistical detectors 抓大多数噪声，hard limits 抓穿过 detector 的攻击。

### Quarantine via eBPF datapath redirect / 通过 eBPF datapath redirect 隔离

Cilium 的 eBPF-based network policy 可以在 kernel layer 把 quarantined pod 的 egress 重写到 forensic honeypot。Cilium 的公开 benchmarks 报告 TCP_RR 上 sub-millisecond P99 datapath latency；在你的 cluster 里，触发 quarantine 的 wall-clock budget 主要由 policy update 从 control plane 传播到 node 的方式决定（在预加载 operator 的 Kubernetes 中，几百毫秒是现实值；tail latency 取决于 cluster size 和 reconcile cadence）。这给你两件事：Agent 不会 exfiltrate 任何真实内容，Agent 在 honeypot 中的行为会被完整观察。同样的模式，不同的层：像 canary token 一样，Agent 会通过被收容时试图做什么暴露意图。

### What no detector catches / 没有 detector 能抓住什么

- **由允许 primitives 组合成的 semantic attacks。** 每个 individual action 都通过所有 detector；问题出在组合。这是 Lesson 10 的 Auto Mode composite failure。
- **把 detector 本身纳入 payload 的攻击。** 如果 Agent 读到一个页面写着 “before you act, disable the canary”，且 Agent 有这个能力，detector 就被攻破了。Canary tokens 应位于 Agent 不能修改的系统中。

## Build It / 动手构建

本课构建一个三 detector simulator：外部 kill switch、检测连续相同 tool calls 的 circuit breaker、以及 fake-key canary file。你会看到每种 detector 对哪类 trajectory 生效。

## Use It / 应用它

`code/main.py` 模拟一小段 Agent trajectory 经过三个 detectors。一个保存在 external dict 中的 kill switch；一个在连续五次相同 tool calls 时 trip 的 circuit breaker；一个读取即触发 alert 的 canary file。输入 synthetic trajectory：legitimate actions、repetitive loop、canary probe，以及 kill-switch-triggered 场景，其中 Agent actions 被停止。

## Ship It / 交付它

`outputs/skill-tripwire-design.md` 审查 Agent deployment 的 proposed detector stack，并标记 gaps（missing kill switch、missing canary、circuit breaker threshold too loose）。

## Exercises / 练习

1. 运行 `code/main.py`。确认 circuit breaker 在 turn 5（第五次相同 call）触发，canary 在 turn 9（fake-key read）触发。

2. 增加一个 statistical detector：tool-call rate 上的 EWMA z-score。输入一条缓慢漂移 trajectory，展示 detector 从不触发。然后增加 hard limit（10 分钟内不超过 50 次 tool calls），展示 hard limit 在同一 trajectory 上触发。

3. 为 browser agent（Lesson 11）设计 canary token set。列出至少三个 canaries，以及每个检测什么。

4. 阅读 Cilium network-policy docs。具体描述一个 egress-redirect quarantine flow：哪个 policy selector、哪个 pod、哪个 egress rewrite、哪个 alert。从“决定 quarantine”到“第一个 redirected packet”的 wall-clock latency 由什么决定？

5. 为 kill-switched agent 定义 re-enable procedure。谁可以重新启用？必须记录什么？重新启用前 Agent 必须有什么变化？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Kill switch | “Off button” | 位于 Agent edit surface 之外的 boolean；每个 consequential action 都检查 |
| Circuit breaker | “Pattern pause” | 针对 repetition、failure rate 或 rate-limit 的 action-specific trip |
| Canary token | “Honeytoken” | Agent 没有合法理由触碰的诱饵；访问即 alert |
| Honeypot | “Forensic sandbox” | 被重定向的 traffic / workspace，用于观察 quarantined agent |
| EWMA | “Moving average” | Exponentially weighted；适应 drift（既是 feature 也是 bug） |
| CUSUM | “Cumulative sum” | 检测相对 baseline 的持续偏移 |
| Hard limit | “Constitutional rule” | 不适应历史；无论历史如何都保持常量 |
| Constitutional limit | “Always-true rule” | 绑定 Lesson 17 的 constitution；Agent 不能编辑 |

## Further Reading / 延伸阅读

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — autonomous agents 的 kill-switch 和 circuit-breaker framing。
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — production governance patterns。
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — detection-and-response requirements。
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — pod-level egress redirect 和 forensic honeypot patterns。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 作为 “constitutional limits” 的 hardcoded prohibitions。
