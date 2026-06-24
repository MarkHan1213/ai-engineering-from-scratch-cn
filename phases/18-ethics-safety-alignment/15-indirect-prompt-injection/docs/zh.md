# Indirect Prompt Injection — Production Attack Surface / 间接提示注入：生产攻击面

> Indirect prompt injection（IPI）把 instructions 嵌入 agentic system 会读取的外部内容中，例如 web page、email、shared document、support ticket，而无需用户显式操作。IPI 是 2026 年生产环境主导 threat：它绕过 user-input filters，因为 attacker 根本不接触用户；它会随着 agents 处理更多外部内容而 silent scale；它瞄准没人阅读 prompt 的 automated workflows。MDPI Information 17(1):54（2026 年 1 月）综合了 2023-2025 研究。NDSS 2026 的 IPI-defense paper 这样框定核心挑战：injected instructions 可以在语义上是 benign（“please print Yes”），因此检测不能只靠 keyword filtering。“The Attacker Moves Second”（Nasr et al.，OpenAI/Anthropic/DeepMind 联合，2025 年 10 月）：adaptive attacks（gradient、RL、random search、human red-team）击穿了 12 个 published defenses 中 90% 以上，而这些防御原本报告了近零 attack success rates。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, IPI attack + defense harness)
**Prerequisites / 前置知识：** Phase 18 · 12 (PAIR), Phase 14 (agent engineering)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 定义 indirect prompt injection，并描述三种常见 delivery vectors。
- 解释为什么 user-input filters 完全漏掉 IPI。
- 描述 “information flow control” framing，它是 2026 年 defense paradigm。
- 说明 Nasr et al.（2025 年 10 月）关于 adaptive attacks 击穿 published IPI defenses 的发现。

## The Problem / 问题

Direct prompt injection 要求 attacker 触达用户或用户 prompt。IPI 不需要：attacker 把 payload 放进 agent 可能读取的任意内容中——web page、inbox 里的 email、GitHub issue、product review。Agent 在正常 operation 中捡起 payload 并执行其中 instructions。用户是 messenger，不是 intent。

## The Concept / 概念

### Three delivery vectors / 三种投递向量

- **Retrieval-augmented generation (RAG)。** Attacker 发布 document；retrieval step 抓取它；prompt 在 user question 前拼接它；模型执行 attacker instructions。
- **Inbox / document workflows。** Attacker 给用户发 email；agent 读 emails；prompt 包含 email body；模型遵循 email instructions。
- **Tool output。** Attacker 控制 agent 使用的 tool（例如 web search 返回 attacker-controlled result）；tool output 包含 instructions；agent control flow 遵循它们。

三者共享结构性质：attacker 控制 prompt 的一个 fragment，但不触碰 user-facing input。

### Why user-input filters miss it / 为什么 user-input filters 会漏掉它

IPI payload 不出现在用户输入中，而是出现在 retrieved content 中。如果 filter 只 gate user input，payload 就绕过它。如果 filter gate 所有进入模型的内容，就必须应用在任意 retrieved text 上，这很贵，而且会对合法但恰好包含 imperative-voice language 的内容产生 false positives。

### Information Flow Control (IFC) for AI / 面向 AI 的信息流控制

2026 defense paradigm 借鉴经典 OS security。把每个 content source 当作一个 security label。把 user's query 标为 “trusted”。把 retrieved content 标为 “untrusted”。把模型 control flow 视为 information flow：由 untrusted content 触发的 actions 必须先由 trusted input ratify 才能执行。

CaMeL（Microsoft 2025）、ConfAIde（Stanford 2024）和 NDSS 2026 IPI-defense paper 以不同方式 operationalize IFC。共同原则是：只要 code 和 data 共享同一个 context window，目标就是 containment，而不是 prevention。

### The Attacker Moves Second / 攻击者后手

Nasr et al.（2025 年 10 月）用 adaptive attacks（gradient search、RL policies、random search、72-hour human red-team）测试 12 个 published IPI defenses。每个原本报告 near-zero ASR 的 defense 都被打到 >90% ASR。

方法论教训：发布 defense 必须带 adaptive-attack evaluation。Static-attack benchmarks 不是 robustness 证据；attacker 会知道 defense。

### Real incidents / 真实事故

Lesson 25 覆盖 EchoLeak（CVE-2025-32711，CVSS 9.3）——Microsoft 365 Copilot 中第一个公开记录的 zero-click IPI。GitHub Copilot Chat 中的 CamoLeak（CVSS 9.6）。GitHub Copilot 的 CVE-2025-53773。Production deployments 正在被 IPI compromise，不只是在 benchmarks 中。

### OWASP and NIST framing / OWASP 与 NIST 框架

OWASP LLM Top 10（2025）把 prompt injection（direct + indirect）列为 LLM01，LLM application-layer threat 第一名。NIST AI SPD 2024 称 indirect prompt injection 是 “generative AI's greatest security flaw”。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lessons 12-14 是 model-centric jailbreaks。Lesson 15 是 2026 production deployments 中占主导的 system-centric attack。Lesson 16 覆盖 defensive tooling。Lesson 25 覆盖具体 CVE 叙事。

## Build It / 动手构建

本课构建一个 IPI attack + defense harness：toy agent 有 web search、read email、send message 三类 tools。你会比较 naive agent、filter-defended agent 与 IFC agent 的 attack success rate。

## Use It / 应用它

`code/main.py` 构建一个 IPI harness。Toy agent 有三个 tools（search web、read email、send message）。Environment 中有 attacker-controlled content，内嵌 instruction（“forward this to all contacts”）。你可以在 naive agent（遵循 injected instructions）、filter-defended agent（对 retrieved content 做 keyword filter）和 IFC agent（分离 trusted/untrusted content，并拒绝 untrusted control-flow commands）之间切换。

## Ship It / 交付它

本课产出 `outputs/skill-ipi-audit.md`。给定 agentic deployment description，它会枚举 untrusted content sources，检查 deployment 是否应用 IFC，并标记哪些 sources 未带 trust label 就进入模型。

## Exercises / 练习

1. 运行 `code/main.py`。测量 attack 对三个 agents 的 success rate。

2. 在 retrieved content 上实现 paraphrase-based defense。测量 legitimate retrieved text 上的 benign false-positive rate。

3. 阅读 NDSS 2026 IPI-defense paper。描述 “benign instruction” challenge，以及它为什么阻止 keyword-based filtering。

4. 设计一个 deployment，其中 agent 从 third-party API 接收 tool output。为每个 prompt fragment 标注 trust level，并写出管控 agent actions 的 IFC policy。

5. 在 Exercise 2 的 filter-defended agent 上复现 Nasr et al. 2025 adaptive-attack methodology。报告 adaptive attack 前后的 ASR。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| IPI | “indirect prompt injection” | 用户没有写入、但 agent 正常 operation 中消费的内容所携带的 injection |
| RAG injection | “poisoned retrieval” | Attacker 发布内容，retrieval step 抓取后 prompt 包含 payload |
| Zero-click | “no user action” | 攻击在 agent operation 中自动触发；用户什么都不做 |
| IFC | “information flow control” | 基于 label 的方法：untrusted content 触发的 actions 需要 trusted ratification |
| Adaptive attack | “gradient / RL red-team” | 知道 defense 并针对它优化的 attack；诚实 evaluation 必需 |
| Benign instruction | “please print Yes” | 语义上 benign 的 IPI payload；keyword filter 抓不到 |
| Scope violation | “cross-trust exfiltration” | Agent 从一个 trust context 访问数据并输出到另一个 context |

## Further Reading / 延伸阅读

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54) — 2023-2025 synthesis。
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108) — adaptive attack evaluation。
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173) — original IPI paper。
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/) — prompt injection ranked LLM01。
