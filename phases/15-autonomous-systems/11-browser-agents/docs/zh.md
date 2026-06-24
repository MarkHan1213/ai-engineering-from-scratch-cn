# Browser Agents and Long-Horizon Web Tasks / 浏览器 Agent 与长周期 Web 任务

> ChatGPT agent（2025 年 7 月）把 Operator 和 deep research 合并成一个 browser/terminal agent，并在 BrowseComp 上达到 68.9% SOTA。OpenAI 在 2025 年 8 月 31 日关闭 Operator——这是产品层的整合。Anthropic 收购 Vercept 后，把 Claude Sonnet 在 OSWorld 上从低于 15% 推到 72.5%。WebArena-Verified（ServiceNow, ICLR 2026）修复了原 WebArena 中 11.3 个百分点的 false-negative rate，并发布 258-task Hard subset。这些数字是真的。Attack surface 也是真的：OpenAI 的 preparedness 负责人公开表示，针对 browser agents 的 indirect prompt injection “is not a bug that can be fully patched”。2025-2026 已记录攻击包括 Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）和 Perplexity Comet 的 one-click hijacks。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, indirect prompt-injection attack surface model)
**Prerequisites / 前置知识：** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 描述 browser agent 为什么同时是 long-horizon agent 和 untrusted-content processor。
- 对比 BrowseComp、OSWorld、WebArena-Verified 测量的能力轴。
- 识别 indirect prompt injection、URL fragment injection、memory-binding 和 CSRF-shaped attacks。
- 解释为什么 browser-agent prompt injection 不能被完全 patch。
- 为真实浏览器 Agent workflow 设计 read/write boundary、HITL 和 canary memory controls。

## The Problem / 问题

Browser agent 是一种会读取不可信内容并采取有后果动作的 long-horizon agent。Agent 访问的每个页面都是用户没有写过的输入。每个页面上的每个 form 都可能是 command channel。2025-2026 年的攻击语料说明这不是假设：Tainted Memories 让攻击者通过 crafted page 把恶意指令绑定到 Agent memory；HashJack 把命令藏在 Agent 访问的 URL fragments 里；Perplexity Comet hijacks 一次点击就能触发。

防御图景并不舒服。OpenAI 的 preparedness 负责人把不便之处说得很直接：indirect prompt injection “is not a bug that can be fully patched.” 原因是攻击位于 Agent 的 reading-vs-acting boundary，而这个边界在架构上天然模糊——模型读到的每个 token 原则上都可能被当作 instruction。

本课会命名 attack surface，命名 benchmark landscape（BrowseComp、OSWorld、WebArena-Verified），并建模一个最小 indirect-prompt-injection 场景，让你能为 Lessons 14 和 18 的真实防御建立推理框架。

## The Concept / 概念

### The 2026 landscape, in one paragraph per system / 2026 版图：每个系统一段

**ChatGPT agent (OpenAI).** 2025 年 7 月发布。统一 Operator（browsing）和 Deep Research（multi-hour research）。Standalone Operator 于 2025 年 8 月 31 日关闭。在 BrowseComp 上以 68.9% 达到 SOTA；在 OSWorld 和 WebArena-Verified 上也有强结果。

**Claude Sonnet + Vercept (Anthropic).** Anthropic 收购 Vercept 后聚焦 computer-use capabilities。把 Claude Sonnet 在 OSWorld 上从 <15% 推到 72.5%。Claude Computer Use 作为 tool API 交付。

**Gemini 3 Pro with Browser Use (DeepMind).** Browser Use integration 交付 computer-use controls；FSF v3（2026 年 4 月，Lesson 20）专门跟踪 ML R&D domain 中的 autonomy。

**WebArena-Verified (ServiceNow, ICLR 2026).** 修复一个记录充分的问题：原 WebArena 有约 11.3% false-negative rate（实际完成的任务被标失败）。Verified release 用 human-curated success criteria 重新评分，并加入 258-task Hard subset（ICLR 2026 paper, openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena / BrowseComp、OSWorld 与 WebArena

| Benchmark | What it measures | Horizon |
|---|---|---|
| BrowseComp | 在 open web 上限时寻找特定事实 | minutes |
| OSWorld | Agent 操作完整 desktop（mouse、keyboard、shell） | tens of minutes |
| WebArena-Verified | 模拟网站中的 transactional web tasks | minutes |
| Hard subset | 带 multi-page state transitions 的 WebArena-Verified tasks | tens of minutes |

这些 benchmark 测的是不同轴。高 BrowseComp 分数说明 Agent 能找事实；不说明它能订机票。OSWorld 分数更接近“它能否操作我的桌面”。WebArena-Verified 更接近“它能否完成一个流程”。任何生产决策都需要匹配 task distribution 的 benchmark。

### The attack surface, named / 命名攻击面

1. **Indirect prompt injection / 间接提示注入。** 不可信页面内容包含指令。Agent 读取它们。Agent 执行它们。公开例子包括 2024 Kai Greshake et al.、2025 Tainted Memories paper、2026 HashJack（Cato Networks）。
2. **URL fragment / query injection / URL 片段或查询注入。** 被爬取 URL 的 `#fragment` 或 query string 中包含命令。它们不会可视渲染，但仍在 Agent context 中。
3. **Memory-binding attacks / 记忆绑定攻击。** 页面指示 Agent 写入 persistent memory（Lesson 12 覆盖 durable state）。下一 session 中，该 memory 在没有可见触发器的情况下触发 payload。
4. **CSRF-shaped attacks on authenticated sessions / 登录会话上的 CSRF 形攻击。** Tainted Memories 类：Agent 已在某处登录；攻击者页面发出 state-changing requests，Agent 带着用户 cookies 执行。
5. **One-click hijack / 一键劫持。** 视觉上无害的按钮携带 Agent 会跟随的 payload。Comet 类。
6. **Content-Security-Policy holes in the agent's host surface / Agent host surface 中的 CSP 漏洞。** Rendering 和 tool layers 本身也可能成为 attack vectors；browser-in-a-browser-agent stack 很宽。

### Why "not fully patchable" / 为什么“无法完全 patch”

攻击与 Agent 能力同构。Agent 必须读取不可信内容才能完成工作。Agent 读取的任何内容都可能包含指令。Agent 遵循的任何指令都可能偏离用户真实请求。防御（trust boundaries、classifiers、tool allowlists、HITL on consequential actions）会提高攻击成本并降低 blast radius，但不会关闭这个类别。

这与 Lob's theorem（Lesson 8）是同一种推理模式：Agent 无法证明下一个 token 一定安全；它只能建立一个让 unsafe tokens 更可检测的系统。

### Defense posture that actually ships / 能实际交付的防御姿态

- **Read / write boundary / 读写边界。** Reading 永远不产生后果。Writing（提交表单、发布内容、调用有副作用 tool）如果由 trust boundary 外部内容发起，就需要 fresh human approval。
- **Tool allowlist per task / 每任务 tool allowlist。** Agent 可以浏览；除非为该任务显式启用，否则不能发起 wire transfer。Lesson 13 覆盖 budgets。
- **Session isolation / 会话隔离。** Browser agent sessions 使用 scoped credentials 运行。没有 production auth，没有 personal email。保留每个 HTTP request 的日志用于 audit。
- **Content sanitizer / 内容清洗器。** Fetched HTML 在拼接进 model context 前会剥离 known-bad patterns。（能减少简单攻击；不能阻止复杂 payloads。）
- **HITL on consequential actions / 有后果动作上的 HITL。** Propose-then-commit pattern（Lesson 15）。
- **Canary tokens on memory / 记忆中的 canary tokens。** 如果 memory entry 触发，用户会看到它（Lesson 14）。

## Build It / 动手构建

本课构建一个极小 browser-agent attack-surface model：三个合成页面分别代表 benign page、visible prompt injection 和 invisible URL-fragment injection。通过对比 naive agent、read/write boundary 与 sanitizer，观察防御各自能抓住什么。

## Use It / 应用它

`code/main.py` 建模一个 tiny browser-agent run，面对三个 synthetic pages。一个页面 benign，一个在 visible text 中有 direct prompt-injection blob，一个有 URL-fragment injection（不可见但在 Agent context 内）。脚本展示：（a）naive agent 会做什么，（b）read/write boundary 抓住什么，（c）sanitizer 抓住什么，（d）两者都抓不到什么。

## Ship It / 交付它

`outputs/skill-browser-agent-trust-boundary.md` 为 proposed browser-agent deployment 划定 scope：它触达哪些 trust zones，被授权写什么，以及首次运行前必须具备哪些 defenses。

## Exercises / 练习

1. 运行 `code/main.py`。识别 sanitizer 能抓住但 read/write boundary 抓不住的攻击，以及只有 read/write boundary 能抓住的攻击。

2. 扩展 sanitizer，检测一种 HashJack-style URL-fragment injection。用带 legitimate fragments 的 benign URLs 测量 false-positive rate。

3. 选择一个你熟悉的真实 browser-agent workflow（例如 “book a flight”）。列出所有 read 和 write。标记哪些 write 需要 HITL，并说明原因。

4. 阅读 WebArena-Verified ICLR 2026 paper。找出原 WebArena scoring 不可靠的一类任务，并解释 Verified subset 如何解决。

5. 为 browser-agent setting 设计一个 memory canary。你会存什么、存在哪里、什么触发 alarm？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Indirect prompt injection | “坏网页文本” | Agent 读取的页面中存在不可信指令，并被 Agent 执行 |
| Tainted Memories | “Memory attack” | Agent 把攻击者提供的指令写入 durable memory；下一 session 触发 |
| HashJack | “URL fragment attack” | Payload 藏在 URL fragment / query string 中，在 Agent context 内但不渲染 |
| One-click hijack | “坏按钮” | 可见 affordance 携带 Agent 会执行的 follow-on payload |
| BrowseComp | “Web search benchmark” | 在 open web 上找特定事实；分钟级 horizon |
| OSWorld | “Desktop benchmark” | 完整 OS 控制；multi-step GUI tasks |
| WebArena-Verified | “修正后的 web-task benchmark” | ServiceNow 重新评分的 WebArena，带 Hard subset |
| Read/write boundary | “Side-effect gate” | Reading 不产生后果；如果内容来自 out-of-trust，writing 需要 fresh approval |

## Further Reading / 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 与 deep research 合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator lineage，以及后来成为 ChatGPT agent 的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始 benchmark。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 fixed-subset paper。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含 computer-use agents 的 attack-surface discussion。
