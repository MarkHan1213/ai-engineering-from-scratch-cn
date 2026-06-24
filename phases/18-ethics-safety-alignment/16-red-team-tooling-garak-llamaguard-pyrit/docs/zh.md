# Red-Team Tooling — Garak, Llama Guard, PyRIT / 红队工具：Garak、Llama Guard、PyRIT

> 三个生产工具定义了 2026 年 red-team stack。Llama Guard（Meta）——一个在 14 个 MLCommons hazard categories 上 fine-tuned 的 Llama-3.1-8B classifier；2025 年 Llama Guard 4 是从 Llama 4 Scout pruning 而来的 12B natively multimodal classifier。Garak（NVIDIA）——开源 LLM vulnerability scanner，带 static、dynamic 和 adaptive probes，覆盖 hallucination、data leakage、prompt injection、toxicity、jailbreaks。PyRIT（Microsoft）——multi-turn red-team campaigns，支持 Crescendo、TAP 和 custom converter chains，用于 deep exploitation。Llama Guard 3 记录在 Meta “Llama 3 Herd of Models”（arXiv:2407.21783）；Llama Guard 3-1B-INT4 见 arXiv:2411.17713；Garak 的 probe architecture 在 github.com/NVIDIA/garak。这些工具是 2026 年 red-team research（Lessons 12-15）与 deployment（Lesson 17+）之间的生产接口。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Prerequisites / 前置知识：** Phase 18 · 12-15 (jailbreaks and IPI)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 描述 Llama Guard 3/4 在 safety stack 中的位置：input classifier、output classifier，或二者皆有。
- 说出 14 个 MLCommons hazard categories，并指出一个不那么显然的类别（Code Interpreter Abuse）。
- 描述 Garak 的 probe architecture：probes、detectors、harnesses。
- 描述 PyRIT 的 multi-turn campaign structure，以及它如何与 Garak probes 组合。

## The Problem / 问题

Lessons 12-15 呈现 attack surface。Production deployments 需要可重复、可扩展的 evaluation。2026 年主导的三个工具是：Llama Guard（defense classifier）、Garak（scanner）、PyRIT（campaign orchestrator）。它们分别针对 red-team lifecycle 的不同层。

## The Concept / 概念

### Llama Guard (Meta) / Llama Guard（Meta）

Llama Guard 3 是一个 Llama-3.1-8B model，针对 MLCommons AILuminate 14 categories 做 input/output classification fine-tuning：
- Violent crimes, non-violent crimes, sex-related, CSAM, defamation
- Specialized advice, privacy, IP, indiscriminate weapons, hate
- Suicide/self-harm, sexual content, elections, code-interpreter abuse

支持 8 种语言。用法：放在 LLM 前（input moderation）、LLM 后（output moderation），或两边都放。两种用法产生不同 training distributions；Llama Guard 3 作为单一模型同时处理二者。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，mobile CPU 上约 30 tokens/s）是 quantized edge variant。

Llama Guard 4（2025 年 4 月）为 12B，natively multimodal，从 Llama 4 Scout pruning 而来。它用一个 classifier 取代 8B text 和 11B vision 两个前代。

### Garak (NVIDIA) / Garak（NVIDIA）

开源 vulnerability scanner。架构：
- **Probes。** 为 hallucination、data leakage、prompt injection、toxicity、jailbreaks 生成 attacks。Static（固定 prompts）、dynamic（生成 prompts）、adaptive（根据 target output 响应）。
- **Detectors。** 根据 expected failure modes 对 outputs 打分：toxic、leaked、jailbroken。
- **Harnesses。** 管理 probe-detector pairs，运行 campaigns，生成 reports。

TrustyAI 将 Garak 与 Llama-Stack shields（Prompt-Guard-86M input classifier、Llama-Guard-3-8B output classifier）集成，用于 end-to-end shielded-target evaluation。Tier-based scoring（TBSA）取代二元 pass/fail：同一 probe 上，模型可能在 severity tier 3 通过、tier 5 失败。

### PyRIT (Microsoft) / PyRIT（Microsoft）

Python Risk Identification Toolkit。用于 multi-turn red-team campaigns。核心组件：
- **Converters。** 转换 seed prompt：paraphrase、encode、translate、roleplay。
- **Orchestrators。** 运行 campaign：Crescendo（escalation）、TAP（branching）、RedTeaming（custom loop）。
- **Scoring。** LLM-as-judge 或 classifier-as-judge。

PyRIT 是 Garak 的更重 cousin。Garak 跑成千上万 single-turn probes；PyRIT 跑深层 multi-turn campaigns，用于击穿特定 failure modes。

### The stack / 工具栈

把 Llama Guard 放在模型前后。每晚跑 Garak 做 regression。发布前跑 PyRIT campaigns。这是大多数 production deployments 的 2026 默认配置。

### Evaluation pitfalls / 评估陷阱

- **Judge identity。** 三个工具都可以用 LLM judge；judge calibration 会决定 reported ASRs（Lesson 12）。报告工具时必须同时指定 judge。
- **Probe staleness。** Garak probes 会随着模型被 patch 而老化。Adaptive probes（PAIR-shaped）比 static probes 老得慢。
- **Llama Guard FPR on benign content。** 早期 Llama Guard versions 对 political 和 LGBTQ+ content 过度 flag；Llama Guard 3/4 calibration 已改善，但仍不是按每个 deployment 校准。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lessons 12-15 是 attack families。Lesson 16 是 production tooling。Lesson 17（WMDP）是 dual-use capability evaluation。Lesson 18 是把这些工具包进 policy structure 的 frontier safety frameworks。

## Build It / 动手构建

本课实现一个 toy Llama Guard-style classifier、Garak harness 和 PyRIT-style converter chain。你会比较 single-turn scanning 与 multi-turn campaign 在覆盖面上的差异。

## Use It / 应用它

`code/main.py` 构建一个 toy Llama Guard-style classifier（14 categories 上的 keyword + semantic features）、一个 toy Garak harness（probe-detector loop）和一个 PyRIT-style multi-turn converter chain。你可以把三个工具跑在 mock target 上，观察不同 coverage signatures。

## Ship It / 交付它

本课产出 `outputs/skill-red-team-stack.md`。给定 deployment description，它会说明三个工具中哪些适合、每个要配置什么，以及 regression cadence 应该如何运行。

## Exercises / 练习

1. 运行 `code/main.py`。比较 Llama-Guard-style classifier 在 single-turn 与 multi-turn attacks 上的 detection rate。

2. 实现一个新的 Garak probe：base64-encoded harmful request。测量 Llama-Guard-style classifier 对它的 detection。

3. 在 PyRIT-style converter chain 中加入 “translate to French, then paraphrase” converter。重新测量 attack success。

4. 阅读 Llama Guard 3 的 hazard-category list。识别两个在 legitimate developer content 上 realistically 会产生 high false-positive rates 的 categories。

5. 比较 Garak 与 PyRIT 的设计原则。论证一个 deployment 中各自何时是正确工具。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Llama Guard | “the classifier” | 带 14 hazard categories 的 fine-tuned Llama-3.1-8B/4-12B safety classifier |
| Garak | “the scanner” | NVIDIA 开源 vulnerability scanner；probes、detectors、harnesses |
| PyRIT | “the campaign tool” | Microsoft multi-turn red-team orchestrator；converters、orchestrators、scoring |
| Prompt-Guard | “the small classifier” | Meta 的 86M prompt-injection classifier，与 Llama Guard 配套 |
| TBSA | “tier-based scoring” | Garak 的 tier-based pass/fail，用于替代 binary outcomes |
| Converter chain | “paraphrase + encode + ...” | PyRIT 用于构造 multi-step attacks 的 composition primitive |
| MLCommons hazard categories | “the 14 taxonomies” | Llama Guard 面向的 industry-standard taxonomy |

## Further Reading / 延伸阅读

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B classifier。
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — quantized mobile classifier。
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — scanner repo 与 documentation。
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — campaign toolkit。
