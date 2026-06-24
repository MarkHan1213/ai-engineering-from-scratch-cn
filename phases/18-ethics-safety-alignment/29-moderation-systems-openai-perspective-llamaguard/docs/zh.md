# Moderation Systems — OpenAI, Perspective, Llama Guard / Moderation Systems：OpenAI、Perspective、Llama Guard

> 生产 moderation systems 把 Lessons 12-16 中定义的 safety policies operationalize。OpenAI Moderation API：`omni-moderation-latest`（2024）基于 GPT-4o，一次 call 同时分类 text + images；在 multilingual test set 上比上一代好 42%；response schema 返回 13 个 category booleans——harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；对大多数 developers 免费。Layered patterns：Input moderation（pre-generation）、Output moderation（post-generation）、Custom moderation（domain rules）。Async parallel calls 隐藏 latency；flag 时可返回 placeholder responses。Llama Guard 3/4（Lesson 16）：14 个 MLCommons hazards、Code Interpreter Abuse、8 languages（v3）、multi-image（v4）。Perspective API（Google Jigsaw）：早于 LLM-as-moderator wave 的 toxicity scoring；主要是 single-dimension toxicity，带 severe-toxicity/insult/profanity variants；仍是 content-moderation research baseline。Deprecations：Azure Content Moderator 2024 年 2 月 deprecated，2027 年 2 月 retired，由 Azure AI Content Safety 替代。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, three-layer moderation harness)
**Prerequisites / 前置知识：** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 OpenAI Moderation API 的 category taxonomy，以及它与 Llama Guard 3 的 MLCommons set 有何不同。
- 描述三层 moderation pattern（input、output、custom），并说出每层一个 failure mode。
- 描述 Perspective API 作为 pre-LLM-era baseline 的位置，以及它为什么仍用于研究。
- 说明 Azure deprecation timeline。

## The Problem / 问题

Lessons 12-16 描述 attacks 和 defense tooling。Lesson 29 覆盖 deployed moderation systems，它们在用户接触产品的表面 operationalize defenses。三层 pattern 是 2026 年默认生产配置。

## The Concept / 概念

### OpenAI Moderation API / OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o。一次 call 分类 text + images。对大多数 developers 免费。

Categories（response schema 中 13 个 booleans）：
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

Multimodal support 适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余 text-only。

在 `code/main.py` 的 code harness 中，为了教学简化，我们会把 `/threatening`、`/intent`、`/instructions` 和 `/graphic` sub-categories 折叠到 top-level parents。Production code 应使用完整 13-category schema。

比上一代 moderation endpoint 在 multilingual test set 上好 42%。返回 per-category scores；applications 自行设 threshold。

### Llama Guard 3/4 / Llama Guard 3/4

Lesson 16 已覆盖。14 个 MLCommons hazard categories（组织方式不同于 OpenAI 的 13 response-schema booleans）。支持 8 languages（v3）。Llama Guard 4（2025 年 4 月）是 natively multimodal、12B。

OpenAI 与 Llama Guard taxonomies 有重叠也有分歧。OpenAI 有 broad “illicit”；Llama Guard 分开 “violent crimes” 和 “non-violent crimes”。Deployment 根据 policy-taxonomy fit 选择。

### Perspective API (Google Jigsaw) / Perspective API（Google Jigsaw）

早于 LLM-as-moderator wave 的 toxicity scoring system（pre-2020）。Categories：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。Primary score 是 single-dimension TOXICITY，附带 sub-dimension variants。

它仍广泛用作 content-moderation research baseline，因为 API 稳定、文档完善，并有多年 calibration data。对现代 LLM-adjacent use cases，Llama Guard 或 OpenAI Moderation 通常更适合。

### The three-layer pattern / 三层模式

1. **Input moderation。** 在 generation 前分类 user prompt。若 flagged 则 reject。Latency：一次 classifier call。
2. **Output moderation。** 在交付前分类 model output。若 flagged 则替换为 refusal。Latency：generation 后一次 classifier call。
3. **Custom moderation。** Domain-specific rules（regex、allowlists、business policy）。可在 input 或 output 侧运行。

三层按设计是 sequential：input moderation 必须在 generation 前完成，output moderation 在 generation 后运行。Parallelism 作用在同一层内部——在同一文本上并发运行多个 classifiers（例如 OpenAI Moderation + Llama Guard + Perspective），可以隐藏单个 classifier latency。作为可选优化，可以在 input moderation 完成、token-1 streaming 延迟期间展示 placeholder response（“one moment, checking...”）。Flag behaviour 可配置：refuse、sanitize、escalate to human review。

### Failure modes / 失败模式

- **Input only。** 抓不到 output hallucinations（Lesson 12-14 的 encoding attacks 会绕过 input classifiers）。
- **Output only。** 允许任意 input 到达模型；增加成本；把 internal reasoning 暴露给 attacker。
- **Custom only。** 不跨 categories robust；regexes 脆弱。

Layered 是默认。Belt-and-suspenders。

### Azure deprecation / Azure 弃用时间线

Azure Content Moderator：2024 年 2 月 deprecated，2027 年 2 月 retired。由 Azure AI Content Safety 替代，后者是 LLM-based，并与 Azure OpenAI 集成。对 Azure deployments 来说，迁移是 2024-2027 的 field-level project。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lesson 16 覆盖 red-team context 中的 moderation tooling。Lesson 29 覆盖 deployed moderation。Lesson 30 用当前 dual-use capability evidence 收束。

## Build It / 动手构建

本课构建三层 moderation harness：input moderator、output moderator、custom moderator。你会让 benign、borderline 和 harmful inputs 通过三层，观察每层 fires 的位置。

## Use It / 应用它

`code/main.py` 构建一个 three-layer moderation harness：input moderator（keyword + category score）、output moderator（对 output 使用同一 classifier）、custom moderator（domain rules）。你可以让 inputs 通过系统，并观察哪个 layer 捕获了什么。

## Ship It / 交付它

本课产出 `outputs/skill-moderation-stack.md`。给定 deployment，它会推荐 moderation stack configuration：input 用哪个 classifier、output 用哪个、custom rules 写什么，以及 edge cases 用什么 judge。

## Exercises / 练习

1. 运行 `code/main.py`。让 benign、borderline、harmful input 通过三层。报告每个由哪层触发。

2. 用 Perspective-API-style toxicity scoring 扩展 harness 的某个 category。比较它与 category score 的 threshold behaviour。

3. 阅读 OpenAI Moderation API docs 与 Llama Guard 3 category list。把每个 OpenAI category 映射到最接近的 Llama Guard categories。识别三个无法干净映射的 categories。

4. 为 code-assistant deployment（例如 GitHub Copilot）设计 moderation stack。识别最相关和最不相关的 categories，并提出 custom rules。

5. Azure Content Moderator 将于 2027 年 2 月 retired。规划迁移到 Azure AI Content Safety。识别迁移中最高风险的 element。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| OpenAI Moderation | “omni-moderation-latest” | 基于 GPT-4o 的 13-category（text）classifier，部分 multimodal support |
| Perspective API | “Google Jigsaw toxicity” | pre-LLM-era toxicity scoring baseline |
| Llama Guard | “MLCommons 14-category” | Meta hazard classifier（v3：8B text, 8 langs；v4：12B multimodal） |
| Input moderation | “pre-generation filter” | model call 前对 user prompt 做 classifier |
| Output moderation | “post-generation filter” | delivery 前对 model output 做 classifier |
| Custom moderation | “domain rules” | deployment-specific rules（regex、allowlist、policy） |
| Layered moderation | “all three layers” | 标准 production deployment pattern |

## Further Reading / 延伸阅读

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation endpoint。
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard repo。
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — toxicity scoring。
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure replacement。
