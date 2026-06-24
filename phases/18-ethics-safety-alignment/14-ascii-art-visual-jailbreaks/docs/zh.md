# ASCII Art and Visual Jailbreaks / ASCII Art 与视觉越狱

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran，“ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs”（ACL 2024，arXiv:2402.11753）。把 harmful request 中 safety-relevant tokens 遮住，用同一字母的 ASCII-art rendering 替换，再发送 cloaked prompt。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法 robustly recognize ASCII-art tokens。该攻击绕过 PPL（perplexity filters）、Paraphrase defenses 和 Retokenization。相关工作：ViTC benchmark 测量 non-semantic visual prompts 的识别能力；StructuralSleight 将其推广到 Uncommon Text-Encoded Structures（trees、graphs、nested JSON）这一类 encoding attacks。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites / 前置知识：** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 ArtPrompt attack：word-identification step、ASCII-art substitution、final cloaked prompt。
- 解释为什么标准 defenses（PPL、Paraphrase、Retokenization）在 ArtPrompt 上失败。
- 定义 ViTC，并描述它测量什么。
- 描述 StructuralSleight 如何推广到任意 Uncommon Text-Encoded Structures。

## The Problem / 问题

通过 paraphrase 和 roleplay 的攻击（Lesson 12）以及 long context 的攻击（Lesson 13）都作用在 text-level pattern。ArtPrompt 作用在 recognition level：模型没有解析 forbidden token，而是在字符中解析出一张图。Safety filter 看到的是 harmless punctuation。模型看到的是一个词。

## The Concept / 概念

### ArtPrompt, two steps / ArtPrompt 两步

Step 1. Word Identification。给定 harmful request，attacker 用 LLM 找出 safety-relevant words（例如 “how to make a bomb” 中的 “bomb”）。

Step 2. Cloaked Prompt Generation。把每个 identified word 替换成它的 ASCII-art rendering（由字符组成 7x5 或 7x7 的 letter shape）。模型接收的是一格 punctuation 和 spaces，足够强的模型能识别出词；safety filter 只看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 都失败。它们 benchmark subset 上 attack success rate 超过 75%。

### Why the standard defenses fail / 为什么标准防御失败

- **PPL（perplexity filter）。** ASCII art 的 perplexity 高，但所有 novel input 也高。能 block ArtPrompt 的 threshold 也会 block legitimate structured input。
- **Paraphrase。** Paraphrasing prompt 会破坏 ASCII art。但实践中 paraphrase LLMs 经常保留或重构 art。
- **Retokenization。** 换 tokenizer split 不会改变模型通过视觉识别 letter shapes 这件事。

根本问题是：safety filters 在 token 或 semantic level；ArtPrompt 在 visual recognition level。

### ViTC benchmark / ViTC benchmark

识别 non-semantic visual prompts。测量模型读取 ASCII-art、wingdings 和其他 non-text-semantic visual content 的能力。ArtPrompt 有效性与 ViTC accuracy 相关：模型越会读 visual text，ArtPrompt 对它越有效。这是 capability-safety tradeoff。

### StructuralSleight / StructuralSleight

把 ArtPrompt 推广到 Uncommon Text-Encoded Structures（UTES）。Trees、graphs、nested JSON、CSV-in-JSON、diff-style code blocks。如果某种结构在 training safety data 中少见，但模型可 parse，就可能藏入 harmful content。

防御含义：safety 必须泛化到模型能 parse 的 structured representations。这个集合很大，而且还在增长。

### Image-modality analog / 图像模态类比

Visual LLMs（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了 attack surface。实际 images 上的 ArtPrompt-style attacks 比 ASCII-art analogs 更强，因为 image encoders 提供更丰富信号。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lessons 12-14 描述三条正交攻击向量：iterative refinement（PAIR）、context length（MSJ）、encoding（ArtPrompt/StructuralSleight）。Lesson 15 从 model-centric attacks 转向 system-boundary attacks（indirect prompt injection）。Lesson 16 描述 defensive tooling response。

## Build It / 动手构建

本课构建一个 ArtPrompt token-masking harness：把目标词 cloaked 成 ASCII-art glyphs，检查 keyword filter 是否被绕过，再用 simple recognizer 尝试还原。

## Use It / 应用它

`code/main.py` 构建一个 toy ArtPrompt。你可以用 ASCII-art glyphs cloak harmful query 中的特定 words，验证 cloaked string 能通过 keyword filter，并且（可选）用 simple recognizer 把 cloaked string 解码回来。

## Ship It / 交付它

本课产出 `outputs/skill-encoding-audit.md`。给定 jailbreak-defense report，它会枚举 covered encoding attack families（ASCII art、base64、leet-speak、UTF-8 homoglyph、UTES）以及每种由哪个 defense layer 捕获。

## Exercises / 练习

1. 运行 `code/main.py`。验证 cloaked string 通过 simple keyword filter。报告所需 character-level change。

2. 实现第二种 encoding：同一 target word 的 base64。比较它与 ArtPrompt 的 filter-bypass rate 和 recovery difficulty。

3. 阅读 Jiang et al. 2024 Section 4.3（five-model results）。提出一个理由，说明为什么 Claude 在同一 benchmark 上的 ArtPrompt-resistance 高于 Gemini。

4. 设计一个 pre-generation defense，检测 prompt 中 ASCII-art-shaped regions。测量 legitimate code、tables 和 mathematical notation 上的 false-positive rate。

5. StructuralSleight 列出 10 种 encoding structures。Sketch 一个处理全部 10 种的 generalized defense，并估计每个 defended prompt 的 compute cost。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| ArtPrompt | “the ASCII-art attack” | 用 ASCII-art renderings 遮蔽 safety words 的两步 jailbreak |
| Cloaking | “hide the word” | 把 forbidden token 替换成模型能读但 filter 读不到的 visual representation |
| UTES | “uncommon structure” | Uncommon Text-Encoded Structure：tree、graph、nested JSON 等用于夹带内容的结构 |
| ViTC | “visual-text capability” | 衡量模型读取 non-semantic visual encoding 能力的 benchmark |
| Perplexity filter | “PPL defense” | 拒绝 high perplexity prompts；因 legitimate structured input 也高而失败 |
| Retokenization | “tokenizer shift defense” | 用不同 tokenizer 预处理 prompt；因识别是 visual 而失败 |
| Homoglyph | “lookalike characters” | 看起来像 Latin letters 的 Unicode characters，可绕过 substring checks |

## Further Reading / 延伸阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII-art jailbreak paper。
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES generalization。
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — complementary iterative attack。
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — complementary length attack。
