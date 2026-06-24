# Benchmarks: SWE-bench, GAIA, AgentBench / Benchmarks：SWE-bench、GAIA、AgentBench

> 2026 年 Agent 评测有三个锚点。SWE-bench 测代码 patching。GAIA 测 generalist tool use。AgentBench 测 multi-environment reasoning。你需要理解它们的组成、contamination story，以及它们不衡量什么。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 06 (Tool Use)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出 SWE-bench 的 test harness（FAIL_TO_PASS），并解释为什么它用 unit tests 作为 gate。
- 解释 SWE-bench Verified（OpenAI，500 tasks）为什么存在，以及它去掉了什么。
- 描述 GAIA 的设计：对人简单，对 AI 难；三种 difficulty levels。
- 说出 AgentBench 的八个 environments，以及它对 open-source LLMs 的主要阻碍。
- 总结 SWE-bench+ 的 contamination finding 及其含义。

## The Problem / 问题

leaderboard 会告诉你哪个模型在一个 benchmark 上赢了。它不会告诉你：

- benchmark 是否 contaminated（solutions 在训练数据中，test leakage）。
- benchmark 是否衡量了你关心的东西（code vs browsing vs generalist）。
- evaluator 是否稳健（AST matching、state checks、human review）。

引用数字前，先理解这三个锚点 benchmark 和它们的 failure modes。

## The Concept / 概念

### SWE-bench (Jimenez et al., ICLR 2024 oral) / SWE-bench

- 2294 个来自 12 个流行 Python repos 的真实 GitHub issues。
- Agent 获得：pre-fix commit 上的 codebase + 自然语言 issue description。
- Agent 产出：patch。
- Evaluator：应用 patch，运行 repo 的 test suite。patch 必须让 FAIL_TO_PASS tests（之前失败，现在通过）通过，同时不能破坏 PASS_TO_PASS tests。

SWE-agent (Yang et al., 2024) 发布时达到 12.5%，关键在 agent-computer interfaces（模型能理解的 file editor commands、search syntax）。

### SWE-bench Verified / SWE-bench Verified

OpenAI，2024 年 8 月。人工筛选的 500-task subset。去掉 ambiguous issues、不可靠 tests、以及 fix 不清楚的 tasks。它是 “你的 Agent 是否能交付真实 patches” 的主要 benchmark。

### Contamination / 污染

- 超过 94% 的 SWE-bench issues 早于多数 model cutoffs。
- **SWE-bench+** 发现 32.67% successful patches 在 issue text 中泄露了解法（模型在 description 里看到了 fix），31.08% 因弱 test coverage 可疑。
- Verified 更干净，但也不是完全无 contamination。

实际含义：一个在 SWE-bench 上 50% 的模型，可能在 SWE-bench+ 上只有 35%。如果声称 SWE-bench performance，尽量同时报告两者。

### GAIA (Mialon et al., Nov 2023) / GAIA

- 466 个问题；300 个保留给 huggingface.co/gaia-benchmark 的 private leaderboard。
- 设计理念：“conceptually simple for humans (92%) but hard for AI (GPT-4 with plugins: 15%).”
- 测 reasoning、multi-modality、web、tool use。
- 三个 difficulty levels；Level 3 需要跨 modalities 的长 tool chains。

GAIA 用来测 “generalist capability”。不要把它和代码专用 benchmark 混淆。

### AgentBench (Liu et al., ICLR 2024) / AgentBench

- 8 个 environments，覆盖 code（Bash、DB、KG）、games（Alfworld、LTP）、web（WebShop、Mind2Web）和 open-ended generation。
- Multi-turn，每个 split 约 4k-13k turns。
- 主要发现：long-term reasoning、decision-making 和 instruction following 是 OSS LLMs 追上商业模型的阻碍。

### What these do not measure / 它们不衡量什么

- 真实 operational cost（tokens、wall-clock）。
- 对 adversarial conditions 的 safety behavior。
- 你的 domain 的性能（用你自己的 evals，Lesson 30）。
- Tail failures（benchmark 看平均值；生产 operator 关心最差 1%）。

### Where benchmarking goes wrong / benchmark 常见误用

- **Single-number fixation。** SWE-bench 50% 不如 P50/P75/P95 cost + step distribution 信息量大。
- **Contaminated claims。** 报 SWE-bench 却不提 Verified 或 SWE-bench+，会误导。
- **Benchmark-as-development-target。** 针对 benchmark 优化会偏离生产有用性。

## Build It / 动手构建

`code/main.py` 实现一个 toy SWE-bench-like harness：

- Synthetic bug-fix tasks（3 个 tasks）。
- 脚本化 “agent” 产出 patches。
- test runner 检查 FAIL_TO_PASS（bug 已修复）和 PASS_TO_PASS（没有破坏）。
- 基于 question decomposition depth 的 GAIA-style difficulty classifier。

运行：

```
python3 code/main.py
```

输出会展示每个 task 和每个 difficulty 的 resolution rate，并让 evaluator rules 具体化。

## Use It / 应用它

- **SWE-bench Verified** 用于 code agents。总是报告 Verified scores。
- **GAIA** 用于 generalist agents。使用 private leaderboard split。
- **AgentBench** 用于 multi-environment comparison。
- **Custom evals**（Lesson 30）用于你的产品真实形状。

## Ship It / 交付它

`outputs/skill-benchmark-harness.md` 会为任意 codebase-task pair 构建 SWE-bench-style harness，并用 FAIL_TO_PASS / PASS_TO_PASS gating。

## Exercises / 练习

1. 把 toy harness 移植到真实 repo（选你的一个）。为已知 bugs 写 3 个 FAIL_TO_PASS tests。
2. 增加 step-count metric。你的 3 个 tasks 每个 resolution 要多少 Agent steps？
3. 阅读 SWE-bench+ paper。实现 solution-leakage check（用 pattern-match 比较 issue text 和 diff）。
4. 从 public split 下载一个 GAIA question。trace 一个 GPT-4-class Agent 会做什么。它需要哪些 tools？
5. 阅读 AgentBench 的 per-environment breakdown。哪个 environment 像你的 product surface？那里 “SOTA” 长什么样？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| SWE-bench | “Code agent benchmark” | 2294 个 GitHub issues；patch 必须翻转 FAIL_TO_PASS tests |
| SWE-bench Verified | “Clean SWE-bench” | OpenAI 人工筛选的 500 tasks |
| FAIL_TO_PASS | “Fix gate” | patch 后必须通过的、之前失败的 tests |
| PASS_TO_PASS | “No-regression gate” | patch 前通过、patch 后仍必须通过的 tests |
| GAIA | “Generalist benchmark” | 466 个对人简单、对 AI 难的 multi-tool questions |
| AgentBench | “Multi-env benchmark” | 8 个 environments；long-horizon multi-turn |
| Contamination | “Training-set leak” | benchmark tasks 出现在模型训练中 |
| SWE-bench+ | “Contamination audit” | successful SWE-bench patches 中发现 32.67% solution leakage |

## Further Reading / 延伸阅读

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — 原始 benchmark
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated subset
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — generalist benchmark
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — multi-environment suite
