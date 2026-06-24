# AI Scientist v2 — Workshop-Level Autonomous Research / Workshop 级自主研究

> Sakana 的 AI Scientist v2（Yamada et al., arXiv:2504.08066）运行完整研究循环：hypothesis、code、experiments、figures、writeup、submission。它是第一个让生成论文通过 ICLR 2025 workshop peer review 的系统。独立评估（Beel et al.）发现，42% 的实验因为 coding errors 失败，literature review 也经常把已有概念误标成 novel。Sakana 自己的文档警告该 codebase 会执行 LLM-written code，并建议使用 Docker isolation。这个图景的两半都很重要。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, research-loop state-machine toy)
**Prerequisites / 前置知识：** Phase 15 · 03 (AlphaEvolve), Phase 15 · 04 (DGM)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 AI Scientist v2 从 idea 到 submission 的研究循环。
- 区分 workshop acceptance 作为 proof of concept 与可靠性声明之间的差异。
- 解释 coding errors、novelty mislabeling 和 polish masking 如何让研究 Agent 产生误导性输出。
- 为执行 LLM-written code 的研究系统设计 sandbox controls。
- 设计 human-review protocol，避免只被漂亮图表和论文表面质量迷惑。

## The Problem / 问题

研究是开放式任务。不同于 AlphaEvolve 的算法搜索或 DGM 的 benchmark-bounded self-modification，一个研究结果没有 machine-checkable correctness criterion。论文由 reviewers 判断，而不是 unit tests。这让循环更难闭合；但如果能闭合，也更有价值，因为研究正是进展复利发生的地方。

AI Scientist v1（Sakana, 2024）通过人类编写的 templates 闭合循环。LLM 在固定 scaffolding 内填充实验。AI Scientist v2（Yamada et al., 2025）通过 agentic tree search 和 vision-language model critique loop 移除了 template requirement。系统会生成 idea、实现实验、制作 figures、撰写论文，并基于 reviewer feedback 迭代。

Peer review verdict：一篇 v2 生成的论文在披露来源的前提下被 ICLR 2025 workshop 接收。Independent evaluation verdict：系统离可靠还很远。两者都是真的。

## The Concept / 概念

### The architecture / 架构

1. **Idea generation / 想法生成。** LLM 基于 topic 和 prior literature 提出研究 idea。v1 使用 templates；v2 在 hypothesis space 上使用 agentic search。
2. **Novelty check / 新颖性检查。** Literature retrieval step 检查 idea 是否已经发表。Beel et al. 的评估正是在这里发现 mislabeling：已有方法经常被判为 novel。
3. **Experiment plan / 实验计划。** Agent 起草 experimental protocol 并写代码。
4. **Execution / 执行。** 代码在 sandbox 中运行。失败会进入 retry loop。Beel et al. 的测量中，42% 的实验在这个阶段因为 coding errors 失败。
5. **Figure generation / 图表生成。** Vision-language model 阅读生成图表，并重写它们以提升清晰度。这是 v2 的关键技术新增项。
6. **Writeup / 写作。** LLM 起草论文，并与内部 reviewer 迭代。
7. **Optional: submission / 可选：投稿。** 论文提交到 venue。

### What the workshop-acceptance result means / workshop acceptance 代表什么

一篇 v2 生成的论文通过了 ICLR 2025 workshop peer review。作者向 program committee 披露了论文来源。这个 acceptance 是一个数据点；它不是系统“能做研究”的许可证。

重要背景是：workshop paper 的门槛低于 main-conference paper。Peer review 有噪声；任意一天都可能有一小部分 submission 被接受。一次成功是 proof of concept，不是 reliability claim。2026 年的 Nature paper 记录了端到端循环，且本身由人类研究者共同署名；它并不等于“系统写出了一篇 Nature 论文”。

### What the independent evaluation found / 独立评估发现了什么

Beel et al.（arXiv:2502.14297）做了外部评估。核心发现：

- **Experiment failures / 实验失败。** 42% 的实验因为 coding errors 失败（bad imports、shape mismatches、undefined variables）。Retry loop 捕捉了一部分，但不是全部。
- **Novelty mislabeling / 新颖性误标。** Literature-retrieval step 经常把已有概念标成 novel。这是研究场景里的 hallucination。
- **Presentation-quality gap / 表达质量差距。** Vision-language figure critique 会产出接近 publication-grade 的视觉效果，从而掩盖底层实验弱点。

最后一点对本阶段最重要。一个能产出 convincing outputs 却没有做 convincing research 的系统，比明显失败的系统更危险，而不是更安全。Evaluation 必须触达底层 claim，不能停在 figure。

### The sandbox-escape concern / sandbox 逃逸风险

Sakana 自己的 repository README 警告：这类软件会执行 LLM-generated code，因此不能保证安全；风险包括危险 package、不可控 web access、意外进程创建。使用者应自担风险，并考虑 Docker isolation。

这就是未验证领域中自主系统的操作形态。LLM 写代码；代码运行；代码可以做该 process 被允许做的任何事。没有对 filesystem、network 和 process actions 做硬限制的 sandbox，任何 self-directed research agent 都可能 exfiltrate data、烧掉 compute，或重写自己。

AlphaEvolve 的 sandbox story 更简单，因为它的 evaluator 很紧。AI Scientist v2 的 loop 会运行 open-ended code，目标也是 open-ended。所以它需要更强隔离（Docker 是最低线；seccomp / gVisor 更好），并且每个 submission 离开系统前都需要人工审查。

### Where v2 sits in the frontier stack / v2 在前沿技术栈中的位置

| System | Target | Output kind | Evaluator | Known failure |
|---|---|---|---|---|
| AlphaEvolve | algorithms | code | unit + benchmark | bounded by evaluator rigor |
| DGM | agent scaffolding | code | SWE-bench | reward hacking |
| AI Scientist v2 | research papers | text + code + figures | peer review (weak) | experiment failures, mislabeling, polish masking weakness |

三者中，v2 的 automatic evaluator 最弱，output surface 最宽，通向公开 artifact 的路径最短。真正承担多数安全工作的，是 sandbox、review 和 disclosure 这些 operational controls。

## Build It / 动手构建

本课用 state machine 模拟研究 Agent 循环：idea、novelty check、experiment、figure、writeup、review、accept-or-iterate。每个阶段都有可配置失败率，帮助你量化“漂亮但有缺陷”的输出比例。

## Use It / 应用它

`code/main.py` 把 v2 loop 模拟成一个 state machine：idea → novelty check → experiment → figure → writeup → review → accept-or-iterate。每个状态都有来自 Beel et al. findings 的可配置 failure probability。运行 N 次循环并统计：

- 多少 idea 抵达 submission。
- 多少 submission 带有被 polished paper 掩盖的 critical experimental flaw。
- retry budgets 如何在质量与产出率之间 trade off。

## Ship It / 交付它

`outputs/skill-ai-scientist-sandbox-review.md` 是一个 two-gate review checklist，用于任何 research-loop agent 产物离开 sandbox 前的审查。

## Exercises / 练习

1. 用默认参数运行 `code/main.py`。有多少比例的 loop runs 产出 “clean” paper？有多少比例产出带 experiment-failure flaw、但被 figure critique 修饰过的 paper？

2. 默认值已经使用 Beel et al. 的 42% / 25%。用 `--experiment-failure 0.20 --novelty-mislabel 0.10` 重跑，再用 `--experiment-failure 0.60 --novelty-mislabel 0.40` 重跑。polished-but-flawed share 如何变化？

3. 阅读 Sakana 的 AI Scientist v2 repo README 中关于 sandbox requirements 的内容。说出两个 Docker 之外、你会为 multi-day autonomous run 加上的额外限制。

4. 阅读 Beel et al. 第 4 节关于 presentation-quality gap 的内容。设计一个额外 evaluator，用来抓住外观精美但实验有缺陷的 paper。

5. 提出一个 research-agent outputs 的 human-review protocol，使其比 “a PhD reads every paper” 更可扩展。指出 bottleneck，并围绕它设计。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| AI Scientist v1 | “Sakana 的 templated research agent” | 把实验填进固定 scaffold |
| AI Scientist v2 | “Template-free research agent” | 带 VLM figure critique 的 agentic tree search |
| Agentic tree search | “分支式研究 Agent” | 并行扩展多个 experiment plans，由 internal critic 剪枝 |
| Vision-language critique | “VLM 修饰图表” | Multimodal model 阅读 figures 并重写以提升清晰度 |
| Literature retrieval | “Novelty check” | 搜索 prior work 确认 idea novelty；已有误标记录 |
| Polish masking | “漂亮论文，坏研究” | 表达质量超过实验质量，从而隐藏弱点 |
| Sandbox escape | “LLM code 跑出边界” | Agent-executed code 做了 loop designer 未授权的事 |

## Further Reading / 延伸阅读

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — 论文。
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — vendor summary，含 peer-review context。
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — 外部评估数字。
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — templated predecessor。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — open-ended research agents 的更广阔 framing。
