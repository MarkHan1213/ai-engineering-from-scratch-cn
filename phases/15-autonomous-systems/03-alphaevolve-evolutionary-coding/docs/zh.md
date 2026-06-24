# AlphaEvolve — Evolutionary Coding Agents / 演化式编码 Agent

> 把前沿 coding model、evolutionary loop 和 machine-checkable evaluator 配在一起，让循环运行足够久。它会发现一个只用 48 次标量乘法的 4x4 complex-matrix multiplication procedure——这是 Strassen 56 年来首次被改进。它还找到一个用于 Google-wide Borg scheduling 的 heuristic，在生产中回收约 0.7% cluster compute。这个架构故意很朴素。真正的收益来自 evaluator 的严谨性。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, evolutionary-loop toy)
**Prerequisites / 前置知识：** Phase 15 · 01 (long-horizon framing), Phase 15 · 02 (self-taught reasoning)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 AlphaEvolve 的 generator + evaluator + archive 循环。
- 解释为什么 machine-checkable evaluator 是演化式编码 Agent 的前提。
- 识别 reward hacking 在代码搜索循环中的典型形态。
- 说明 MAP-elites / island model 如何维持搜索多样性。
- 为一个新领域写出 evaluator signature，并加入 held-out 和 anti-hacking 检查。

## The Problem / 问题

大语言模型可以写代码。演化算法可以搜索代码。两者单独都被尝试了几十年，也都撞到过天花板。LLM 的天花板是 confabulation：模型写出看似合理但并不按声称方式工作的代码。演化算法的天花板是搜索成本：在语法上随机 mutation 很少能产出可编译程序，更不用说更好的程序。

AlphaEvolve（Novikov et al., DeepMind, arXiv:2506.13131, 2025 年 6 月）把两者结合起来。LLM 对 program database 中的程序提出定向修改；automatic evaluator 给每个 variant 打分；高分 variant 变成未来 generation 的 parent。LLM 负责写出可行代码这个昂贵步骤；evaluator 抓住 confabulations。循环会运行数小时到数周。

报告结果包括：48 次标量乘法的 4x4 complex matrix multiplication（Strassen 1969 年界限是 49）、Google 生产中的 Borg scheduling heuristic、FlashAttention kernel 32.5% speedup、Gemini training throughput 改进。

这个架构有效，是因为 evaluator 可以机器检查。evaluator 不可机器检查的地方，它就不成立。这个不对称性正是本课重点。

## The Concept / 概念

### The loop / 循环

1. 从一个正确但非最优的 seed program `P_0` 开始。
2. 维护一个 variant program database，每个 variant 都有 evaluator score。
3. 从 database 中采样一个或多个 parent（MAP-elites-style 或 island-based）。
4. Prompt LLM（大量候选用 Gemini Flash，困难候选用 Gemini Pro）对 parent 产出修改后的 variant。
5. 编译、运行，并在 held-out evaluator 上评估该 variant。
6. 按 score 和 feature vector 把它插入 database。
7. 重复。

两个细节很关键。第一，prompt 给 LLM 的不只是 parent program，通常还包括 database 中若干 top variants、evaluator signature，以及简短任务描述。模型的工作是提出一个可能提升 score 的定向修改。第二，database 是结构化的（MAP-elites grid、island-based），所以循环探索多样性，而不是只追逐当前 leader。

### What makes the evaluator non-negotiable / 为什么 evaluator 不可协商

AlphaEvolve 的成果都来自 evaluator 快速、确定、难以被钻空子的领域：

- **Matrix multiplication algorithm / 矩阵乘法算法**：unit test 会做矩阵乘法并逐 bit 检查相等。
- **Borg scheduling heuristic / Borg 调度启发式**：生产级 simulator 会 replay 历史 cluster load，并度量 wasted compute。
- **FlashAttention kernel / FlashAttention 内核**：correctness test 加真实硬件上的 wall-clock benchmark。
- **Gemini training throughput / Gemini 训练吞吐**：按每 step 的 GPU-seconds 度量。

在每个案例里，evaluator 都抓住了原本会主导结果的 LLM 错误类别：编造正确性声明、在硬件上消失的性能声明、edge-case failures。移除 evaluator，循环优化的就只是漂亮代码。

### Reward hacking is the other face of that statement / reward hacking 是同一件事的另一面

演化会优化 evaluator 测量的任何东西。如果 evaluator 不完美，循环会找到不完美之处。在未验证领域里，循环会优化表面特征，而不是意图行为。DeepMind 在论文里明确提醒：AlphaEvolve 的成功只会迁移到 evaluator 严谨性匹配搜索野心的领域。

2025-2026 年代码搜索循环中的具体 reward hacking 例子：

- 奖励 “time to complete” 的优化目标，反而奖励提交空解。
- 奖励 correctness-under-test 的 benchmark score，反而奖励记忆测试并过拟合。
- 一个 “code quality” proxy 奖励删除注释、改写变量名，但语义没有变化。

AlphaEvolve 的修复方式：交付一个 LLM 从未见过的 held-out evaluator，并在 evaluation time 生成输入。即便如此，DeepMind 仍建议对任何 proposed deployment 做强审查。

### Why LLM + search beats either alone / 为什么 LLM + search 胜过单独任何一方

LLM 可以产出可编译、语义上合理的修改。对一个 2000 行 Python 文件做 random-mutation GA，几乎总是产出语法错误。LLM 也会把搜索集中到合理邻域（修改一个函数，而不是随机字节），大幅减少浪费的 evaluator calls。

反过来，evaluator 会抓住 LLM 的 confabulations。LLM 会自信地声称一个函数 “is O(n log n) in the limit”，即使它实际是 O(n^2)；wall-clock benchmark 会把问题变成定论。

### Where AlphaEvolve fits in the frontier stack / AlphaEvolve 在前沿技术栈里的位置

| System | Generator | Evaluator | Domain | Example win |
|---|---|---|---|---|
| AlphaEvolve | Gemini | correctness + benchmark | algorithms, kernels, schedulers | 48-mul 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | correctness | combinatorial math | cap-set lower bounds |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM critique + experiment | ML research | ICLR workshop paper |
| Darwin Godel Machine (L4) | agent scaffolding | SWE-bench / Polyglot | agent code | 20% → 50% SWE-bench |

四者都是同一个 recipe 的变体：generator 加 evaluator，再加 loop。差异在于 evaluator 评什么，以及它有多严谨。

## Build It / 动手构建

本课会构建一个最小 AlphaEvolve-like loop：用 stdlib proxy 模拟 “LLM” 生成小型语法 mutation，用 held-out evaluator 给 toy symbolic-regression program 打分，再用 archive 保留多样候选。

## Use It / 应用它

`code/main.py` 在一个 toy symbolic-regression problem 上实现了最小 AlphaEvolve-like loop。“LLM” 是一个 stdlib proxy，会对计算 target function 的 program 提出小型语法 mutation。“evaluator” 会在 held-out test points 上测量 mean squared error。

观察：

- best score 如何随 generations 改进。
- MAP-elites grid 如何保留多样解，避免循环收敛到 local minimum。
- 移除 held-out test（training-only evaluator）后，循环如何严重 overfit。

## Ship It / 交付它

`outputs/skill-evaluator-rigor-audit.md` 是考虑在新领域使用 AlphaEvolve-style loop 前的前置条件：你的 evaluator 是否真的能抓住你在意的失败？

## Exercises / 练习

1. 运行 `code/main.py`。记录 best score trajectory。禁用 held-out evaluator（flag `--no-holdout`）后重跑。量化 overfitting。

2. 阅读 AlphaEvolve 论文第 3 节关于 MAP-elites grid 的内容。为一个新问题（例如 compiler optimization passes）设计 feature-vector descriptor，用来保持搜索多样性。

3. 48 次乘法的 4x4 结果在 56 年后改进了 Strassen 的 49-mul bound。阅读论文 Appendix F，用三句话解释为什么这个问题的 evaluator 特别容易做对，以及为什么多数领域并非如此。

4. 提出一个 AlphaEvolve 会失败的领域。明确指出 evaluator 在哪里坏掉，以及为什么。

5. 为你熟悉的一个领域写出 evaluator signature。包括：（a）correctness conditions，（b）performance metric，（c）held-out input generation rule，（d）至少一个 anti-reward-hacking check。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| AlphaEvolve | “DeepMind 的 evolutionary coding agent” | Gemini + program database + machine-checkable evaluator |
| MAP-elites | “保持多样性的 archive” | 由 feature vectors 键控的 grid；每个 cell 保存该 descriptor 下的最佳 variant |
| Island model | “并行演化子种群” | 独立 populations 定期迁移；防止 premature convergence |
| Machine-checkable evaluator | “确定性 oracle” | LLM 无法伪造的 unit test、simulator 或 benchmark；此循环的前提 |
| Reward hacking | “优化指标而非目标” | 循环找到最大化 score 但不完成意图任务的方法 |
| Seed program | “起点” | 循环从中演化的初始正确但非最优程序 |
| Held-out evaluator | “LLM 从未见过的 evaluation data” | evaluation time 生成输入，防止 memorization |

## Further Reading / 延伸阅读

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — 完整论文。
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — vendor writeup 和结果摘要。
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — 被发现的算法，包括 48-mul 4x4 matmul。
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — 前身系统。
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 把 evaluator-bound autonomy 作为关键研究方向来框定。
