# STaR, V-STaR, Quiet-STaR — Self-Taught Reasoning / 自举式推理

> 最小的自我改进循环可以藏在 rationale 里。模型生成 chain of thought，保留那些最终答案正确的样本，再在这些样本上微调。这就是 STaR。V-STaR 加入 verifier，让 inference-time selection 更好。Quiet-STaR 把 rationale 下沉到每个 token。三者都有效，但都不是魔法：这个循环会保留任何刚好抵达正确答案的捷径。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, bootstrap-loop simulator)
**Prerequisites / 前置知识：** Phase 13 · 01-03 (Reasoning and CoT), Phase 15 · 01 (long-horizon framing)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 STaR 如何用模型生成的 rationale 自举训练数据。
- 区分 STaR、V-STaR 和 Quiet-STaR 的训练信号、推理成本与失败模式。
- 解释 answer-conditioned gradient 为什么会强化 shortcut rationale。
- 设计 OOD held-out evaluation 来暴露自举推理中的捷径。
- 判断 verifier 何时能提升 inference-time selection，何时只是把错误包装得更自信。

## The Problem / 问题

教模型推理的直接方式，是收集人类写出的 reasoning traces。这很贵、很慢，而且受限于人类愿意写多少高质量 chain-of-thought。

STaR（Self-Taught Reasoner，Zelikman et al., 2022）提出一个问题：如果让模型自己写 rationale，再用已知答案给它打分呢？循环是：

1. 采样一条 reasoning trace 加答案。
2. 如果最终答案正确，保留这条 trace。
3. 在保留下来的 traces 上微调。
4. 重复。

这个方法有效。GSM8K 和 CommonsenseQA 都在没有新增人工标注的情况下提升了。但这个循环内置了偏差：任何得到正确答案的 rationale 都会被保留，不管推理本身是否可靠。V-STaR（Hosseini et al., 2024）用 learned verifier 修补这一点；Quiet-STaR（Zelikman et al., 2024）把这个想法推广到 per-token internal rationales。

## The Concept / 概念

### STaR: bootstrap on what worked / STaR：在有效样本上自举

从一个具备弱推理能力的 base model 开始。对每个训练问题，采样 rationale 加答案。如果答案匹配标签，就保留 `(problem, rationale, answer)` 三元组。然后在保留集合上微调模型。重复这个过程。

有一个关键 twist。如果模型永远做不对某个问题，循环就无法从它上面学习。STaR 加入 **rationalization**：对模型失败的问题，把正确答案作为 hint 注入，再重新 prompt 模型生成能导向该答案的 rationale。Rationalized rationales 也会加入训练集。

原始论文结果（Zelikman et al., 2022）：GPT-J base model 通过带 rationalization 的多轮 STaR，在 GSM8K 上从 5.8% 提升到 10.7%，绝对提升约 5 个百分点。在 CommonsenseQA 上，STaR 训练后的 GPT-J 6B 达到 72.5%，接近用手写 rationale 微调的 GPT-3 175B（约 73%）——后者模型规模大约 30 倍。

### V-STaR: train a verifier with DPO / V-STaR：用 DPO 训练 verifier

STaR 会丢弃错误 rationale。Hosseini et al.（2024）观察到，这些也是数据：每一对 `(rationale, "is this correct")` 都能训练 verifier。他们对正确和错误解法使用 Direct Preference Optimization 来构建 ranker。推理时采样 N 条 rationale，选 verifier 排名最高的一条。

报告的增益：在 GSM8K 和 MATH 上，相比此前 self-improvement baseline 提升 +4 到 +17 个百分点；大部分收益来自用 verifier 做 inference-time selection，而不是继续微调 generator。

### Quiet-STaR: per-token internal rationales / Quiet-STaR：每个 token 的内部 rationale

Zelikman et al.（2024）问了一个更细的问题：如果模型学会在每个 token 位置生成一小段 internal rationale，而不只是问题与答案之间生成 rationale，会怎样？Quiet-STaR 训练模型在预测每个 token 前发出隐藏的 “thought”，再通过 learned weight 把 thought-aware prediction 与 baseline prediction 混合。

结果：Mistral 7B 在不做 task-specific fine-tuning 的情况下，GSM8K zero-shot 从 5.9% 提升到 10.9%，CommonsenseQA 从 36.3% 提升到 47.2%。模型学会了 “when to think”：困难 token 会产生更长的内部 rationale，简单 token 几乎不需要。

### Why all three share a safety concern / 三者为什么共享同一个安全问题

三种方法都把最终答案作为梯度信号。一个 rationale 即使通过有缺陷的推理得到正确答案——利用捷径、猜测，或使用不可泛化的模式——也会被正向强化。在同分布问题上，这种捷径有效；在分布外问题上，它会静默失效。

V-STaR 的 verifier 通过学习排序 rationale 来缓解这个问题，但 verifier 仍然训练在同一套标签上。它可能学会偏好格式漂亮但错误的推理，而不是诚实的不确定性。更安全的设计，是把 STaR 风格数据与两件事结合起来：（a）process-supervised reward models（奖励中间步骤，而不只是答案），（b）能打破简单捷径的 held-out OOD evaluation。

### Comparison / 对比

| Method | Training signal | Inference cost | Data waste | Known failure mode |
|---|---|---|---|---|
| STaR | 答案正确就保留 `(rationale, answer)` | 1x | 丢弃所有错误 rationale | shortcut rationales |
| STaR + rationalization | 上述信号 + 带正确答案 hint 的重试 | 1x | 更少 | rationalized rationales 可能不可信 |
| V-STaR | STaR + 基于两类样本的 DPO verifier | Nx (best-of-N) | 最小 | verifier 可能强化自信的错误 |
| Quiet-STaR | per-token rationale + mixing weight | 1.5-3x | 最小 | 仍然是 answer-conditioned gradient |

### Where this sits in the 2026 stack / 它在 2026 技术栈中的位置

STaR 并不新。但这个模式在 2025-2026 年到处重新出现。对可验证数学问题做 RL（DeepSeek-R1、Kimi-k1.5、o1）就是把 STaR 的 answer-conditioned gradient signal 放大。Process reward models（Lightman et al., 2023；OpenAI 的 “Let's verify step by step”）是 process-supervised 替代路线。AlphaEvolve（Lesson 3）是代码版 STaR，只是用 program evaluator 代替标签。Darwin Godel Machine（Lesson 4）是对 Agent scaffolding 自身做 STaR。

理解 STaR，后面这些系统都会变得清楚。它是最小可行的 self-improvement loop。

```figure
reflection-loop
```

## Build It / 动手构建

本课用一个 toy arithmetic task 搭建 STaR 循环：生成 rationale、按答案筛选、更新模型倾向，再加入 V-STaR 风格的 verifier。重点不是追求模型规模，而是观察 shortcut 如何被训练循环保留下来。

## Use It / 应用它

`code/main.py` 会在 toy arithmetic task 上运行一个模拟 STaR loop。你可以观察：

- accuracy 如何随 bootstrap rounds 上升。
- 捷径如何混入：模拟器包含一种 “lazy” rationale class，它 40% 的时候能答对，但泛化很差。观察 STaR 是否会保留它们。
- verifier（V-STaR 风格）如何在推理时提供帮助，但不能完全清除训练阶段引入的捷径。

## Ship It / 交付它

`outputs/skill-star-loop-reviewer.md` 帮你在训练前审计一个 proposed self-taught-reasoning pipeline。

## Exercises / 练习

1. 运行模拟器。把 shortcut frequency 设为 0，再设为 0.4。即使两次在训练分布上都超过 90%，最终 accuracy 会分歧多少？

2. 给模拟器加入 held-out OOD test。从不同分布抽取问题，并在 in-distribution 和 OOD set 上评估自举后的模型。量化差距。

3. 阅读 Quiet-STaR paper（arXiv:2403.09629）第 3 节。分别用三句话解释 “end-of-thought” token 和 mixing-weight head。

4. 对比 STaR 的 keep-if-correct 过滤器与一个 process-supervised 替代方案，后者独立奖励每个 rationale step。指出 labelling cost 差异和可能的质量差异。

5. 设计一个能抓住 deployed model 中 shortcut rationales 的评测。它不必完美，但必须能打破 STaR loop 最容易强化的简单捷径。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| STaR | “Self-Taught Reasoner” | 在模型生成且最终答案正确的 rationales 上微调；重复 |
| Rationalization | “带 hint 的重试” | 对 base model 做错的问题，注入正确答案并重新 prompt 生成 rationale |
| V-STaR | “Verifier STaR” | 用正确和错误 rationales DPO-train verifier，并在推理时用于选择 |
| Quiet-STaR | “Per-token rationales” | 在每个 token 位置生成 hidden thoughts，再与 baseline prediction 混合 |
| Answer-conditioned gradient | “基于 outcome 的信号” | 训练循环奖励最终答案，而不是推理步骤 |
| Process reward model | “Step-level verifier” | 奖励每一步正确性的 reward model，而不是只看 outcome；与 STaR 形成对比 |
| Shortcut rationale | “答案对、推理错” | 通过不可泛化模式抵达标签的 rationale；STaR 会保留它 |

## Further Reading / 延伸阅读

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — 原始论文。
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — 为 inference-time selection 加入 DPO verifier。
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — per-token internal rationales。
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — process reward models，另一种梯度信号。
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在可验证任务上做 RL，把 STaR 放大到 frontier training。
