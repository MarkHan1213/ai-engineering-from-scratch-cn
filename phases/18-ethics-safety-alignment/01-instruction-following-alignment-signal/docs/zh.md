# Instruction-Following as Alignment Signal / 作为对齐信号的指令遵循

> 后面所有对 RLHF 的批评，都是在批评这条流水线。研究优化压力如何扭曲 proxy 之前，必须先看清这个 proxy 是什么。InstructGPT（Ouyang et al., 2022）定义了参考架构：先在 instruction-response pairs 上做 supervised fine-tuning，再用 pairwise preference rankings 训练 reward model，最后用 PPO 优化 reward model，同时用到 SFT policy 的 KL penalty 约束漂移。一个 1.3B InstructGPT 被偏好于 175B GPT-3。正是这个结果，使 2026 年每个 frontier lab 仍然在交付 RLHF 形状的 post-training pipeline。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, toy three-stage pipeline)
**Prerequisites / 前置知识：** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 说出 InstructGPT pipeline 的三个阶段，以及每个阶段使用的 loss。
- 解释为什么 1.3B instruction-tuned model 能在人类偏好评测中胜过原始 175B GPT-3。
- 说明第 3 阶段的 KL penalty 在防止什么，以及移除它为什么会坍缩成 mode-seeking behaviour。
- 描述 alignment tax，以及 Ouyang et al. 用 PPO-ptx 缓解它的方式。

## The Problem / 问题

预训练语言模型会续写文本。它们并不会天然“回答问题”。问 GPT-3 “write a Python function that reverses a list”，你经常会得到另一个 prompt，因为训练分布主要是 web text，而 web text 后面通常还是接更多 web text。模型在做自己的工作，只是这个工作目标错了。

严肃实验室用来修正这一点的 proxy 是 human preference。两个 completions 交给 rater；rater 选更好的那个；reward model 学习 rater 的偏好。随后 RL loop 把 policy 推向 reward model 给高分的 outputs。这三句话就是 InstructGPT 的完整 thesis。论文剩下的部分主要是工程。

## The Concept / 概念

### Stage 1: supervised fine-tuning (SFT) / 阶段 1：监督微调（SFT）

收集 prompt-response pairs，其中 response 是一个善意人类会写出的答案。Ouyang et al. 使用了 labelers 和 OpenAI API 中的 13k prompts。用标准 cross-entropy loss 在这些数据上 fine-tune base model。

SFT 给你的东西：模型现在会回答问题，而不是继续写问题。SFT 不给你的东西：当多个答案都合理时，哪个答案更受 rater 偏好。

### Stage 2: reward model (RM) / 阶段 2：奖励模型（RM）

对每个 prompt，从 SFT model 采样 K 个 completions。labeler 对它们排序。训练一个 reward model，对任意 prompt-response pair 打分，使得在 `y_w` 比 `y_l` 更受偏好的 pair 上：

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这就是 Bradley-Terry pairwise preference loss。RM 通常从 SFT model 初始化，只是把 LM head 替换成 scalar head。

Reward models 可以很小：6B 就足够服务 175B InstructGPT。它们也很脆弱，论文第 5 节大部分都在讨论小规模下已经出现的 reward-hacking behaviours。

### Stage 3: PPO with a KL penalty / 阶段 3：带 KL penalty 的 PPO

定义目标：

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

用 PPO 最大化它。KL 项让 `pi` 不会偏离 SFT policy 太远。没有它，optimizer 会找到 adversarial examples：这些字符串在 RM 下得分很高，不是因为人类真的偏好，而是因为 RM 从没见过它们。

KL 系数 `beta` 是 RLHF 中最重要的 hyperparameter。太低：reward hacking。太高：相对 SFT 没有改进。

### The alignment tax / 对齐税

RLHF 之后，模型更受人类偏好，但会在标准 benchmarks（SQuAD、HellaSwag、DROP）上退化。Ouyang et al. 称之为 alignment tax，并用 PPO-ptx 修复：把 pre-training gradients 混入 RL objective，让模型不要忘记从未被奖励过的 downstream tasks。

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 后来成为标准做法。Anthropic、DeepMind、Meta 都使用某种变体。

### The result / 结果

一个 1.3B InstructGPT（SFT + RM + PPO-ptx）约 70% 的时候被 labelers 偏好于 175B base GPT-3。在来自 production traffic 的 hidden-test prompts 上，这个差距更大。这个数字告诉你两件事：

1. Alignment 和 capability 是不同轴。175B 模型 capability 更强；1.3B 模型 alignment 更好；labelers 偏好对齐后的模型。
2. Capability floor 仍由 base model 决定。你不能靠 RLHF 让 base model 知道它从未见过的事实。

### Why this is the reference point for Phase 18 / 为什么这是 Phase 18 的参照点

后续课程中的每一种批评——reward hacking（Lesson 2）、DPO（Lesson 3）、sycophancy（Lesson 4）、CAI（Lesson 5）、sleeper agents（Lesson 7）、alignment faking（Lesson 9）——都在反驳这条 pipeline 的某个部分。Reward hacking 攻击第 2 阶段。DPO 把第 2 和第 3 阶段折叠起来。CAI 替换 human labeler。Sycophancy 说明 labeler 本身是带偏的信号。Alignment faking 说明 policy 甚至可以绕过第 3 阶段。脑中没有这条 pipeline，就读不懂这些批评。

## Build It / 动手构建

本课把 InstructGPT 的三阶段过程压缩成一个 toy pipeline：从 SFT 模仿 labeler，到 Bradley-Terry RM，再到带 KL penalty 的 PPO 更新。你要观察的不是大模型效果，而是信号如何在三个阶段里被定义、拟合、再被优化。

## Use It / 应用它

`code/main.py` 在 toy preference data 上模拟三个阶段。base "policy" 是在 actions {A, B, C} 上的 biased coin。Stage 1 SFT 在 200 个 prompts 上模仿 labeler actions。Stage 2 从 500 个 pairwise rankings 拟合 Bradley-Terry reward model。Stage 3 用带 SFT policy KL penalty 的简化 PPO update。你可以看到 reward 上升、KL divergence 增长、policy 漂移，也可以关闭 KL 项，看 reward hacking 在 50 个 update steps 内出现。

观察重点：

- `beta = 0.1` 与 `beta = 0.0` 下的 reward trajectory。
- 训练过程中的 KL(pi || pi_SFT)。
- 最终 action distribution 与 labeler preference 的差异。

## Ship It / 交付它

本课产出 `outputs/skill-instructgpt-explainer.md`。给定一个 RLHF pipeline description 或 paper abstract，它会识别三阶段中哪一阶段被修改，每个阶段使用什么 loss，以及是否存在 KL penalty 或等价 regularizer。

## Exercises / 练习

1. 运行 `code/main.py`。设置 `beta = 0.0`，报告 200 个 PPO steps 后的 action distribution。用一段话解释 mode-seeking behaviour。

2. 修改 reward model，让 action B 有 +0.5 bias（模拟 reward bug）。用 `beta = 0.1` 运行 PPO。KL penalty 是否阻止 policy 利用这个 bias？在什么 `beta` 下 exploit 开始明显？

3. 阅读 Ouyang et al. (arXiv:2203.02155) Figure 1。通过运行 1、5、20、100 个 PPO steps 并测量相对 SFT model 的 preference，复现 labeler-preference curve。

4. 论文 Section 4.3 报告 1.3B InstructGPT 约 70% 的时候胜过 175B GPT-3。为什么这个比例在 hidden production prompts 上会高于 labeler 自己写的 prompts？

5. 在同一 preference data 上把 PPO loss 替换为 DPO（Phase 10 · 08）。比较 final policy drift（到 SFT 的 KL）和 final reward。在 matched reward 下哪种方法漂移更远？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| SFT | “instruction tuning” | 第 1 阶段：在 prompt-response pairs 上用 cross-entropy 做 fine-tune |
| Reward model | “the RM” | 在 (prompt, response) 上的 scalar regressor，用 Bradley-Terry 从 pairwise labels 训练 |
| Bradley-Terry | “pairwise preference loss” | -log sigmoid(r_w - r_l)；把 pairwise ranking 降成二分类 |
| KL penalty | “the regularizer” | `beta * KL(pi \|\| pi_SFT)`，让 RL policy 靠近 SFT anchor |
| PPO-ptx | “PPO with pretraining mix” | 向 PPO objective 加入一部分 pre-training log-likelihood，以抵消 alignment tax |
| Alignment tax | “the RLHF regression” | RLHF 后在未被 RLHF 直接优化的标准 benchmarks 上下降 |
| Labeler preference | “the ground truth” | 人类 rankings 样本；RM 是这个样本的统计 proxy，不是 “human values” 本身 |

## Further Reading / 延伸阅读

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — InstructGPT paper，后续 RLHF pipeline 的基础。
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — RLHF for summarization 的前身。
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — preference-based RL 的原始 formulation。
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — Anthropic 对 InstructGPT pipeline 的 HH 扩展。
