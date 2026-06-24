# The Direct Preference Optimization Family / Direct Preference Optimization 家族

> Rafailov et al. (2023) 说明 RLHF 的 optimum 可以用 preference data 写成闭式形式，因此可以跳过显式 reward model，直接优化 policy。这个 insight 产生了一整个家族：IPO、KTO、SimPO、ORPO、BPO，每个都在修补 DPO 的某种 failure mode。到 2026 年，direct alignment algorithms 在 frontier post-training runs 中比 PPO 更常见。但 Lesson 2 的 over-optimization curve 仍然适用：DAAs 没有逃离 Goodhart，只是改变了 Goodhart 咬人的位置。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites / 前置知识：** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 从 RLHF-with-KL optimum 推导 DPO closed form。
- 说明 IPO、KTO、SimPO、ORPO、BPO 分别修复 DPO 的哪种 failure mode。
- 区分 “implicit reward gap” 与 “preference strength”，并解释 IPO 的 identity mapping 为什么重要。
- 解释为什么 Rafailov et al. (NeurIPS 2024) 证明即使没有显式 RM，DAAs 仍然会 over-optimize。

## The Problem / 问题

RLHF objective（Lesson 1）是：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

它有一个已知 optimum：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此 reward 可以由 optimal policy 和 reference 的比例隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

把它代入 Bradley-Terry preference likelihood，partition function `Z(x)` 会抵消，因为它只依赖 `x`。剩下的就是只含 policy parameters 的 loss，不再需要 reward model。这就是 DPO。

麻烦在于：推导假设 optimum 可达、preference data 在分布内、reference policy 是真正的 mode anchor。这些都不完全成立。家族中的每个成员都在修补一个被违反的假设。

## The Concept / 概念

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出错的地方：

- implicit reward gap `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 无界。一个很小的 preference 可以产生任意大的 gap。
- loss 会把 chosen 和 rejected 的 log-probs 往相反方向推。只要 rejected 掉得更快，它甚至可以把 chosen absolute log-prob 往下推。这就是 Degraded Chosen Response 现象。
- out-of-distribution preferences（rare rare pair vs rare rare pair）会产生任意 implicit rewards。

### IPO (Azar et al., 2024)

Identity Preference Optimization 用 preference probability 上的 identity mapping 替换 log-sigmoid。loss 变成有界 target 上的 squared-error：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

margin 被 `1/(2 beta)` 有界。Preference strength 与 implicit-reward gap 成比例，不会 blow up。

### KTO (Ethayarajh et al., 2024)

Kahneman-Tversky Optimization 完全放弃 pairwise structure。给定单个 labeled output 和一个二元 “desirable” 或 “undesirable” signal，把它映射到 prospect-theory utility：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

并对 gains 与 losses 使用不同权重（loss aversion）。好处是可以使用 unpaired data，而这类数据丰富得多。

### SimPO (Meng et al., 2024)

Simple Preference Optimization 让 training signal 更贴近 generation。完全移除 reference policy，并按长度归一化 log-likelihood：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

再用 margin `gamma` 稳定训练。长度归一化去掉了利用 DPO length-bias failure mode 的激励（更长的 `y_w` 按构造会带来更大的 log-prob gap）。

### ORPO (Hong et al., 2024)

Odds-Ratio Preference Optimization 把 preference term 加到标准 SFT negative log-likelihood 上：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有 reference policy，SFT term 就是 regularizer。从 base model 到 aligned model 单阶段训练，不需要单独的 SFT checkpoint。

### BPO (ICLR 2026 submission, OpenReview id=b97EwMUWu7)

识别了 Degraded Chosen Responses 问题：DPO 保持 ranking `y_w > y_l`，但 `y_w` 的 absolute log-prob 可能下降。BPO 添加一个 single-line correction，惩罚 chosen response 的向下移动。论文报告在 Llama-3.1-8B-Instruct 数学推理上相对 DPO 有 +10.1% accuracy。

### The universal result: DAAs still over-optimize / 通用结果：DAAs 仍会过度优化

Rafailov et al. “Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms”（NeurIPS 2024）在多个数据集和 KL budgets 上用 DPO、IPO、SLiC 训练 policies。gold-reward-vs-KL curves 仍有 Gao et al. 的 peak-and-collapse 形状。implicit reward 会在训练中查询 out-of-distribution samples；KL regularization 无法稳定这一点。

DAAs 没有逃离 Goodhart。它们只是把问题表面从 “reward model 被 over-optimized” 变成 “reference policy ratio 被 over-optimized”。通用修复——更好数据、ensembles、early stopping——两边都适用。

### Choosing among them (2026) / 2026 年如何选择

- 如果有大量 paired preference data：用保守 `beta` 的 DPO；如果明显有 length bias，用 SimPO。
- 如果有 unpaired binary feedback：KTO。
- 如果想从 base model 单阶段训练：ORPO。
- 如果 DPO logs 里出现 degraded chosen log-probs：BPO。
- 如果 preference strengths 差异很大且 DPO saturating：IPO。

每个实验室都会在一套 battery 上跑五种方法，再按 task 选 winner。没有理由认为 math reasoning 与 safety 的 optimum 相同。

```figure
dpo-margin
```

## Build It / 动手构建

本课构建一个六种 preference loss 的比较器。你会在同一 toy preference dataset 上优化 DPO、IPO、KTO、SimPO、ORPO、BPO，观察 win rate、chosen-log-prob drift 和 implicit reward spread 如何分化。

## Use It / 应用它

`code/main.py` 在一个 toy preference dataset 上比较六种 losses（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实 preference strength 随 pair 改变。每个 loss 都在同一个 500-pair sample 和小 softmax policy 上优化。输出每种方法的 final win rate、chosen-log-prob drift 和 implicit-reward spread。

## Ship It / 交付它

本课产出 `outputs/skill-preference-loss-selector.md`。给定 dataset statistics（paired vs unpaired、variable vs uniform preference strength、length distribution）和 target（single-stage 或 SFT-then-preference），推荐 preference loss，并说明它防护的 failure mode。

## Exercises / 练习

1. 运行 `code/main.py`。报告 DPO 与 BPO 的 final chosen-log-prob drop。BPO 应该保留更高 chosen absolute probability，验证这一点。

2. 修改 preference data，让所有 pairs 具有相同 strength。六种方法中哪一种最 robust？哪一种退化？解释 IPO 在这里的优势。

3. 让 rejected responses 平均比 chosen 长 2x。其他不变，数值展示 DPO 的 length exploitation 和 SimPO 的修复。

4. Rafailov et al. (NeurIPS 2024) 声称 DAAs 会 over-optimize。复现一个 single-point version：画出 chosen-minus-rejected KL divergence，并观察 DPO 在大 `beta` 下的 over-optimization。

5. 阅读 BPO paper abstract（OpenReview b97EwMUWu7）。写下 BPO 在 DPO 上添加的 one-line correction，并与 `code/main.py` 中的实现核对。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| DPO | “RLHF without a reward model” | 从 closed-form RLHF optimum 推导出的 loss；只含 policy parameters |
| Implicit reward | “the log-ratio” | `beta * log(pi(y\|x) / pi_ref(y\|x))`，DPO 隐含的 reward |
| IPO | “bounded DPO” | 用 identity 替换 log-sigmoid；implicit reward gap 被 `1/(2 beta)` 截断 |
| KTO | “unpaired DPO” | 基于单标签的 prospect-theory utility，并带 loss aversion |
| SimPO | “reference-free DPO” | 长度归一化 log-likelihood + margin；无 reference policy |
| ORPO | “one-stage DPO” | NLL + odds-ratio preference term；从 base model 一遍训练 |
| BPO | “chosen-preserving DPO” | DPO 加一个惩罚 chosen response absolute log-prob 下降的项 |
| Degraded Chosen | “chosen goes down” | 只要 rejected 掉得更快，DPO 会降低 chosen log-prob |
| DAA | “direct alignment algorithm” | 任何跳过显式 RM 的 preference-loss 方法 |

## Further Reading / 延伸阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO。
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
