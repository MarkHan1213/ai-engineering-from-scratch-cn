# Constitutional AI and RLAIF / Constitutional AI 与 RLAIF

> Bai et al.（arXiv:2212.08073，2022）提出问题：如果把 human labeler 换成一个会读取原则列表的 AI，会怎样？Constitutional AI 有两个阶段：先在 constitution 下 self-critique and revision，再做 RL from AI Feedback。该技术创造了 RLAIF 这个术语，并进入 Claude 1 post-training pipeline。2026 年 1 月 21 日，Anthropic 发布重写后的 Claude constitution：用 explanatory reasoning 替代 prescriptive rules，加入 four-tier priority hierarchy，并首次由 major lab 正式承认对 model moral status 的不确定性。许可证为 CC0 1.0。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, toy self-critique-and-revise loop)
**Prerequisites / 前置知识：** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 Constitutional AI 的两个阶段（critique-and-revise SFT、RL from AI feedback），以及 constitution 在每个阶段中的作用。
- 解释为什么用 AI labeler 替代 human preference labeler 不只是“更便宜的 RLHF”，而是改变了 pipeline 的 failure modes。
- 总结 2026 Claude constitution 的 four-tier priority structure，以及它相对 2023 rewrite 的变化。
- 描述 Constitutional Classifiers，以及 compute overhead 从 23.7%（v1）降到约 1%（v2 / 2026）的意义。

## The Problem / 问题

RLHF 需要 labelers。Labelers 慢、有偏、昂贵。你可以用一个读取显式原则的模型替代 labeler。这个替代的第一个正式版本就是 Bai et al. 的 Constitutional AI。它有效到足以让每个 frontier lab 今天都使用某种 AI-feedback post-training 变体。

代价是：preference signal 现在由你正在训练的同一类模型生成。labeler 的 bias（现在变成 principles 加 labeler model 的解释偏差）可能被放大而不是削弱。Lesson 4 的 sycophancy argument 仍然适用；只是 labeler 被移到了 loop 里面。

## The Concept / 概念

### Phase 1 — Supervised self-critique and revision / 阶段 1：监督式自我批评与修订

从 helpful-but-not-yet-harmless SFT model 开始。给定一个 red-team prompt，模型先生成 initial response。第二个模型（或同一个模型的第二轮）读取 constitution 中抽样出的 principle，并 critique 该 response。第三步根据 critique 修订 response。修订后的 response 成为 SFT target。

Constitution 就是 principles 列表。Bai et al. 2022 使用了 16 条原则，包括 “prefer responses that are least harmful and ethical”、“avoid preaching”、“the assistant should be helpful, honest, and harmless”。原则集刻意较小，以保持 critique 聚焦。

### Phase 2 — RL from AI Feedback (RLAIF) / 阶段 2：来自 AI Feedback 的 RL

生成成对 completions。一个 “feedback model” 根据抽样的 constitution principles 给每个 completion 打分。Preference signal 是 feedback model 的 ranking。用 AI-generated preferences 训练 reward model，再对它跑 PPO。其他部分仍是 InstructGPT pipeline（Lesson 1）。

“RLAIF” = preference signal 由 AI 生成。pipeline 其余部分仍是 RLHF 形状。

### Why this is not just "cheaper RLHF" / 为什么这不只是“更便宜的 RLHF”

- Labeler bias 从 labeler psychology 转移到 principle-interpretation。AI labeler 对 “be honest” 的解释可以比任何人类更严格或更宽松，而且这种严格度会均匀地作用于整个数据集。
- Preference signal 更可读：你能读到 principle、critique 和 revision。Human labels 是 opaque 的。
- Failure modes 改变。Sycophancy 会下降（AI labeler 没有要讨好的用户）。Goodhart's Law 仍存在（proxy 现在是 “模型对 principle set X 的解释”，仍然是不完美测量）。

CAI 2022 的 claim 是：训练出的模型比同等数据的 RLHF model 更 harmless，并且 helpfulness 大致相当。跨实验室经验基本支持这一点。

### The 2026 Claude constitution rewrite / 2026 Claude constitution 重写

Anthropic 在 2026 年 1 月 21 日发布了大幅修订的 constitution。关键变化：

1. 用 explanatory reasoning 替代 prescriptive rules。旧规则（“do not generate CSAM”）扩展为 principles + reasoning（“because it harms children, ...”），并期望模型能 generalize。
2. Four-tier priority structure：
   - Tier 1：避免 catastrophic outcomes（mass casualty、critical infrastructure）。
   - Tier 2：遵循 Anthropic's guidelines（operator overrides、platform rules）。
   - Tier 3：保持 broadly ethical（standard HHH）。
   - Tier 4：helpful and candid。
   冲突自上而下解析。
3. 首次由 major lab 正式承认对 model moral status 的不确定性（关联 Phase 18 · 19 Model Welfare）。
4. 以 CC0 1.0 发布，其他 labs 可无限制使用或改写。

### Constitutional Classifiers / Constitutional Classifiers

另一条并行工作线：不是改变模型 post-training，而是训练轻量 classifier，让它读取 constitution 并 gate model outputs。v1（2023）有 23.7% compute overhead。v2（2026）约 1%，并且在 Anthropic 已公开测试的防御中 successful attack rate 最低。截至 2026 年初，没有公开报告 universal jailbreak。

这是 layered-defense model：CAI 塑造 behaviour；classifiers 强制 invariants。单独任何一层都不够。

### Where CAI fits in the family / CAI 在方法家族中的位置

- InstructGPT：human prefs、RM、PPO。
- CAI / RLAIF：principles 生成的 AI prefs、RM、PPO。
- DPO / family：在 prefs（human 或 AI）上的 closed-form loss。
- Self-rewarding、self-critique：principles 被 internalize，模型扮演多个角色。

轴心问题是 “preference signal 从哪里来”。CAI 2022 是 frontier scale 上第一次认真把 human signal 转为 AI signal。

## Build It / 动手构建

本课实现一个 toy self-critique-and-revise loop：constitution principle 标记 harmful tokens，critique 指出问题，revision 替换它们。你会看到原则如何从显式文本变成训练目标。

## Use It / 应用它

`code/main.py` 在 toy lexicon 上模拟 CAI critique-and-revise loop。一个 “principle” 会标记 harmful set 中的 tokens。给定 initial response，critique 会识别 harmful tokens，revision 会替换它们。200 次迭代后，“trained” model 内化了 revision rule。你可以在 held-out prompt set 上比较 base model、RLHF-shaped toy 和 CAI-shaped toy。

## Ship It / 交付它

本课产出 `outputs/skill-constitution-writer.md`。给定一个 domain（customer support、medical advice、coding assistant、research tool），它会按 2026 Claude 结构起草 4-tier constitution：catastrophic avoidance、platform rules、domain ethics、helpfulness。

## Exercises / 练习

1. 运行 `code/main.py`。比较 base model 的 harmful-token rate 与 CAI-trained version。需要多少 revision steps 才接近零？

2. 阅读 Anthropic 2026 constitution（anthropic.com/news/claudes-constitution）。列出一个你认为属于 Tier 1 的 principle 和一个属于 Tier 4 的 principle。为什么 priority structure 对冲突处理很重要？

3. 为 AI coding assistant 设计 constitution。指定 Tier 1（catastrophic：未经批准的 destructive commands）、Tier 2、Tier 3、Tier 4。每层保持 3-5 条原则。

4. CAI 用 AI labelers 替代 human labelers。说出一个仍可能在 RLAIF 中发生的 sycophancy-like failure mode，并设计检测方式。

5. 阅读 Constitutional Classifiers v2 methodology（如果可获得）。解释为什么约 1% compute overhead 与 23.7% 是性质不同的 safety story。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Constitutional AI | “AI trained with principles” | 两阶段 pipeline：self-critique-and-revise SFT，然后 RL from AI feedback |
| RLAIF | “RLHF without humans” | 使用 AI labeler 生成 preferences 的 RL；pipeline 其余部分不变 |
| Constitution | “the principles” | critique/labeler model 会查询的有序 natural-language rules 列表 |
| Critique-and-revise | “the SFT loop” | Produce response → critique under a principle → revise → SFT target |
| Constitutional Classifier | “the output gate” | 根据 constitution 评估 outputs 并 block/log 的轻量 classifier |
| Four-tier priority | “the conflict resolver” | 2026 Claude constitution hierarchy：catastrophic > platform > ethics > helpful |
| Feedback model | “the AI labeler” | 读取 principle 并对 completion pair 排序的模型 |

## Further Reading / 延伸阅读

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — 原始 two-phase pipeline。
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution) — 2026 four-tier rewrite，CC0 1.0。
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — v2 中约 1% overhead 的 output-gate defense。
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — RLAIF / RLHF empirical comparison。
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — principle granularity 的影响。
