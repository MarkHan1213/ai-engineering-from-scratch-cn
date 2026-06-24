# Reward Modeling & RLHF / 奖励建模与 RLHF

> 人类写不出“好助手回答”的 reward function，但可以比较两个回答并选出更好的那个。用这些比较训练 reward model，然后用 RL 针对它优化语言模型。Christiano 2017。InstructGPT 2022。这套 recipe 把 GPT-3 推向 ChatGPT。到 2026 年它大多被 DPO 取代，但心智模型仍然重要。

**类型：** 构建
**语言：** Python
**前置知识：** 第 05 阶段 · 05（Sentiment）, 第 09 阶段 · 08（PPO）
**时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 RLHF 中 SFT → RM → PPO 三阶段 pipeline。
- 用 Bradley-Terry pairwise logistic loss 训练 reward model。
- 在 toy policy 上加入 RM reward 与 KL-to-reference penalty。
- 监控 KL、reward hacking、length hacking 等 RLHF 关键诊断。
- 判断 2026 年何时用 RLHF-PPO、DPO、GRPO、PRM 或 RLAIF。

## The Problem / 问题

你已经用 next-token-prediction objective 训练了一个语言模型。它能写出语法正确的英文。它也会撒谎、跑题、该拒绝时不拒绝。继续预训练不能修复这个问题，因为 web text 本身就是问题的一部分。

你想要一个 *scalar reward*，能判断“对 instruction X，response A 是否比 response B 更好”。手写这个 reward function 不现实。“helpfulness” 不是 token 上的闭式表达式。但人类可以比较两个输出并标注偏好。这种数据可以低成本大规模收集。

RLHF（Christiano et al. 2017; Ouyang et al. 2022）把偏好转成 reward model，再用 PPO 针对这个 reward 优化 LM。三步：SFT → RM → PPO。这是 ChatGPT、Claude、Gemini 以及 2023–2025 年几乎所有 aligned-LLM 的上线 recipe。

2026 年，PPO 这一步大多被 DPO（Phase 10 · 08）取代，因为它更便宜，alignment tuning 效果也接近。但 *reward model* 这块仍然支撑 Best-of-N sampler、RL-from-verifiable-rewards pipeline，以及使用 process reward model 的 reasoning model。理解 RLHF，就理解了整个 alignment stack。

## The Concept / 概念

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**Stage 1: Supervised Fine-Tuning (SFT).** 从 pretrained base model 开始。用人类写的目标行为 demonstrations 做 fine-tune（instruction-following responses、helpful replies 等）。结果是模型 `π_SFT`：它 *偏向好行为*，但 action space 仍然无界。

**Stage 2: Reward Model training.**

- 收集 prompts `x` 对应的 response pairs `(y_+, y_-)`，由人类标注 “y_+ is preferred over y_-”。
- 训练 reward model `R_φ(x, y)`，让它给 `y_+` 更高分。
- Loss 是 **Bradley-Terry pairwise logistic**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。reward 差值隐含偏好的 log-odds。BT 自 1952 年（Bradley-Terry）以来就是标准方法，也是现代 RLHF 的主流选择。

- `R_φ` 通常从 SFT model 初始化，在顶部加一个 scalar head。相同 transformer backbone，一个 linear layer 输出 reward。

**Stage 3: PPO against the RM with KL penalty.**

- 从 `π_SFT` 初始化可训练 policy `π_θ`。保留一个冻结的 *reference* `π_ref = π_SFT`。
- response `y` 结束时的 reward：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL penalty 防止 `π_θ` 任意偏离 `π_SFT`；它是 *regularizer*，不是 hard trust region。`β` 通常是 `0.01`-`0.05`。
- 用这个 reward 运行 PPO（Lesson 08）。advantages 在 token-level trajectory 上计算，但 RM 只给完整 response 打分。

**为什么需要 KL？** 没有它，PPO 会非常愿意找到 reward-hacking 策略：RM 只在 in-distribution completions 上训练过。一个 out-of-distribution response 可能比任何人类写的回答得分更高。KL 把 `π_θ` 拉回 RM 训练过的 manifold 附近。这是 RLHF 中最重要的 knob。

**2026 status:**

- **DPO** (Rafailov 2023)：通过闭式代数把 Stage 2+3 压成一个 preference data 上的 supervised loss。没有 RM，没有 PPO。alignment benchmark 上质量接近，compute 只需一小部分。Phase 10 · 08 会讲。
- **GRPO** (DeepSeek 2024–2025)：用 group-relative baseline 替代 critic 的 PPO，reward 来自 *verifier*（代码运行 / 数学答案匹配），而不是人类训练的 RM。reasoning model 的主流做法。Phase 9 · 12 会讲。
- **Process reward models (PRMs):** 对 partial solutions（每个 reasoning step）打分，用在 reasoning 的 RLHF 与 GRPO variants 中。
- **Constitutional AI / RLAIF:** 用 aligned LLM 生成偏好，替代人类标注。扩展 preference budget。

## Build It / 动手构建

本课使用 tiny synthetic “prompts” 和 “responses”，都表示为字符串。RM 是 bag-of-tokens 表示上的 linear scorer。没有真实 LLM；关键是 pipeline *形状*，不是规模。见 `code/main.py`。

### Step 1: synthetic preference data / 合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

真实 RLHF 中，这里换成人类标注者。形状完全相同：`(prompt, preferred_response, rejected_response)`。

### Step 2: Bradley-Terry reward model / Bradley-Terry 奖励模型

Linear score：`R(x, y) = w · bag(y)`。训练目标是最小化 BT pairwise log-loss：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百次 update 后，`w` 会给 good-word tokens 正权重，给 bad tokens 负权重。

### Step 3: PPO-like policy on top of RM / 在 RM 上做 PPO-like policy

我们的 toy policy 从词表中产生单个 token。用 RM 给 token 打分，计算 `log π_θ(token | prompt)`，加上 KL-to-reference penalty，然后应用 clipped PPO surrogate。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Step 4: monitor the KL / 监控 KL

每次 update 跟踪 mean `KL(π_θ || π_ref)`。如果它爬到 `~5-10` 以上，说明 policy 已经远离 `π_SFT`；通常是 `β` 太低或 reward hacking 正在开始。这是真实 RLHF 中最重要的诊断。

### Step 5: the production recipe with TRL / 使用 TRL 的生产 recipe

理解 toy pipeline 后，真实 library 用户写的也是同一个循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现：Stage 2 用 `RewardTrainer`，Stage 3 用内置 KL-to-reference 的 `PPOTrainer`。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

library 替你做了三件事。`adap_kl_ctrl=True` 实现 adaptive-β schedule：如果 observed KL 超过 `target_kl`，β 翻倍；如果低于一半，β 减半。reference model 按约定必须冻结；你不能不小心让它和 `policy` 共享参数。value head 和 policy 位于同一个 backbone 上（`AutoModelForCausalLMWithValueHead` 会附加一个 scalar MLP head），所以 TRL 会分别报告 `policy/kl` 和 `value/loss`。

## Pitfalls / 常见陷阱

- **Over-optimization / reward hacking.** RM 不完美；`π_θ` 会找到高分但糟糕的对抗性 completions。症状：reward 无限上升，而 human eval score 持平或下降。修复：早停、提高 `β`、扩展 RM training data。
- **Length hacking.** 在 helpful responses 上训练的 RM 往往隐式奖励长度。policy 学会填充回答。修复：length-normalized reward，或使用 length-aware RM 的 RLAIF。
- **Too-small RM.** RM 至少要和 policy 一样大。太小的 RM 无法可靠评分 policy 输出。
- **KL tuning.** β 太低 → drift 和 reward hacking。β 太高 → policy 几乎不变。标准技巧是使用 target fixed KL per step 的 *adaptive* β。
- **Preference-data noise.** 人类标签大约 30% 有噪声或歧义。可以用 agreement-filtered data 训练 RM，或在 BT 上使用 temperature。
- **Off-policy problems.** PPO 数据在第一个 epoch 后会略微 off-policy。像 Lesson 08 一样监控 clip fraction。

## Use It / 应用它

2026 年 RLHF 是分层使用的：

| Layer | Target | Method |
|-------|--------|--------|
| Instruction following, helpfulness, harmlessness | Alignment | DPO (Phase 10 · 08) preferred over RLHF-PPO. |
| Reasoning correctness (math, code) | Capability | GRPO with verifier reward (Phase 9 · 12). |
| Long-horizon multi-step tasks | Agentic | PPO / GRPO with process reward models over steps. |
| Safety / refusal behavior | Safety | RLHF-PPO with separate safety RM, or Constitutional AI. |
| Best-of-N at inference | Fast alignment | Use RM at decode time; no policy training needed. |
| Reward distillation | Inference compute | Train a small "reward head" on top of a frozen LM. |

RLHF 是 2022–2024 年的 *the* method。到 2026 年，生产 alignment pipeline 通常 DPO-first，只有 RM-intensive 或 safety-critical steps 才使用 PPO。

## Ship It / 交付它

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Exercises / 练习

1. **Easy.** 在 `code/main.py` 中，用 500 个 synthetic preference pairs 训练 Bradley-Terry reward model。在 100 个 held-out pairs 上测 pairwise accuracy，应该超过 90%。
2. **Medium.** 用 `β ∈ {0.0, 0.1, 1.0}` 跑 toy PPO-RLHF loop。分别画 RM score vs KL-to-reference over updates。哪些 run 发生 reward-hack？
3. **Hard.** 在同一份 preference data 上实现 DPO（closed-form preference-likelihood loss），并与 RLHF-PPO pipeline 比较 compute 使用量和最终达成的 RM score。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| RLHF | “Alignment RL” | 三阶段 SFT + RM + PPO pipeline（Christiano 2017, Ouyang 2022）。 |
| Reward Model (RM) | “The scoring net” | 通过 Bradley-Terry 拟合 pairwise preferences 得到的 learned scalar function。 |
| Bradley-Terry | “Pairwise logistic loss” | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM objective。 |
| KL penalty | “Stay near the reference” | reward 中的 `β · KL(π_θ \|\| π_ref)`；防 reward hacking 的 regularizer。 |
| Reward hacking | “Goodhart's law” | Policy 利用 RM 缺陷；症状是 reward 上升、human eval 持平。 |
| RLAIF | “AI-labeled preferences” | 标签来自另一个 LM 而不是人类的 RLHF。 |
| PRM | “Process Reward Model” | 对 partial reasoning steps 打分；用于 reasoning pipeline。 |
| Constitutional AI | “Anthropic's method” | 由显式规则指导的 AI-generated preferences。 |

## Further Reading / 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — RLHF 起点论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的 recipe。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 更早的 summarization RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年 post-RLHF 默认方法。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 与 self-critique loop。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 生产级 `RewardTrainer` 与 `PPOTrainer`。阅读 trainer source 能看到 adaptive-KL 与 value-head 细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 带图解释三阶段 pipeline 的经典 walk-through。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — library 本身；`examples/` 有 Llama、Mistral、Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — reward-hypothesis 视角；思考 reward hacking 的必要前置。
