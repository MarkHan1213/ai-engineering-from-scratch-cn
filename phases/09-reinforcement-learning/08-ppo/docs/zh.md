# Proximal Policy Optimization (PPO) / 近端策略优化

> A2C 每个 rollout 只更新一次就丢弃。PPO 用 clipped importance ratio 包住 policy gradient，让同一批数据能做 10+ 个 epoch，而 policy 不至于爆炸。Schulman et al. (2017)。到 2026 年，它仍然是默认 policy-gradient 算法。

**类型：** 构建
**语言：** Python
**前置知识：** 第 09 阶段 · 06（REINFORCE）, 第 09 阶段 · 07（Actor-Critic）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 PPO 为什么能在 on-policy 数据上做 multi-epoch 更新。
- 推导 importance ratio 与 clipped surrogate objective 的作用。
- 实现 rollout-time `log π_old(a | s)` 记录与 clipped update。
- 监控 KL、clip fraction、explained variance 三个关键诊断。
- 判断 PPO 在 RLHF、游戏、机器人和 reasoning LLM 中的变体位置。

## The Problem / 问题

A2C（Lesson 07）是 on-policy：gradient `E_{π_θ}[A · ∇ log π_θ]` 要求数据来自 *当前* `π_θ`。更新一次后，`π_θ` 就变了；刚用过的数据已经 off-policy。复用它会让 gradient 有偏。

Rollout 很贵。Atari 上，8 个 envs × 128 steps 的一次 rollout = 1024 transitions，还要十几秒环境时间。只做一次 gradient step 就丢掉很浪费。

Trust Region Policy Optimization（TRPO, Schulman 2015）是第一个修复：限制每次 update，使 old policy 和 new policy 的 KL divergence 小于 `δ`。理论干净，但每次 update 都需要 conjugate-gradient solve。2026 年几乎没人跑 TRPO。

PPO（Schulman et al. 2017）把硬 trust-region constraint 换成简单 clipped objective。多一行代码。每个 rollout 十个 epochs。不需要 conjugate gradients。理论保证够用。九年后，它仍然是从 MuJoCo 到 RLHF 的默认 policy-gradient 算法。

## The Concept / 概念

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**Importance ratio.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是 new policy 相对于收集数据的 old policy 的 likelihood ratio。`r_t = 1` 表示没有变化。`r_t = 2` 表示 new policy 采取 `a_t` 的概率是 old policy 的两倍。

**Clipped surrogate.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两种情况：

- 如果 advantage `A_t > 0`，而 ratio 想超过 `1 + ε`，clip 会把梯度压平：不要把一个好动作的概率推到比旧概率高出 `+ε` 之外。
- 如果 advantage `A_t < 0`，而 ratio 想跌破 `1 - ε`，clip 会截住梯度：不要把坏动作概率压得比旧概率低太多。

`min` 处理另一个方向：如果 ratio 沿着 *有利* 方向移动，你仍然得到梯度（只在会伤害稳定性的那一侧裁剪）。

典型 `ε = 0.2`。把 objective 画成 `r_t` 的函数：一个 piecewise-linear 函数，在“好的一侧”有平顶，在“坏的一侧”有平底。

**Full PPO loss.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

和 A2C 是同样的 actor-critic 结构。三个系数通常是 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**Training loop.**

1. 在 `N` 个 parallel envs 上各收集 `T` 步，共 `N × T` transitions。
2. 计算 advantages（GAE），并把它们冻结为常数。
3. 把 `π_{θ_old}` 冻结为当前 `π_θ` 的 snapshot。
4. 对 `K` 个 epochs，对每个 `(s, a, A, V_target, log π_old(a|s))` minibatch：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + value loss + entropy。
   - 做 gradient step。
5. 丢弃 rollout。回到第 1 步。

`K = 10`、minibatch size 64 是标准超参组。PPO 很 robust：精确数值在 ±50% 范围内通常都能工作。

**KL-penalty variant.** 原论文提出过另一个版本：adaptive KL penalty，`L = L^{PG} - β · KL(π_θ || π_old)`，并根据 observed KL 调整 `β`。Clipping 版本成为主流；KL 版本则在 RLHF 中保留下来，因为你本来就总是需要约束 reference policy 的 KL。

## Build It / 动手构建

### Step 1: capture `log π_old(a | s)` at rollout time / rollout 时记录 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

snapshot 只在 rollout 时取一次。update epochs 中它不会变化。

### Step 2: compute GAE advantages (Lesson 07) / 计算 GAE advantages（Lesson 07）

和 A2C 一样。跨 batch 归一化。

### Step 3: clipped surrogate update / clipped surrogate 更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

“clipped → zero gradient” 这个模式就是 PPO 的核心。如果 new policy 已经在有利方向上漂移得太远，更新会停止。

### Step 4: value and entropy / value 与 entropy

像 A2C 一样，加上 critic target 的标准 MSE，以及 actor 的 entropy bonus。

### Step 5: diagnostics / 诊断

每次 update 观察三件事：

- **Mean KL** `E[log π_old - log π_θ]`。应该保持在 `[0, 0.02]`。如果超过 `0.1`，降低 `K_EPOCHS` 或 `LR`。
- **Clip fraction** — ratio 落在 `[1-ε, 1+ε]` 之外的样本比例。应该约为 `~0.1-0.3`。如果接近 `~0`，clip 没触发 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，说明你在过拟合这批 rollout → 降低它们。
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`。critic 质量指标。critic 学会后应接近 1。

## Pitfalls / 常见陷阱

- **Clip coefficient mistuned.** `ε = 0.2` 是事实标准。`0.1` 会让 update 过于胆小；`0.3+` 容易不稳定。
- **Too many epochs.** `K > 20` 经常让训练不稳定，因为 policy 离 `π_old` 漂得太远。限制 epochs，尤其是大网络。
- **No reward normalization.** 大 reward scale 会吃掉 clip range。计算 advantages 前先归一化 rewards（running std）。
- **Forgetting advantage normalization.** 每个 batch 做 zero-mean/unit-std 是标准做法。跳过它会让 PPO 在多数 benchmark 上崩掉。
- **Learning rate not decayed.** PPO 受益于把 LR 线性衰减到 0。Constant LR 通常更差。
- **Importance ratio math errors.** 为了数值稳定，始终用 `exp(log_new - log_old)`，不要用 `new / old`。
- **Wrong gradient sign.** 最大化 surrogate = *最小化* `-L^{CLIP}`。符号反了是最常见的 PPO bug。

## Use It / 应用它

PPO 是 2026 年许多领域默认的 RL 算法：

| Use case | PPO variant |
|----------|-------------|
| MuJoCo / robotics control | PPO with Gaussian policy, GAE(0.95) |
| Atari / discrete games | PPO with categorical policy, rolling 128-step rollouts |
| RLHF for LLMs | PPO with KL penalty to reference model, reward from RM at end of response |
| Large-scale game agents | IMPALA + PPO (AlphaStar, OpenAI Five) |
| Reasoning LLMs | GRPO (Lesson 12) — PPO variant without critic |
| Preference-only data | DPO — closed-form collapsing of PPO+KL, no online sampling |

PPO 的 *loss shape*——clipped surrogate + value + entropy——是 DPO、GRPO 和几乎所有 RLHF pipeline 的脚手架。

## Ship It / 交付它

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Exercises / 练习

1. **Easy.** 在 4×4 GridWorld 上运行 PPO，使用 `ε=0.2, K=4`。在匹配 env steps 的情况下，和 A2C（每个 rollout 一个 epoch）比较 sample efficiency。
2. **Medium.** 扫 `K ∈ {1, 4, 10, 30}`。画 return vs env steps，并跟踪每次 update 的 mean KL。在这个任务上，`K` 到多少时 KL 会爆炸？
3. **Hard.** 把 clipped surrogate 换成 adaptive KL penalty（如果 `KL > 2·target`，`β` 翻倍；如果 `KL < target/2`，`β` 减半）。比较 final return、稳定性，以及不用 clip 的表现。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Importance ratio | “r_t(θ)” | `π_θ(a\|s) / π_old(a\|s)`；相对于收集数据的 policy 的偏离程度。 |
| Clipped surrogate | “PPO's main trick” | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利侧越过 clip 后梯度变平。 |
| Trust region | “TRPO / PPO intent” | 限制每次 update 的 KL，保证 monotone improvement。 |
| KL penalty | “Soft trust region” | PPO alternative：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| Clip fraction | “How often clipping triggers” | 诊断指标，应该在 0.1-0.3；超出说明调参有问题。 |
| Multi-epoch training | “Data reuse” | 每个 rollout 做 K 个 epochs；用一点方差代价换 sample efficiency。 |
| On-policy-ish | “Mostly on-policy” | PPO 名义上是 on-policy，但 K>1 epochs 会安全地使用 slightly-off-policy data。 |
| PPO-KL | “The other PPO” | KL-penalty variant；RLHF 中会用，因为 KL-to-reference 本来就是约束。 |

## Further Reading / 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 原论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 对所有 PPO hyperparameters 做 ablation。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；RLHF 中的 PPO recipe。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 带 PyTorch 的清晰现代讲解。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 很多论文使用的单文件 PPO 参考实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — language model 上 PPO 的生产 recipe；和 Lesson 09（RLHF）一起读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — “37 个代码级优化”论文；区分哪些 PPO 技巧是关键，哪些只是 folklore。
