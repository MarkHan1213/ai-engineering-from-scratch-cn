# Policy Gradient — REINFORCE from Scratch / 策略梯度：从零实现 REINFORCE

> 停止估计 value，直接参数化 policy，计算 expected return 的梯度，然后向上走。Williams (1992) 用一个定理写清楚了这件事。PPO、GRPO 和所有 LLM RL loop 都建立在这里。

**类型：** 构建
**语言：** Python
**前置知识：** 第 03 阶段 · 03（Backpropagation）, 第 09 阶段 · 03（Monte Carlo）, 第 09 阶段 · 04（TD Learning）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 policy gradient 为什么直接优化 `π_θ(a | s)` 而不是 `Q`。
- 推导并实现 REINFORCE 的 `G · ∇ log π` 更新。
- 实现 softmax policy、动作采样和 log-probability 计算。
- 理解 baseline、reward-to-go 与 entropy bonus 如何降低方差和保持探索。
- 识别 REINFORCE、A2C、PPO、GRPO、DPO 之间的继承关系。

## The Problem / 问题

Q-learning 和 DQN 参数化的是 *value* function。你通过 `argmax Q` 选择动作。对离散动作和离散状态来说这没问题。但当动作是连续的（怎么对 10 维 torque 做 `argmax`？），或者你希望 policy 本身是随机的（`argmax` 天生确定性）时，它就不合适了。

Policy gradients 改为直接参数化 *policy*。`π_θ(a | s)` 是一个输出动作分布的神经网络。执行时从中采样。计算 expected return 对 `θ` 的梯度。向上走。没有 `argmax`。没有 Bellman recursion。只有对 `J(θ) = E_{π_θ}[G]` 的 gradient ascent。

REINFORCE theorem（Williams 1992）告诉你这个梯度可计算：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。跑一个 episode。计算 return。把每一步的 `∇ log π_θ(a | s)` 乘以 return。取平均。做 gradient-ascent。完成。

2026 年的每个 LLM-RL 算法——PPO、DPO、GRPO——都是 REINFORCE 的改进版。把它写到手指里，是本 phase 后续内容、Phase 10 · 07（RLHF implementation）和 Phase 10 · 08（DPO）的前提。

## The Concept / 概念

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**Policy gradient theorem.** 对任意由 `θ` 参数化的 policy `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从 step `t` 开始的 discounted return。期望是对从 `π_θ` 采样出的完整 trajectories `τ` 取的。

**证明很短。** 在 expectation 下对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（log-derivative trick）。把 `log P(τ; θ)` 分解成 `Σ log π_θ(a_t | s_t)` 加上不依赖 θ 的 environment terms。environment terms 消失。两行代数就得到定理。

**Variance reduction tricks.** Vanilla REINFORCE 的方差很凶：returns 有噪声，`∇ log π` 有噪声，它们的乘积更有噪声。两个标准修复：

1. **Baseline subtraction.** 把 `G_t` 替换为 `G_t - b(s_t)`，其中 `b(s_t)` 是任意不依赖 `a_t` 的 baseline。它无偏，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择是由 critic 学出的 `b(s_t) = V̂(s_t)` → actor-critic（Lesson 07）。
2. **Reward-to-go.** 把 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换成 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。给定动作只影响未来 returns；过去 rewards 只会贡献零均值噪声。

组合起来得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带 baseline 的 REINFORCE，也是 A2C（Lesson 07）和 PPO（Lesson 08）的直接祖先。

**Softmax policy parameterization.** 对离散动作，标准选择是：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任意输出每个动作 score 的神经网络。梯度形式很干净：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

也就是 taken action 的 score 减去 policy 下的期望 score。

**Gaussian policy for continuous actions.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式解。Phase 9 · 07 的 SAC 需要的就是这些。

```figure
policy-gradient-landscape
```

## Build It / 动手构建

### Step 1: softmax policy network / softmax policy 网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

在 tabular env 上使用 linear policy（每个动作一个 weight vector）。换到 Atari 时，替换成 CNN，保留 softmax head。

### Step 2: sampling and log-probability / 采样与 log-probability

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Step 3: rollout with log-probs captured / rollout 时记录 log-probs

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Step 4: REINFORCE update / REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的 onehot 减去概率）是 softmax policy gradients 的核心。把它练成肌肉记忆。

### Step 5: baselines / baseline

对最近 episodes 的 `G` 做 running mean，就足以让 4×4 GridWorld 跑起来；大约 500 个 episode 能收敛。把 baseline 升级为学出来的 `V̂(s)`，就得到 actor-critic。

## Pitfalls / 常见陷阱

- **Exploding gradients.** Returns 可能很大。乘上 `∇ log π` 之前，一定把 `G` 在 batch 内归一到接近 `~N(0, 1)`。
- **Entropy collapse.** Policy 过早收敛到近似确定性动作，停止探索并卡住。修复：给目标加 entropy bonus `β · H(π(·|s))`。
- **High variance.** Vanilla REINFORCE 需要数千个 episode。critic baseline（Lesson 07）或 TRPO/PPO 的 trust region（Lesson 08）是标准修复。
- **Sample inefficiency.** On-policy 意味着每条 transition 更新一次后就丢弃。Importance sampling 可以把旧数据捞回来，但会增加方差（PPO 的 ratio 就是 clipped IS weight）。
- **Non-stationary gradients.** 100 个 episode 前的同一条 gradient 使用的是旧 `π`。On-policy 方法因此每几个 rollout 就要更新。
- **Credit assignment.** 没有 reward-to-go 时，过去 rewards 也会贡献噪声。总是使用 reward-to-go。

## Use It / 应用它

2026 年，REINFORCE 很少直接运行，但它的梯度公式无处不在：

| Use case | Derived method |
|----------|---------------|
| Continuous control | PPO / SAC with Gaussian policy |
| LLM RLHF | PPO with KL penalty, running on token-level policy |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE with group-relative baseline, no critic |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| Discrete action robotics | A2C, A3C, PPO |
| Preference-only settings | DPO — REINFORCE rewritten as a preference-likelihood loss, no sampling |

当你在 2026 年的训练脚本里看到 `loss = -advantage * log_prob`，那就是带 baseline 的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都可以看作是在这一行上做 variance-reduction。

## Ship It / 交付它

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Exercises / 练习

1. **Easy.** 用 linear softmax policy 在 4×4 GridWorld 上实现 REINFORCE。不使用 baseline 训练 1,000 个 episode。画 learning curve，并测量 returns 的 variance（std）。
2. **Medium.** 加入 running-mean baseline 后重新训练。和 vanilla run 比较 sample efficiency 与 variance。baseline 把收敛步数降低了多少？
3. **Hard.** 加入 entropy bonus `β · H(π)`。扫 `β ∈ {0, 0.01, 0.1, 1.0}`。画 final return 与 policy entropy。这个任务上的 sweet spot 在哪里？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Policy gradient | “Train the policy directly” | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`；由 log-derivative trick 推导。 |
| REINFORCE | “The original PG algorithm” | Williams (1992)；Monte Carlo returns 乘以 log-policy gradient。 |
| Log-derivative trick | “Score function estimator” | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；让 expectation 的梯度可计算。 |
| Baseline | “Variance reduction” | 从 `G` 中减去任意 `b(s)`；无偏，因为 `E[b · ∇ log π] = 0`。 |
| Reward-to-go | “Only future returns count” | 使用 `G_t^{from t}` 而不是完整 `G_0`；正确且低方差。 |
| Entropy bonus | “Encourage exploration” | `+β · H(π(·\|s))` 项让 policy 不至于坍缩。 |
| On-policy | “Train on what you just saw” | 梯度期望相对于当前 policy；不能直接复用旧数据。 |
| Advantage | “How much better than average” | `A(s, a) = G(s, a) - V(s)`；REINFORCE-with-baseline 乘上的有符号量。 |

## Further Reading / 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 原始 REINFORCE 论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 带 function approximation 的现代 policy-gradient theorem。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 教材讲法。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 带 PyTorch 代码的清晰教学材料。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — variance-reduction 与 natural-gradient 视角，把 REINFORCE 连到 trust-region 家族（TRPO、PPO）。
