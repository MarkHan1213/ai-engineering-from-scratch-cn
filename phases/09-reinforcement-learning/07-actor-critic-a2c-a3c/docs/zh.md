# Actor-Critic — A2C and A3C / Actor-Critic：A2C 与 A3C

> REINFORCE 噪声很大。加一个学习 `V̂(s)` 的 critic，把它从 return 中减掉，就得到期望相同但方差低得多的 advantage。这就是 actor-critic。A2C 同步运行，A3C 跨线程异步运行。二者是所有现代 deep-RL 方法的心智模型。

**类型：** 构建
**语言：** Python
**前置知识：** 第 09 阶段 · 04（TD Learning）, 第 09 阶段 · 06（REINFORCE）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 critic baseline 如何降低 policy gradient 的方差。
- 区分 MC advantage、TD advantage、n-step advantage 与 GAE。
- 实现 critic update、advantage computation 和 actor-critic combined update。
- 比较 A2C 与 A3C 的并行化方式。
- 理解 PPO、IMPALA、SAC、GRPO 与 actor-critic 架构的关系。

## The Problem / 问题

Vanilla REINFORCE 能工作，但方差很糟。Monte Carlo returns `G_t` 在不同 episode 之间可能差一个数量级。把这种噪声乘以 `∇ log π` 再求平均，会得到一个非常慢的 gradient estimator；它可能需要几千个 episode 才能移动 policy，而 DQN 少得多的 updates 就能做到。

方差来自直接使用 raw returns。如果你减去 baseline `b(s_t)`——任何 state 函数，包括学出来的 value——期望不变，方差下降。可实现的最好 baseline 是 `V̂(s_t)`。于是乘以 `∇ log π` 的量变成 *advantage*：

`A(s, a) = G - V̂(s)`

一个动作好不好，取决于它是否产生了高于平均水平的 return；低于平均则不好。带 learned critic 的 REINFORCE 就是 *actor-critic*。critic 给 actor 一个低方差 teacher。这就是 2015 年之后所有 deep-policy 方法（A2C、A3C、PPO、SAC、IMPALA）。

## The Concept / 概念

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**两个网络，一个 shared loss：**

- **Actor** `π_θ(a | s)`：policy。用于采样行动，并通过 policy gradient 训练。
- **Critic** `V_φ(s)`：估计从 state 出发的 expected return。通过最小化 `(V_φ(s) - target)²` 训练。

**Advantage.** 两种标准形式：

- *MC advantage:* `A_t = G_t - V_φ(s_t)`。无偏，方差更高。
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），但方差远低。也叫 *TD residual* `δ_t`。

**n-step advantage.** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。多数实现中，Atari 用 `n = 5`，PPO on MuJoCo 用 `n = 2048`。

**Generalized Advantage Estimation (GAE).** Schulman et al. (2016) 提出对所有 n-step advantages 做指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差、高偏差）。`λ = 1` 是 MC（高方差、无偏）。`λ = 0.95` 是 2026 年默认值；调它就是在调 bias/variance 旋钮。

**A2C: synchronous advantage actor-critic.** 在 `N` 个 parallel environments 中各收集 `T` 步。为每一步计算 advantages。把所有 batch 合起来更新 actor 与 critic。重复。它是 A3C 更简单、更可扩展的兄弟。

**A3C: asynchronous advantage actor-critic.** Mnih et al. (2016)。启动 `N` 个 worker threads，每个跑一个 env。每个 worker 在自己的 rollout 上本地计算 gradients，然后异步应用到共享 parameter server。不需要 replay buffer，worker 通过不同 trajectories 去相关。A3C 证明了在 CPU 上可以大规模训练。到 2026 年，GPU-based A2C（batched parallel envs）占主导，因为 GPU 更喜欢大 batch。

**Combined loss.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：policy-gradient loss、value regression、entropy bonus。`c_v ~ 0.5`、`c_e ~ 0.01` 是经典起点。

## Build It / 动手构建

### Step 1: a critic / 一个 critic

Linear critic `V_φ(s) = w · features(s)`，用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在 tabular env 上，critic 几百个 episode 就会收敛。在 Atari 上，把 linear critic 换成 shared CNN trunk + value head。

### Step 2: n-step advantage / n-step advantage

给定长度为 `T` 的 rollout，以及最后 bootstrap 的 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic target。`advantages` 是乘以 `∇ log π` 的量。

### Step 3: combined update / 联合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy，每个 rollout 更新一次；actor 和 critic 使用分开的 learning rates。

### Step 4: parallelization (A3C vs A2C) / 并行化（A3C vs A2C）

- **A3C:** 启动 `N` 个 threads。每个线程运行自己的 env 和 forward pass。周期性把 gradient updates 推到 shared master。master 上不加锁也可以；race 只会增加噪声。
- **A2C:** 在单进程里运行 `N` 个 env instances，把 observations 堆成 `[N, obs_dim]` batch，做 batched forward pass 和 batched backward pass。GPU 利用率更高、确定性更强、也更容易推理。2026 年默认选这个。

我们的 toy code 为清晰起见是单线程；改成 batched A2C 只需要三行 numpy。

## Pitfalls / 常见陷阱

- **Critic bias before actor gradient.** 如果 critic 还是随机的，它的 baseline 没有信息量，你其实在纯噪声上训练。先 warm up critic 几百步，再开启 policy gradient；或者使用较慢的 actor learning rate。
- **Advantage normalization.** 每个 batch 把 advantages 归一成 zero-mean/unit-std。几乎零成本，却能大幅稳定训练。
- **Shared trunk.** 图像输入上，actor 和 critic 使用 shared feature extractor，再接 separate heads。共享特征能同时受益于两个 losses。
- **On-policy contract.** A2C 的数据只复用一次。复用更多会让 gradient 有偏（PPO 添加的就是 importance-sampling correction）。
- **Entropy collapse.** 没有 `c_e > 0`，policy 几百次 update 后就会接近确定性并停止探索。
- **Reward scale.** Advantage magnitude 依赖 reward scale。使用 reward normalization（例如除以 running std）让不同任务的梯度量级一致。

## Use It / 应用它

A2C/A3C 在 2026 年很少是最终选择，但它们是后续方法不断精炼的 architecture：

| Method | Relation to A2C |
|--------|----------------|
| PPO | A2C + clipped importance ratio for multi-epoch updates |
| IMPALA | A3C + V-trace off-policy correction |
| SAC (Phase 9 · 07) | Off-policy A2C with a soft-value critic (next lesson) |
| GRPO (Phase 9 · 12) | A2C without the critic — group-relative advantage |
| DPO | A2C collapsed into a preference-ranking loss, no sampling |
| AlphaStar / OpenAI Five | A2C with league training + imitation pre-training |

如果你在 2026 年论文里看到 “advantage”，先想到 actor-critic。

## Ship It / 交付它

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## Exercises / 练习

1. **Easy.** 在 4×4 GridWorld 上用 MC advantage（`G_t - V(s_t)`）训练 actor-critic。和 Lesson 06 中 REINFORCE-with-running-mean-baseline 的 sample efficiency 对比。
2. **Medium.** 切换到 TD-residual advantage（`r + γ V(s') - V(s)`）。测量 advantage batches 的 variance。下降了多少？
3. **Hard.** 实现 GAE(λ)。扫 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。画 final return vs sample efficiency。这个任务的 bias/variance sweet spot 在哪里？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Actor | “The policy net” | `π_θ(a\|s)`，由 policy gradient 更新。 |
| Critic | “The value net” | `V_φ(s)`，通过 MSE regression 到 returns / TD targets 更新。 |
| Advantage | “How much better than average” | `A(s, a) = Q(s, a) - V(s)` 或其 estimator；是 `∇ log π` 的乘子。 |
| TD residual | “δ” | `δ_t = r + γ V(s') - V(s)`；one-step advantage estimate。 |
| GAE | “The interpolation knob” | n-step advantages 的指数加权和，由 `λ` 参数化。 |
| A2C | “Synchronous actor-critic” | 跨 env batching；每个 rollout 做一次 gradient step。 |
| A3C | “Async actor-critic” | Worker threads 把 gradients 推到 shared param server。原始论文方法；2026 年较少见。 |
| Bootstrap | “Use V at the horizon” | 截断 rollout，用 `γ^n V(s_{t+n})` 闭合求和。 |

## Further Reading / 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C 原始 async actor-critic 论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础内容；当 critic 是神经网络时，配合第 9 章 function approximation 一起读。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 可扩展 distributed actor-critic，带 V-trace off-policy correction。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的生产 A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — two-timescale actor-critic 分解的基础收敛结果。
