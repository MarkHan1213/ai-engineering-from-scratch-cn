# Monte Carlo Methods — Learning from Complete Episodes / 蒙特卡洛方法：从完整回合中学习

> Dynamic programming 需要 model。Monte Carlo 只需要 episode（回合）。运行 policy，观察 return（回报），然后取平均。这是 RL 中最简单的想法，也是后续所有方法的入口。

**类型：** 构建
**语言：** Python
**前置知识：** 第 09 阶段 · 01（MDPs）, 第 09 阶段 · 02（Dynamic Programming）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 Monte Carlo 如何只用完整 episode 估计 `V^π` 与 `Q^π`。
- 实现 rollout、return 反向计算和 first-visit MC evaluation。
- 用 ε-greedy 构建 on-policy MC control。
- 理解探索覆盖、方差和 non-stationary policy 的影响。
- 用 DP gold standard 检查 MC 估计的收敛质量。

## The Problem / 问题

Dynamic programming 很优雅，但它假设你能对每个状态和动作查询 `P(s' | s, a)`。现实世界几乎不是这样。机器人不能解析计算某个关节力矩之后的摄像头像素分布。定价算法无法积分所有可能的顾客反应。LLM 不能枚举某个 token 后的所有可能续写。

你需要一种只要求能从环境中 *sample* 的方法。运行 policy。得到一条 trajectory：`s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它估计 value。这就是 Monte Carlo。

从 DP 到 MC 的转变在思想上很重要：我们从 *known model + exact backup* 走向 *sampled rollouts + averaged return*。方差会升高，但适用范围会暴涨。后面的每个 RL 算法——TD、Q-learning、REINFORCE、PPO、GRPO——本质上都是 Monte Carlo estimator，只是有时在上面叠加了 bootstrapping。

## The Concept / 概念

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**核心想法，一行就够：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是在 policy `π` 下访问 `s` 后观察到的 returns。

**First-visit vs every-visit MC.** 如果一个 episode 多次访问状态 `s`，first-visit MC 只计算第一次访问后的 return；every-visit MC 计算所有访问。二者在极限下都是无偏的。First-visit 更容易分析（iid samples）。Every-visit 每个 episode 用到更多数据，实践中通常收敛更快。

**Incremental mean.** 不必存储所有 returns，可以维护 running average：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重写为：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。把 `1/n` 换成常数 step-size `α ∈ (0, 1)`，就得到一个能追踪 `π` 变化的 non-stationary MC estimator。这个动作就是从 MC 跳到 TD、再跳到现代 RL 的关键。

**Exploration is now a problem.** DP 通过枚举触达每个状态。MC 只能看到 policy 实际访问的状态。如果 `π` 是确定性的，状态空间中的大片区域永远不会被采样，value estimate 会一直停在零。三个修复办法按历史顺序是：

1. **Exploring starts.** 每个 episode 从随机 (s, a) pair 开始。能保证覆盖，但实践中不现实（你无法把机器人“重置”到任意状态）。
2. **ε-greedy.** 相对于当前 Q 采取 greedy 动作，但以概率 `ε` 随机选动作。渐近上所有 state-action pair 都会被采样。
3. **Off-policy MC.** 用 behavior policy `μ` 收集数据，通过 importance sampling 学 target policy `π`。方差很高，但它是 DQN 等 replay-buffer 方法的桥梁。

**Monte Carlo Control.** 和 policy iteration 一样，evaluate → improve → evaluate，只是 evaluation 由采样完成：

1. 运行 `π`，得到一个 episode。
2. 用观察到的 returns 更新 `Q(s, a)`。
3. 让 `π` 相对于 `Q` 变成 ε-greedy。
4. 重复。

在温和条件下（每个 pair 被无限访问，`α` 满足 Robbins-Monro），它以概率 1 收敛到 `Q*` 和 `π*`。

```figure
epsilon-greedy
```

## Build It / 动手构建

### Step 1: rollout → list of (s, a, r) / rollout 得到 (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

没有 model，只有 `env.reset()` 和 `env.step(s, a)`。这和 gym environment 的接口相同，只是被精简到最低限度。

### Step 2: compute returns (reverse sweep) / 反向扫一遍计算 returns

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一遍完成，`O(T)`。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免重复求和。

### Step 3: first-visit MC evaluation / first-visit MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

真正做事的是三行：第一次访问时标记 state，增加 count，更新 running mean。

### Step 4: ε-greedy MC control (on-policy) / ε-greedy MC control（on-policy）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Step 5: compare to DP gold standard / 与 DP gold standard 比较

随着 episodes → ∞，你对 `V^π` 的 MC 估计应该和 Lesson 02 的 DP 结果一致。实践中，在 4×4 GridWorld 上跑 50,000 个 episode，通常可以达到距离 DP 答案 `~0.1` 以内。

## Pitfalls / 常见陷阱

- **Infinite episodes.** MC 要求 episode *terminate*。如果 policy 可能无限循环，就设置 `max_steps`，并把 cap 视为隐式失败。随机 policy 下的 GridWorld 经常 timeout，这是正常的，但要正确计数。
- **Variance.** MC 使用完整 returns。长 episode 中方差巨大：末尾一次倒霉奖励会同等幅度地影响 `V(s_0)`。TD 方法（Lesson 04）通过 bootstrapping 降低这个问题。
- **State coverage.** 新 Q 上的 greedy MC 如果遇到 tie，只会尝试一个动作。你 *必须* 探索（ε-greedy、exploring starts、UCB）。
- **Non-stationary policies.** 如果 `π` 会变化（如 MC control），旧 returns 来自另一个 policy。Constant-α MC 能处理；sample-average MC 不能。
- **Off-policy importance sampling.** 权重 `π(a|s)/μ(a|s)` 会沿 trajectory 相乘，horizon 一长方差就爆炸。用 per-decision weighted IS 限制，或者切换到 TD。

## Use It / 应用它

2026 年 Monte Carlo methods 的位置：

| Use case | Why MC |
|----------|--------|
| Short-horizon games (blackjack, poker) | Episodes terminate naturally; returns are clean. |
| Offline evaluation of a logged policy | Average discounted returns over stored trajectories. |
| Monte Carlo Tree Search (AlphaZero) | MC rollouts from tree leaves guide selection. |
| LLM RL evaluation | Compute average reward over sampled completions for a given policy. |
| Baseline estimation in PPO | The advantage target `A_t = G_t - V(s_t)` uses an MC `G_t`. |
| Teaching RL | Simplest algorithm that actually works — strip bootstrapping to see the core. |

现代 deep-RL 算法（PPO、SAC）通过 `n`-step returns 或 GAE，在纯 MC（full returns）与纯 TD（one-step bootstrap）之间插值。两个端点都属于同一个 estimator 家族。

## Ship It / 交付它

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Exercises / 练习

1. **Easy / 简单。** 在 4×4 GridWorld 上实现 uniform-random policy 的 first-visit MC evaluation。运行 10,000 个 episode。画出 `V(0,0)` 随 episode count 的变化，并和 DP 答案对比。
2. **Medium / 中等。** 实现 ε-greedy MC control，取 `ε ∈ {0.01, 0.1, 0.3}`。比较 20,000 个 episode 后的 mean return。曲线是什么样？bias-variance tradeoff 体现在哪里？
3. **Hard / 困难。** 实现 *off-policy* MC with importance sampling：用 uniform-random policy `μ` 收集数据，估计 deterministic optimal policy `π` 的 `V^π`。比较 plain IS、per-decision IS 和 weighted IS。哪个方差最低？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Monte Carlo | “随机采样” | 通过对分布中的 iid samples 求平均来估计 expectation。 |
| Return `G_t` | “未来奖励” | 从 step `t` 到 episode 结束的折扣奖励和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| First-visit MC | “每个 state 只数一次” | 每个 episode 中只有第一次访问贡献 value estimate。 |
| Every-visit MC | “所有访问都用上” | 每次访问都贡献；略有偏差但 sample-efficient。 |
| ε-greedy | “Exploration noise” | 以概率 `1-ε` 选 greedy action；以概率 `ε` 随机选 action。 |
| Importance sampling | “纠正从错误分布采样” | 用 `π(a\|s)/μ(a\|s)` 的乘积重加权 returns，从 `μ` 数据估计 `V^π`。 |
| On-policy | “用自己的数据学” | Target policy = behavior policy。Vanilla MC、PPO、SARSA。 |
| Off-policy | “用别人的数据学” | Target policy ≠ behavior policy。Importance-sampled MC、Q-learning、DQN。 |

## Further Reading / 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 经典教材章节。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — first-visit vs every-visit 分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC 与方差控制。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代低方差 IS estimator。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD self-play 收敛到超人水平的早期大规模实证；概念上是本 phase 后半部分所有课程的前身。
