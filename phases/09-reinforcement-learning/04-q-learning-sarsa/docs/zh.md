# Temporal Difference — Q-Learning & SARSA / 时间差分：Q-Learning 与 SARSA

> Monte Carlo 要等 episode 结束。TD 通过 bootstrap 下一个 value estimate，每一步都能更新。Q-learning 是 off-policy 且偏乐观；SARSA 是 on-policy 且更谨慎。二者都只有一行核心代码，也支撑了本 phase 的所有 deep-RL 方法。

**类型：** 构建
**语言：** Python
**前置知识：** 第 09 阶段 · 01（MDPs）, 第 09 阶段 · 02（Dynamic Programming）, 第 09 阶段 · 03（Monte Carlo）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 TD 如何用单步 transition 同时避开完整 episode 和已知 model。
- 实现 SARSA 与 Q-learning，并定位两者 target 的差异。
- 说明 on-policy 与 off-policy 在探索行为上的实际影响。
- 用 learning curve 与 DP truth 验证 tabular TD 是否健康。
- 识别 α、ε、max bias、episode cap 等常见调参风险。

## The Problem / 问题

Monte Carlo 能工作，但有两个昂贵要求。它需要 episode 结束，并且只能在最终 return 出来后更新。如果一个 episode 有 1,000 步，MC 要等 1,000 步才更新任何东西。它高方差、低偏差，但实践中慢。

Dynamic programming 正好相反：零方差的 bootstrapped backups，但要求已知 model。

Temporal difference (TD) learning 折中二者。只用一个 transition `(s, a, r, s')`，构造一步 target `r + γ V(s')`，然后把 `V(s)` 往这个 target 推。无需 model。无需完整 episode。因为 RHS 用了近似的 `V`，会引入 bias；但方差远低于 MC，而且从第一步开始就能 online update。

现代 RL——DQN、A2C、PPO、SAC——都围绕这个 pivot 展开。Phase 9 后续内容，就是在你本课写出的 one-step TD update 上叠加 function approximation 和各种稳定技巧。

## The Concept / 概念

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**TD(0) update for V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

方括号里的量是 TD error：`δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的 online 版本。收敛要求 `α` 满足 Robbins-Monro（`Σ α = ∞`，`Σ α² < ∞`），并且所有状态都被无限访问。

**Q-learning.** 控制问题中的 off-policy TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

其中 `max` 假设从 `s'` 之后会遵循 *greedy* policy，而不管 Agent 实际接下来采取什么动作。这个解耦让 Q-learning 在 Agent 通过 ε-greedy 探索时，仍然学习 `Q*`。Mnih et al. (2015) 把它扩展成 Atari 上的 deep Q-learning（Lesson 05）。

**SARSA.** on-policy TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名字来自 tuple `(s, a, r, s', a')`。SARSA 使用 Agent 下一步 *实际* 采取的动作 `a'`，而不是 greedy `argmax`。它收敛到当前 ε-greedy `π` 的 `Q^π`；当 `ε → 0` 时，极限变成 `Q*`。

**The cliff-walking difference.** 在经典 cliff-walking 任务（掉下悬崖 = reward -100）中，Q-learning 会学到贴着悬崖边走的 optimal path，但探索时偶尔会踩空吃惩罚。SARSA 会学到离悬崖一格的更安全路径，因为它把探索噪声也计入 Q-value。训练后，当 `ε → 0`，二者都会达到 optimal。实践中这很重要：如果部署时仍会探索，SARSA 的行为更保守。

**Expected SARSA.** 用 `π` 下的期望值替换 `Q(s', a')`：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

它比 SARSA 方差更低（不用 sample `a'`），但目标仍是 on-policy。现代教材中常把它作为默认版本。

**n-step TD and TD(λ).** 在 TD(0) 和 MC 之间插值：等 `n` 步再 bootstrap。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 求平均。多数 deep-RL 使用 3 到 20 之间的 `n`。

```figure
qlearning-gridworld
```

## Build It / 动手构建

### Step 1: SARSA on ε-greedy policy / 在 ε-greedy policy 上运行 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。和 Q-learning 的 *唯一* 区别就是 target 那一行。

### Step 2: Q-learning / Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 把 target 与 behavior 解耦。这个符号就是 on-policy 和 off-policy 的分界线。

### Step 3: learning curves / 学习曲线

每 100 个 episode 跟踪一次 mean return。在简单确定性 GridWorld 上，Q-learning 收敛更快；在 cliff-walking 中，SARSA 更保守。在 `code/main.py` 的 4×4 GridWorld 上，使用 `α=0.1, ε=0.1` 时，二者大约在 2,000 个 episode 后接近 optimal。

### Step 4: compare to DP truth / 与 DP truth 比较

运行 value iteration（Lesson 02）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。健康的 tabular TD agent 在 4×4 GridWorld 上跑 10,000 个 episode 后，通常能落在 `~0.5` 以内。

## Pitfalls / 常见陷阱

- **Initial Q values matter.** Optimistic init（负奖励任务中 `Q = 0`）会鼓励探索。Pessimistic init 可能让 greedy policy 永远困住。
- **α schedule.** 常数 `α` 适合 non-stationary problem。衰减 `α_n = 1/n` 理论上收敛，但实践中太慢；把 `α` 固定在 `[0.05, 0.3]`，同时监控 learning curve。
- **ε schedule.** 从高值开始（`ε=1.0`），衰减到 `ε=0.05`。“GLIE”（greedy in the limit with infinite exploration）是收敛条件。
- **Max bias in Q-learning.** 当 `Q` 有噪声时，`max` operator 有向上偏差，会导致 overestimation。Hasselt 的 Double Q-learning（Lesson 05 的 DDQN 使用它）用两张 Q table 修复。
- **Non-terminating episodes.** TD 可以在没有 terminal 的情况下学习，但你需要 cap steps，或在 cap 处正确 bootstrap。标准做法：把 cap 当作 non-terminal，继续 bootstrap。
- **State hashing.** 如果 state 是 tuple/tensor，使用 hashable key（tuple，不要 list；float tuple 先 round，不要用原始浮点）。

## Use It / 应用它

2026 年 TD 方法版图：

| Task | Method | Reason |
|------|--------|--------|
| Small tabular environments | Q-learning | Learns optimal policy directly. |
| On-policy safety-critical | SARSA / Expected SARSA | Conservative during exploration. |
| High-dimensional state | DQN (Phase 9 · 05) | Neural-net Q-function with replay and target net. |
| Continuous actions | SAC / TD3 (Phase 9 · 07) | TD update on a Q-network; policy net emits actions. |
| LLM RL (reward-model-based) | PPO / GRPO (Phase 9 · 08, 12) | Actor-critic with TD-style advantage via GAE. |
| Offline RL | CQL / IQL (Phase 9 · 08) | Q-learning with conservative regularization. |

你在 2026 年论文里读到的 90% “RL”，都是 Q-learning 或 SARSA 的某种扩展。先把 tabular update 写到手指里，再读更深的内容。

## Ship It / 交付它

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Exercises / 练习

1. **Easy.** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。画出 2,000 个 episode 的 learning curves（每 100 个 episode 的 mean return）。谁收敛更快？
2. **Medium.** 构建一个 cliff-walking environment（4×12，最后一行是 cliff，reward -100，并 reset 到起点）。比较 Q-learning 和 SARSA 的最终 policies。截取各自路径。谁更贴近悬崖？
3. **Hard.** 实现 Double Q-learning。在 noisy-reward GridWorld（每步奖励加 Gaussian noise σ=5）上，展示 Q-learning 会明显高估 `V*(0,0)`，而 Double Q-learning 不会。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| TD error | “The update signal” | `δ = r + γ V(s') - V(s)`，bootstrapped residual。 |
| TD(0) | “One-step TD” | 每个 transition 后更新，只使用 next state's estimate。 |
| Q-learning | “Off-policy RL 101” | 对 next-state actions 取 `max` 的 TD update；无论 behavior policy 如何，都学习 `Q*`。 |
| SARSA | “On-policy Q-learning” | 使用实际 next action 的 TD update；学习当前 ε-greedy π 的 `Q^π`。 |
| Expected SARSA | “The low-variance SARSA” | 用 π 下的 expectation 替换 sampled `a'`。 |
| GLIE | “Correct exploration schedule” | Greedy in the Limit with Infinite Exploration；Q-learning 收敛所需条件。 |
| Bootstrapping | “Using current estimate in the target” | TD 区别于 MC 的地方；带来 bias，但大幅降低 variance。 |
| Maximization bias | “Q-learning overestimates” | 对 noisy estimates 取 `max` 会向上偏；Double Q-learning 修复。 |

## Further Reading / 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文和收敛证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — maximization bias 的修复。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — expected SARSA 的动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 提出 SARSA 名称的论文（当时称为 “modified connectionist Q-learning”）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 把 TD(0) 推广到 TD(n)，也就是从 Q-learning 通向 eligibility traces、再通向 PPO 中 GAE 的路径。
