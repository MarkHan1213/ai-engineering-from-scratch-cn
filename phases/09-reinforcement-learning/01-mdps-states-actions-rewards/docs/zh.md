# MDPs, States, Actions & Rewards / MDP、状态、动作与奖励

> 马尔可夫决策过程可以拆成五件事：状态、动作、转移、奖励、折扣。RL 里的 Q-learning、PPO、DPO、GRPO，最终都在这个结构上优化。把它学透，后面的强化学习会顺很多。

**类型：** 学习
**语言：** Python
**前置知识：** 第 01 阶段 · 06（Probability & Distributions）, 第 02 阶段 · 01（ML Taxonomy）
**时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 把任意 RL 任务拆成 `(S, A, P, R, γ)` 五个对象。
- 解释 Markov property 对状态表示的要求。
- 区分 policy、return、value 与 Q-value。
- 用 Bellman equation 精确计算小型 MDP 的 `V^π`。
- 根据任务 horizon 选择合适的 discount `γ`。

## The Problem / 问题

你在写一个国际象棋机器人。或者库存规划器。或者交易 Agent。或者训练推理模型的 PPO 循环。四个领域完全不同，但有一个很关键的共同点：它们都能压缩成同一个数学对象。

监督学习给你 `(x, y)` 对，让你拟合一个函数。强化学习没有标签，只有一串状态、你采取的动作，以及一个标量奖励。刚才那一步棋赢了吗？补货决策省钱了吗？交易赚钱吗？LLM 刚生成的 token 是否让 judge 给出更高奖励？

如果不先形式化，你无法从这条流里学习。“我看到了什么”“我做了什么”“接下来发生了什么”“这件事有多好”，每一项都必须变成可推理的对象。这套形式化就是 Markov Decision Process。Phase 9 里的所有 RL 算法，包括后面的 RLHF 与 GRPO 循环，都是在这个结构上优化。

## The Concept / 概念

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**五个对象。**

- **States** `S`。Agent 做决策所需的全部信息。GridWorld 里是格子；国际象棋里是棋盘；LLM 里是上下文窗口加上任何外部记忆。
- **Actions** `A`。可选择的动作。向上/下/左/右移动，下棋，或者生成一个 token。
- **Transitions** `P(s' | s, a)`。给定状态 `s` 和动作 `a` 后，下一个状态的分布。国际象棋里是确定性的，库存里是随机的，LLM 解码里通常近似确定。
- **Rewards** `R(s, a, s')`。标量反馈。赢 = +1，输 = -1；收入减成本；GRPO 里的 log-likelihood ratio 项。
- **Discount** `γ ∈ [0, 1)`。未来奖励相对当前奖励的权重。`γ = 0.99` 大约对应 100 步 horizon；`γ = 0.9` 大约对应 10 步。

**Markov property** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只依赖当前状态。如果这个条件不成立，说明状态表示不完整；这不是方法失败，而是状态定义失败。

**Policies and returns.** policy `π(a | s)` 把状态映射到动作分布。return `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣和。value `V^π(s) = E[G_t | s_t = s]` 是从 `s` 出发并遵循 policy `π` 时的期望回报。Q-value `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是从某个具体动作开始的期望回报。每个 RL 算法都在估计这两者之一，然后据此改进 `π`。

**Bellman equations.** Phase 9 中所有方法都会用到的 fixed-point equations：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

它们把期望回报拆成“这一步的奖励”加上“落点状态的折扣价值”。这是递归结构。Phase 9 的每个算法要么迭代这个方程直到收敛（dynamic programming），要么从中采样（Monte Carlo），要么用一步 bootstrap 近似它（temporal difference）。

```figure
discount-horizon
```

## Build It / 动手构建

### Step 1: a tiny deterministic MDP / 一个极小的确定性 MDP

一个 4×4 GridWorld。Agent 从左上角出发，终点在右下角，每走一步奖励 -1，动作是 `{up, down, left, right}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行就是整个环境：确定性转移、固定步进惩罚、吸收终止状态。

### Step 2: roll out a policy / 展开一个 policy

policy 是从状态到动作分布的函数。最简单的 policy 是均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

把随机 policy 跑 1000 次。在这个 4×4 棋盘上，平均 return 大约在 -60 到 -80。最优 return 是 -6（沿直线向下再向右）。Phase 9 的核心，就是缩小这个差距。

### Step 3: compute `V^π` exactly via the Bellman equation / 用 Bellman equation 精确计算 `V^π`

对小型 MDP，Bellman equation 是一个线性系统。枚举状态，套入期望，反复迭代直到 value 不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是 iterative policy evaluation。它是 Sutton & Barto 里的第一个算法，也是后续所有 RL 方法的理论地基。

### Step 4: `γ` is a hyperparameter with physical meaning / `γ` 是有物理含义的超参数

effective horizon 大致是 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

太低，Agent 会短视。太高，credit assignment 会变得很嘈杂，因为很多早期动作都要为很远之后的奖励共同负责。LLM RLHF 通常使用 `γ = 1`，因为 episode 很短且有边界。控制任务常用 `0.95–0.99`。长 horizon 策略游戏会用 `0.999`。

## Pitfalls / 常见陷阱

- **Non-Markovian state.** 如果你需要最近三帧观察才能决策，那么“状态”就不只是当前观察。修复方式：堆叠帧（Atari 上的 DQN 堆叠 4 帧）或使用 recurrent state（在观察序列上用 LSTM/GRU）。
- **Sparse rewards.** 在大状态空间里，只给胜负奖励几乎学不动。可以做 reward shaping（中间信号），或者用 imitation bootstrap（Phase 9 · 09）。
- **Reward hacking.** 优化 proxy reward 往往会产生病态行为。OpenAI 的赛艇 Agent 曾经不停转圈刷 powerups，而不是冲向终点。奖励要从目标结果定义，不要从代理指标定义。
- **Discount mis-spec.** 在无限 horizon 任务上用 `γ = 1` 会让所有 value 变成无穷。要么使用有限 horizon，要么保证 `γ < 1`。
- **Reward scale.** {+100, -100} 与 {+1, -1} 的最优 policy 相同，但梯度量级完全不同。接入 PPO/DQN 前，通常把奖励归一到接近 `[-1, 1]`。

## Use It / 应用它

2026 年的 RL stack 会在写代码前先把所有 RL pipeline 规约成 MDP：

| Situation | State | Action | Reward | γ |
|-----------|-------|--------|--------|---|
| Control (locomotion, manipulation) | Joint angles + velocities | Continuous torques | Task-specific shaped | 0.99 |
| Games (chess, Go, poker) | Board + history | Legal move | Win=+1 / loss=-1 | 1.0 (finite) |
| Inventory / pricing | Stock + demand | Order qty | Revenue - cost | 0.95 |
| RLHF for LLMs | Context tokens | Next token | Reward-model score at end | 1.0 (episode ~200 tokens) |
| GRPO for reasoning | Prompt + partial response | Next token | Verifier 0/1 at end | 1.0 |

在写任何 training loop 之前，先写出这五元组。大多数“RL 不工作”的 bug report，根因都能追溯到纸面上的 MDP formulation 已经坏了。

## Ship It / 交付它

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Exercises / 练习

1. **Easy.** 在 `code/main.py` 中实现 4×4 GridWorld 和 random-policy rollout。运行 10,000 个 episode，报告 return 的 mean 和 std，并与最优 return（-6）比较。
2. **Medium.** 对 uniform-random policy，用 `γ ∈ {0.5, 0.9, 0.99}` 运行 `policy_evaluation`。把每个 `V` 打印成 4×4 网格。解释为什么终点附近的 state value 在更大的 `γ` 下增长更快。
3. **Hard.** 把 GridWorld 改成随机环境：每个动作以 `p = 0.1` 的概率 slip 到相邻方向。重新评估 uniform policy。`V[start]` 变好还是变差？为什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| MDP | “强化学习设定” | 满足 Markov property 的元组 `(S, A, P, R, γ)`。 |
| State | “Agent 看到的东西” | 对未来 dynamics 足够充分的统计量，前提是 policy class 已选定。 |
| Policy | “Agent 的行为” | 条件分布 `π(a \| s)` 或确定性映射 `s → a`。 |
| Return | “总奖励” | 从当前步开始的折扣和 `Σ γ^t r_t`。 |
| Value | “一个状态有多好” | 从 `s` 出发并遵循 `π` 时的期望 return。 |
| Q-value | “一个动作有多好” | 从 `s` 出发且第一步采取 `a` 时，在 `π` 下的期望 return。 |
| Bellman equation | “动态规划递归” | 把 value / Q 分解成一步奖励加折扣后的 successor value 的 fixed-point。 |
| Discount `γ` | “未来 vs 当前” | 远期奖励的几何权重；effective horizon 约为 `~1/(1-γ)`。 |

## Further Reading / 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 经典教材。第 3 章讲 MDP 与 Bellman equations；第 1 章解释贯穿后续课程的 reward hypothesis。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — Bellman equation 的源头。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从 deep RL 角度写的简洁 MDP 入门。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — MDP 与精确求解方法的 operations-research 参考书。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 把 MDP 推导为 dynamic-programming 特例的清晰材料。
