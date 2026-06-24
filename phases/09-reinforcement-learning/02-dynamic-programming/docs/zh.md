# Dynamic Programming — Policy Iteration & Value Iteration / 动态规划：策略迭代与价值迭代

> Dynamic programming 是“作弊版”RL：你已经知道 transition 和 reward 函数，只需要反复迭代 Bellman equation，直到 `V` 或 `π` 不再移动。它是所有 sampling-based 方法努力逼近的 benchmark。

**类型：** 构建
**语言：** Python
**前置知识：** 第 09 阶段 · 01（MDPs）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 说明 dynamic programming 为什么需要已知 MDP model。
- 实现 policy evaluation、policy improvement 和完整 policy iteration。
- 实现 value iteration，并理解它与 policy iteration 的差异。
- 用 sup-norm convergence 判断 Bellman 迭代是否收敛。
- 把 DP 结果作为 Q-learning、PPO 等采样方法的正确性基线。

## The Problem / 问题

你有一个 model 已知的 MDP：对任意 state-action pair，都能查询 `P(s' | s, a)` 和 `R(s, a, s')`。库存管理器知道需求分布。棋类游戏有确定性转移。gridworld 只要四行 Python。你拥有一个 *model*。

Model-free RL（Q-learning、PPO、REINFORCE）是为没有 model 的场景发明的：你只能从环境中采样。但当你确实有 model 时，有更快、更好的方法：dynamic programming。Bellman 在 1957 年设计了这些方法。它们今天仍然定义“正确性”：当人们说“这个 MDP 的 optimal policy”时，意思就是 DP 会返回的 policy。

2026 年你仍然需要它，原因有三点。第一，RL 研究中的每个 tabular environment（GridWorld、FrozenLake、CliffWalking）都会用 DP 解出 gold-standard policy。第二，精确 value 可以用来 *debug* 采样方法：如果 Q-learning 对 `V*(s_0)` 的估计和 DP 答案差 30%，那是 Q-learning 有 bug。第三，现代 offline RL 和 planning 方法（MCTS、AlphaZero 的 search、Phase 9 · 10 里的 model-based RL）本质上都在给定或学习到的 model 上迭代 Bellman backup。

## The Concept / 概念

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**两个算法，都是 Bellman fixed-point iteration。**

**Policy iteration.** 在两步之间交替，直到 policy 不再变化。

1. *Evaluation:* 给定 policy `π`，反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`，直到计算出 `V^π`。
2. *Improvement:* 给定 `V^π`，让 `π` 对 `V^π` 变 greedy：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

它保证收敛，因为 (a) 每次 improvement 要么保持 `π` 不变，要么严格提高某些状态的 `V^π`，(b) 确定性 policy 的空间是有限的。即使状态空间很大，通常也只需要约 5–20 个 outer iteration。

**Value iteration.** 把 evaluation 和 improvement 压成一次 sweep。应用 Bellman *optimality* equation：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后用 greedy action 提取 policy。每次迭代更快，因为没有内层 evaluation loop，但通常需要更多轮才收敛。

**Generalized policy iteration (GPI).** 这是统一视角。value function 和 policy 锁在一个双向改进循环中；任何把二者推向相互一致的方法（async value iteration、modified policy iteration、Q-learning、actor-critic、PPO）都是 GPI 的实例。

**为什么 `γ < 1` 重要。** Bellman operator 在 sup-norm 下是 `γ`-contraction：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。contraction 意味着唯一 fixed point 和几何收敛。去掉 `γ < 1`，保证就没了；你需要 finite horizon 或 absorbing terminal state。

```figure
value-iteration-gamma
```

## Build It / 动手构建

### Step 1: build the GridWorld MDP model / 构建 GridWorld MDP model

使用 Lesson 01 的 4×4 GridWorld。这里加一个随机版本：Agent 以 `0.1` 的概率 slip 到随机垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回 `(s', r, p)` 列表。这就是整个 model。

### Step 2: policy evaluation / 策略评估

给定 policy `π(s) = {action: prob}`，迭代 Bellman equation，直到 `V` 不再移动：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Step 3: policy improvement / 策略改进

用相对于 `V` 的 greedy policy 替换 `π`。如果 `π` 没变，就返回；我们已经到达 optimum。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Step 4: stitch them together / 串起来

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4 上通常 4–6 个 outer iteration 就收敛。输出 `V*(0,0) ≈ -6`，以及一个严格减少步数的 policy。

### Step 5: value iteration (the one-loop version) / 价值迭代：单循环版本

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

同一个 fixed point，更少代码。

## Pitfalls / 常见陷阱

- **Forgetting to handle terminals.** 如果对 absorbing state 继续套 Bellman，它仍会挑出一个“best action”，虽然什么也不会改变。用 `if s == terminal: V[s] = 0` 做保护。
- **Sup-norm vs L2 convergence.** 用 `max |V_new - V|`，不要用平均值。理论保证是在 sup-norm 上成立的。
- **In-place vs synchronous updates.** 原地更新 `V[s]`（Gauss-Seidel）比单独维护 `V_new` dict（Jacobi）收敛更快。生产代码通常用原地更新。
- **Policy ties.** 如果两个动作 Q-value 相同，`argmax` 可能每轮以不同方式打破平局，导致 “policy stable” 检查震荡。使用稳定 tie-break（固定顺序中的第一个动作）。
- **State-space explosion.** DP 每次 sweep 是 `O(|S| · |A|)`。能处理到大约 `10⁷` 级别状态。再往上就需要 function approximation（Phase 9 · 05 起）。

## Use It / 应用它

2026 年，DP 是正确性基线，也是 planner 的内层循环：

| Use case | Method |
|----------|--------|
| Solve a small tabular MDP exactly | Value iteration (simpler) or policy iteration (fewer outer steps) |
| Verify a Q-learning / PPO implementation | Compare to DP-optimal V* on a toy environment |
| Model-based RL (Phase 9 · 10) | Bellman backup on a learned transition model |
| Planning in AlphaZero / MuZero | Monte Carlo Tree Search = async Bellman backup |
| Offline RL (CQL, IQL) | Conservative Q-iteration — DP with a penalty on OOD actions |

每当有人说 “the optimal value function”，意思就是 “DP fixed point”。当你在论文中看到 `V*` 或 `Q*`，脑中应该浮现这个循环。

## Ship It / 交付它

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Exercises / 练习

1. **Easy.** 在 4×4 GridWorld 上用 `γ ∈ {0.9, 0.99}` 运行 value iteration。到 `max |ΔV| < 1e-6` 需要多少个 sweep？把 `V*` 打印成 4×4 网格。
2. **Medium.** 在 *stochastic* GridWorld（slip probability `0.1`）上比较 policy iteration 与 value iteration。统计 sweep 数、wall-clock time、最终 `V*(0,0)`。哪个按迭代数收敛更快？哪个按 wall-clock 更快？
3. **Hard.** 构建 modified policy iteration：evaluation step 不跑到收敛，只跑 `k` 个 sweep。对 `k ∈ {1, 2, 5, 10, 50}`，画出 `V*(0,0)` error vs `k`。这条曲线说明了 evaluation/improvement 的什么 tradeoff？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Policy iteration | “DP algorithm” | 在 evaluation（`V^π`）和 improvement（相对 `V^π` 的 greedy `π`）之间交替，直到 policy 不再变化。 |
| Value iteration | “Faster DP” | 一次 sweep 应用 Bellman optimality backup；几何收敛到 `V*`。 |
| Bellman operator | “The recursion” | `(T V)(s) = max_a Σ P (r + γ V(s'))`；在 sup-norm 下是 `γ`-contraction。 |
| Contraction | “Why DP converges” | 任何满足 `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` 的 operator 都有唯一 fixed point。 |
| GPI | “Everything is DP” | Generalized Policy Iteration：任何把 `V` 和 `π` 推向相互一致的方法。 |
| Synchronous update | “Jacobi-style” | 整个 sweep 中都使用旧 `V`；便于分析但更慢。 |
| In-place update | “Gauss-Seidel-style” | 使用正在更新中的 `V`；实践中收敛更快。 |

## Further Reading / 延伸阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — policy iteration 与 value iteration 的经典讲法。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — contraction-mapping 论证的严谨处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — modified policy iteration 及其收敛分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 最早的 policy iteration 论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 从 DP 走向 approximate-DP / deep RL 的桥梁，后续课程都会用到。
