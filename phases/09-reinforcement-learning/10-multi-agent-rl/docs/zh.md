# Multi-Agent RL / 多智能体强化学习

> Single-agent RL 假设环境是 stationary。把两个学习中的 Agent 放进同一个世界，这个假设就破了：每个 Agent 都是另一个 Agent 的环境的一部分，而且双方都在变化。Multi-agent RL 就是在 Markov assumption 不再干净成立时，让学习仍能收敛的一组技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 第 09 阶段 · 04（Q-learning）, 第 09 阶段 · 06（REINFORCE）, 第 09 阶段 · 07（Actor-Critic）
**时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 把 multi-agent problem 形式化为 Markov game。
- 解释 non-stationarity、credit assignment、joint action explosion 等 MARL 难点。
- 实现 cooperative GridWorld 和 independent Q-learning baseline。
- 区分 IQL/IPPO、CTDE、self-play、league play 等训练范式。
- 判断不同任务应使用 MAPPO、QMIX、self-play、league 或通信机制。

## The Problem / 问题

一个机器人学习在房间里导航，是 single-agent RL。一个足球队不是。AlphaStar 对抗 StarCraft 对手不是。竞价 Agent 市场不是。两辆车在四向停车路口协商也不是。现实世界许多 many-on-many 问题都不是。

在任何 multi-agent setting 中，从某个 Agent 的视角看，其他 Agents *就是* 环境的一部分。当它们学习并改变行为时，环境就变成 non-stationary。Markov property——“next state 只依赖 current state 和我的 action”——被破坏，因为 next state 还依赖 *其他* Agents 选择了什么，而它们的 policies 也在移动。

这会打破 tabular convergence proofs（Q-learning 的保证假设环境 stationary）。也会打破朴素 deep RL：Agents 互相追着对方改变，在循环中打转，永远无法收敛到稳定 policy。你需要 multi-agent-specific techniques：centralized training / decentralized execution、counterfactual baselines、league play、self-play。

2026 年应用包括 robot swarms、traffic routing、autonomous vehicle fleets、market simulators、multi-agent LLM systems（Phase 16），以及任何有多个智能玩家的游戏。

## The Concept / 概念

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**形式化：Markov Game。** MDP 的泛化：states `S`，joint action `a = (a_1, …, a_n)`，transition `P(s' | s, a)`，以及每个 Agent 的 reward `R_i(s, a, s')`。每个 Agent `i` 在自己的 policy `π_i` 下最大化自己的 return。如果 rewards 完全相同，就是 **fully cooperative**。如果 zero-sum，就是 **adversarial**。如果混合，就是 **general-sum**。

**核心挑战：**

- **Non-stationarity.** 从 Agent `i` 视角看，`P(s' | s, a_i)` 依赖正在变化的 `π_{-i}`。
- **Credit assignment.** 共享 reward 下，哪个 Agent 造成了结果？
- **Exploration coordination.** Agents 要探索互补策略，而不是重复探索同一状态。
- **Scalability.** joint action space 随 `n` 指数增长。
- **Partial observability.** 每个 Agent 只能看到自己的 observation；global state 被隐藏。

**四种主流范式：**

**1. Independent Q-learning / independent PPO (IQL, IPPO).** 每个 Agent 学自己的 Q 或 policy，把其他 Agent 当作环境的一部分。简单，有时能工作（尤其 experience replay 像一种平滑的 agent-modeling trick）。理论收敛：没有。实践中：loosely-coupled tasks 可以，tightly-coupled tasks 很差。

**2. Centralized training, decentralized execution (CTDE).** 现代最常见范式。每个 Agent 有自己的 *policy* `π_i`，condition 在 local observation `o_i` 上；部署时是标准 decentralized execution。*训练* 时，centralized critic `Q(s, a_1, …, a_n)` 可以看到 full global state 与 joint action。例子：
- **MADDPG** (Lowe et al. 2017)：每个 Agent 都有 centralized critic 的 DDPG。
- **COMA** (Foerster et al. 2017)：counterfactual baseline，问“如果我换成 action `a'`，reward 会怎样？”从而隔离我的贡献。
- **MAPPO** / **IPPO** with shared critic (Yu et al. 2022)：带 centralized value function 的 PPO。2026 年 cooperative MARL 的主流。
- **QMIX** (Rashid et al. 2018)：value decomposition，`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，且使用 monotonic mixing。

**3. Self-play.** 同一个 Agent 的两个副本互相对战。对手 policy 就是自己过去某个 snapshot。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适合 zero-sum games；训练信号是对称的。

**4. League play.** self-play 面向 general-sum / adversarial environments 的扩展：保留一组过去和当前 policies，从 league 中采样对手并训练。加入 exploiters（专门击败当前最佳）与 main exploiters（专门击败 exploiters）。AlphaStar（StarCraft II）。当游戏存在“石头剪刀布”策略循环时需要它。

**Communication.** 允许 Agents 互相发送 learned messages `m_i`。在 cooperative settings 中有效。Foerster et al. (2016) 表明 differentiable inter-agent communication 可以端到端训练。今天的 LLM-based multi-agent systems（Phase 16）本质上用自然语言通信。

## Build It / 动手构建

本课使用 6×6 GridWorld，里面有两个 cooperative Agents。它们从相反角落出发，必须到达 shared goal。共享 reward：只要任一 Agent 还在移动，每步 `-1`；两个都到达时 `+10`。见 `code/main.py`。

### Step 1: the multi-agent env / 多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*Joint* action space 是 `|A|² = 16`。global state 是两个位置。

### Step 2: independent Q-learning / 独立 Q-learning

每个 Agent 运行自己的 Q-table，以 joint state 为 key。每一步：两者都选 ε-greedy action，收集 joint transition，然后各自用 shared reward 更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

这个任务能工作，因为 rewards dense 且 aligned。对 tightly-coupled tasks 会失败，例如一个 Agent 必须 *等待* 另一个 Agent 的任务。

### Step 3: centralized Q with decomposed-value update / 使用集中式 Q 做分解价值更新

使用一个 joint actions 上的 Q：`Q(s, a_1, a_2)`。从 shared reward 更新。部署时通过 marginalizing 来 decentralized：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。它用指数级 joint action space 换来 *正确* 的 global view。

### Step 4: simple self-play (adversarial 2-agent) / 简单 self-play（对抗 2-agent）

同一个 Agent，两个角色。训练 Agent A 对抗 Agent B；每 `K` 个 episodes，把 A 的 weights 拷贝到 B。对称训练，稳定进步。AlphaZero recipe 的微缩版。

## Pitfalls / 常见陷阱

- **Non-stationary replay.** 对 independent agents，experience replay 比 single-agent 更糟，因为旧 transitions 是由现在已经过时的 opponents 生成的。修复：按 recency 重新加权或 relabel。
- **Credit assignment ambiguity.** 长 episode 后只有 shared reward；不清楚哪个 Agent 贡献了什么。修复：counterfactual baselines（COMA）或 per-agent reward shaping。
- **Policy drift / chasing.** 每个 Agent 的 best response 会随着其他 Agent 的 update 改变。修复：centralized critic、较慢 learning rates，或一次冻结一方。
- **Reward hacking via coordination.** Agents 会找到 designer 没预料到的 coordinated exploits。auction agents 可能收敛到 bid zero。修复：仔细 reward design，加 behavioral constraints。
- **Exploration redundancy.** 多个 Agents 探索相同 state-action pairs。修复：per-agent entropy bonuses 或 role-conditioning。
- **League cycles.** 纯 self-play 可能陷入 dominance cycle。修复：带 diverse opponents 的 league play。
- **Sample explosion.** `n` 个 Agents × state space × joint actions。用 function approximation 近似；使用 factored action spaces（每个 Agent 一个 policy output head）。

## Use It / 应用它

2026 年 MARL 应用地图：

| Domain | Method | Notes |
|--------|--------|-------|
| Cooperative navigation / manipulation | MAPPO / QMIX | CTDE; shared critic + decentralized actors. |
| Two-player games (chess, Go, poker) | Self-play with MCTS (AlphaZero) | Zero-sum; symmetric training. |
| Complex multiplayer (Dota, StarCraft) | League play + imitation pretraining | OpenAI Five, AlphaStar. |
| Autonomous-vehicle fleets | CTDE MAPPO / PPO with attention | Partial obs; variable team sizes. |
| Auction markets | Game-theoretic equilibrium + RL | Mean-field RL when `n` → ∞. |
| LLM multi-agent systems (Phase 16) | Natural-language comm + role conditioning | RL loop at the agent-planning layer. |

2026 年，MARL 增长最快的方向是 LLM-based：多个 language-model agents 进行协商、辩论、构建软件。RL 出现在 *trajectory-level* outputs 上的 preference optimization，而不是 token-level（Phase 16 · 03）。

## Ship It / 交付它

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Exercises / 练习

1. **Easy.** 在 2-agent cooperative GridWorld 上训练 independent Q-learning。mean return > 0 需要多少个 episode？画出 joint learning curve。
2. **Medium.** 加一个 “coordination” 任务：只有两个 Agents 在同一回合踏上 goal，才算到达。independent Q 还能收敛吗？什么坏了？
3. **Hard.** 实现 MAPPO-style training 的 centralized critic，并在 coordination task 上和 independent PPO 比较收敛速度。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Markov game | “Multi-agent MDP” | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个 Agent 都有自己的 reward。 |
| CTDE | “Centralized training, decentralized execution” | 训练时 joint critic；每个 Agent 的 policy 只使用 local obs。 |
| IPPO | “Independent PPO” | 每个 Agent 单独跑 PPO。简单 baseline；常被低估。 |
| MAPPO | “Multi-agent PPO” | 带 global state centralized value function 的 PPO。 |
| QMIX | “Monotonic value decomposition” | `Q_tot = f_monotone(Q_1, …, Q_n)`，允许 decentralized argmax。 |
| COMA | “Counterfactual multi-agent” | Advantage = my Q minus expected Q marginalizing over my action。 |
| Self-play | “Agent vs past self” | 单个 Agent，两个角色；zero-sum games 的标准做法。 |
| League play | “Population training” | 缓存 past policies，从 pool 中采样 opponents；处理策略循环。 |

## Further Reading / 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带 centralized critic 的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — credit assignment 的 counterfactual baselines。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 带 monotonicity 的 value decomposition。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 中意外强。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模 league play。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — zero-sum games 中的纯 self-play。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包含教材对 multi-agent settings 及 non-stationarity 问题的简短处理，CTDE 正是为此设计。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 覆盖 cooperative、competitive、mixed MARL 及收敛结果的 survey。
