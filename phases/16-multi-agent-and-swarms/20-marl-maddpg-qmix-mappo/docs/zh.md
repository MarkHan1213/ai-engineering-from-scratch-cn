# MARL — MADDPG, QMIX, MAPPO / 多智能体强化学习：MADDPG、QMIX、MAPPO

> 多 Agent coordination 的 reinforcement-learning 传承，到 2026 年仍在影响 LLM-agent 系统。**MADDPG**（Lowe et al., NeurIPS 2017, arXiv:1706.02275）引入 Centralized Training, Decentralized Execution（CTDE）：每个 critic 在训练时看到所有 Agent 的 states 和 actions；测试时只有本地 actor 运行。适用于合作、竞争和混合场景。**QMIX**（Rashid et al., ICML 2018, arXiv:1803.11485）是 value-decomposition，使用 monotonic mixing network；per-agent Q 组合成 joint Q，让 `argmax` 干净地分布式执行，曾在 StarCraft Multi-Agent Challenge（SMAC）上占优。**MAPPO**（Yu et al., NeurIPS 2022, arXiv:2103.01955）是带 centralized value function 的 PPO；在 particle-world、SMAC、Google Research Football、Hanabi 上以最少 tuning 表现 “surprisingly effective”。这些方法支撑必须分散执行的 Agent team policy training。MAPPO 是 **2026 cooperative-MARL 默认基线**。本课从一个小 grid-world toy 构建三者，把三个想法变成肌肉记忆，再接触 LLM-agent training。

**类型：** 学习
**语言：** Python（stdlib, small NumPy-free implementations）
**前置知识：** 第 09 阶段（Reinforcement Learning）, 第 16 阶段 · 09（Parallel Swarm Networks）
**时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 解释 MARL 中 non-stationarity 问题，以及 CTDE 为什么重要
- 区分 MADDPG 的 centralized critic、QMIX 的 value decomposition、MAPPO 的 centralized value function
- 判断 cooperative、competitive、homogeneous、heterogeneous、多 action-space 场景该用哪类算法
- 将 CTDE 当作 LLM-agent 系统的架构设计纪律，而不仅是训练算法

## The Problem / 问题

LLM-agent 系统越来越需要训练 inter-agent coordination policy：什么时候 defer，什么时候行动，调用哪个 peer。告诉你如何训练这类 policy 的文献是 Multi-Agent Reinforcement Learning（MARL），它早于 LLM 浪潮，并已有少数主导算法。

没有模式词汇直接读 MARL 论文会很痛苦。Centralized training with decentralized execution（CTDE）、value decomposition、centralized critics 不是 buzzwords，而是对具体问题的具体回答：

- Independent RL（每个 Agent 单独学习）在每个 Agent 看来环境非平稳。很糟。
- Centralized RL（一个 Agent 控制所有）不扩展，也违反执行约束。
- CTDE 两者兼得：训练时使用全局信息，部署时使用本地 policy。

## The Concept / 概念

### Three environments the papers use / 三类常见环境

- **Particle World (multi-agent particle env).** 简单 2D 物理，含合作/竞争任务。MADDPG 原始 testbed。
- **StarCraft Multi-Agent Challenge (SMAC).** 合作微操、partial observation。QMIX testbed。离散动作、连续状态。
- **Google Research Football, Hanabi, MPE.** MAPPO baselines。

不同环境有不同 action/observation types，算法选择也随之变化。

### MADDPG (2017) — the CTDE pattern / MADDPG：CTDE 模式

每个 Agent `i` 有一个 actor `mu_i(o_i)`，把自己的 observation 映射为 action。每个 Agent 也有一个 critic `Q_i(x, a_1, ..., a_n)`，训练时能看见所有 observations 和所有 actions。actor 根据 critic 评价做 policy gradient update。

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

为什么 CTDE：训练时我们知道所有人的 actions；用它降低每个 critic 的方差。部署时，每个 Agent 只看 `o_i` 并调用 `mu_i(o_i)`。

失败模式：critic 随 Agent 数 N 增长（输入包含所有 actions）。没有近似时，超过约 10 个 Agent 不易扩展。

### QMIX (2018) — value decomposition / QMIX：价值分解

只面向 cooperative。global reward 是 per-agent Q-values 的 monotone function：

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

monotonicity 保证 `argmax_a Q_tot` 可以通过每个 Agent 独立选择 `argmax_{a_i} Q_i` 计算出来。这正是你需要的 **decentralized execution property**。训练时，mixing network 从 per-agent Qs 生成 `Q_tot`。

为什么 QMIX 在 SMAC 上赢：合作 StarCraft 微操有同质 Agent、local obs、global reward，完美适配 value decomposition。

失败模式：monotonicity constraint 很强；有些任务的 reward 结构不能 monotone decomposable（例如一个 Agent 为团队牺牲）。扩展（QTRAN、QPLEX）放松了这一点。

### MAPPO (2022) — the overlooked default / MAPPO：被低估的默认基线

Multi-Agent PPO：带 centralized value function 的 PPO。每个 Agent 有自己的 policy；所有 Agent 共享（或拥有 per-agent）能看全局 state 的 value function。Yu et al. 2022 在五个 benchmark 上把 MAPPO 与 MADDPG、QMIX 及其扩展比较，发现：

- MAPPO 在 particle-world、SMAC、Google Research Football、Hanabi、MPE 上匹配或超过 off-policy MARL methods。
- 需要的 hyperparameter tuning 很少。
- 训练稳定，跨 seeds 可复现。

社区直到这篇论文才重新重视 on-policy MARL。2026 年，MAPPO 是 cooperative MARL 默认 baseline；任何新方法都要先超过它。

### Why LLM-agent engineers should care / LLM-Agent 工程师为什么要关心

三个直接用途：

1. **Router training.** meta-agent 选择哪个 sub-agent 处理任务。这是一个 MARL 问题：N 个 decentralized sub-agents 和一个 centralized router。MAPPO 适配。
2. **Role emergence.** 在 generative-agent simulations 中，训练 Agent 随时间采用互补角色，本质上是 MARL。QMIX-style value decomposition 会把互补性结构化。
3. **Multi-agent tool use.** 当 Agent 共享工具并竞争 budget，用 CTDE 训练能得到尊重资源约束的可部署 local policies。

实用提醒：2026 年多数生产 LLM-agent 系统仍用 prompt 而不是训练 policy。只有当你有 (a) 大量 interaction data，(b) 清晰 reward signal，(c) 愿意投资训练基础设施，MARL 才值得进入。

### CTDE as a design pattern beyond RL / CTDE 作为架构模式

即使不训练，CTDE 也是有用架构模式：

- 在 *design* 阶段，假设可以看到整个团队。
- 在 *runtime* 阶段，强制 decentralized execution：每个 Agent 只看 `o_i`。

这个模式迫使你显式管理 per-agent state，并提前思考 partial observability。很多生产多 Agent 系统默认 everywhere shared state，CTDE discipline 能防止这一点。

### The non-stationarity problem / 非平稳问题

当多个 Agent 同时学习时，每个 Agent 的环境（包含其他 Agent policy）是 non-stationary。经典单 Agent RL 证明失效。本课算法都在处理它：

- MADDPG：global critic 看见所有 actions，让 value estimate 更平稳。
- QMIX：value decomposition 把 learning 移到 joint-Q space，最优性定义清楚。
- MAPPO：centralized value function 降低其他 Agent policy 变化带来的方差。

在 LLM-agent 系统里，non-stationarity 表现为“我的 Agent 上个月还工作，现在上游另一个 Agent 改了，我的就出错”。带 CTDE 的 MARL training 是原则性修复；prompt-level 修复更快但不够耐久。

### What this lesson does NOT cover / 本课不覆盖什么

训练真实神经网络是 Phase 09 主题。本课构建 scripted-policy 版本，演示 CTDE、value-decomposition 和 centralized-value patterns，不做梯度更新。目标是在使用完整 MARL library（PyMARL、MARLlib、RLlib multi-agent）前先内化模式。

## Build It / 动手构建

`code/main.py` 在一个微型 2-Agent cooperative grid-world 上实现三个 pattern demo：

- Environment：2 个 Agent 位于 4x4 grid，1 个 reward pellet。任一 Agent 抵达 pellet，reward = 1，任务结束。
- `IndependentAgents` — 每个 Agent 把其他 Agent 当环境。baseline。
- `MADDPGStyle` — centralized critic 计算 joint value；actor policies 从中更新。scripted policy improvement。
- `QMIXStyle` — 带 monotone mixer 的 value decomposition。
- `MAPPOStyle` — centralized value function；policies 相对 shared baseline 更新。

四者都跑相同 episodes，并报告 average steps-to-goal。CTDE variants 比 independent baseline 收敛到更短路径。

运行：

```
python3 code/main.py
```

预期输出：independent agents 平均约 6 步；CTDE variants 向约 3.5 步收敛（4x4 grid 最优为 3）。即使是 scripted policies，也能看出 pattern 差异。

## Use It / 应用它

`outputs/skill-marl-picker.md` 为给定多 Agent 任务选择 MARL 算法：cooperative vs competitive、homogeneous vs heterogeneous、action-space type、scale、reward signal。

## Ship It / 交付它

MARL 在生产中少见。使用时：

- **Start with MAPPO.** 2022 论文将其确立为 baseline；先复现它能少走很多弯路。
- **Log every agent's observation and action stream.** 没有 per-agent traces，MARL 调试几乎无望。
- **Separate training code from execution code.** CTDE 是纪律；让 execution path 真的只能看 `o_i`。
- **Reward shaping warning.** MARL 对 reward design 极其敏感。shaping 里一个 coordination bug，Agent 就会学会利用它。跑 adversarial tests。
- **For LLM agents**, 先考虑 prompt-level policies。只有 interaction data + reward signal + infrastructure 都具备时，再投资 MARL training。

## Exercises / 练习

1. 运行 `code/main.py`。测量 independent 和 MAPPO-style agents 的 steps-to-goal 差距。换成 6x6 grid 时差距变大还是变小？
2. 实现 competitive variant：两个 Agent，一个 pellet，先到者得 reward。哪种 pattern 最能处理 competition？历史上是 MADDPG。
3. 阅读 MADDPG（arXiv:1706.02275）Section 3。用自己的话把 exact critic update rule 写成 pseudocode。
4. 阅读 MAPPO（arXiv:2103.01955）。作者为什么认为 centralized value + PPO 在 benchmark 上胜过 off-policy MARL？列出三条最强 claim。
5. 把 CTDE 当作设计模式应用到一个假设 LLM-agent 系统（例如 research agent + summarizer + coder）。设计时可用而运行时不可用的 joint information 是什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MARL | “Multi-Agent RL” | 面向多 Agent 系统的强化学习。 |
| CTDE | “Centralized Training, Decentralized Execution” | 训练用全局信息，部署用本地 policy。 |
| MADDPG | “Multi-Agent DDPG” | CTDE，per-agent critic 看所有 observations + actions。 |
| QMIX | “Value decomposition” | per-agent Q 的 monotonic mixing。合作场景。 |
| MAPPO | “Multi-Agent PPO” | 带 centralized value function 的 PPO。2026 默认 baseline。 |
| Value decomposition | “个体 Q 之和” | joint Q 表示为 per-agent Qs 的 monotone function。 |
| Non-stationarity | “移动目标” | 其他 Agent 学习时，每个 Agent 的环境也在变化。MARL 核心问题。 |
| On-policy / off-policy | “从当前 / replay 学习” | PPO 是 on-policy（MAPPO）；DDPG 和 Q-learning 是 off-policy。 |
| SMAC | “StarCraft Multi-Agent Challenge” | 合作微操 benchmark；QMIX 的经典战场。 |

## Further Reading / 延伸阅读

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG；NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX；ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO；NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — MAPPO 结果的可读解释
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge
