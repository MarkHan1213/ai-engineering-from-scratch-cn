# Swarm Optimization for LLMs (PSO, ACO) / 面向 LLM 的群体优化（PSO、ACO）

> 仿生优化正在 LLM 领域回归。**LMPSO**（arXiv:2504.09247）使用 PSO，其中每个 particle 的 velocity 是一个 prompt，由 LLM 生成下一个 candidate；它在结构化序列输出（数学表达式、程序）上表现好。**Model Swarms**（arXiv:2410.11163）把每个 LLM expert 当作模型权重流形上的 PSO particle，在 9 个数据集、12 个 baseline 上用仅 200 个 instances 报告 **13.3% average gain**。**SwarmPrompt**（ICAART 2025）把 PSO + Grey Wolf 混合做 prompt optimization。**AMRO-S**（arXiv:2603.12933）是 ACO-inspired pheromone specialists，用于 multi-agent LLM routing：**4.7x speedup**、可解释 routing evidence、quality-gated asynchronous update，将 inference 与 learning 解耦。本课在 prompt parameter space 上实现 PSO，在 Agent routing 上实现 ACO，并测量这些经典算法为什么适合 LLM 时代，以及什么时候不适合。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 09（Parallel Swarm Networks）, 第 16 阶段 · 14（Consensus and BFT）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 PSO 和 ACO 的经典机制，以及它们为什么适合 gradient-free LLM 优化
- 把 LMPSO 中 “velocity as prompt” 映射到结构化输出搜索
- 用 pheromone matrix 设计 ACO-style Agent routing
- 判断何时使用 PSO、ACO、genetic algorithm 或 gradient-based optimizer

## The Problem / 问题

你有一个 prompt，在任务 eval 上得分 62%。你想提升它。朴素做法是手工做 gradient-free tweaking，但扩展很差。强化学习需要 reward signals 和足够多 rollouts 才能训练。对 prompt 做 backprop 也不现实，因为 prompt 是离散字符串，不是可微参数。

经典仿生优化（PSO 面向连续搜索空间，ACO 面向路径选择）正是为这种场景设计的：gradient-free、population-based、每次 evaluation 相对便宜。把它们和 LLM 组合，让 LLM 做 gradient-free search step，就得到一个出人意料实用的优化器。

同样模式也适用于多 Agent 系统里的 Agent *routing*。ACO-style pheromone trail 记录哪类 task 上哪个 Agent 表现最好，让 router 利用轨迹，同时 pheromone 衰减以便重新发现路线。

## The Concept / 概念

### PSO refresher (Kennedy & Eberhart 1995) / PSO 速记

Particle Swarm Optimization：连续搜索空间中的粒子群。每个 particle 有 position `x_i` 和 velocity `v_i`。每次迭代：

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

其中 `p_best` 是粒子自己的历史最佳，`g_best` 是 swarm 的全局最佳，`w, c1, c2` 是 inertia + cognitive + social weights，`r1, r2` 是随机因子。

### PSO on LLM outputs — LMPSO / LLM 输出上的 PSO

arXiv:2504.09247 把 PSO 适配到 LLM-generated structured outputs（数学表达式、程序）。每个 particle 是一个 candidate output。velocity 是一个 *prompt*，描述如何把当前输出向 personal/global best 修改。LLM 根据 velocity prompt 生成新输出。velocity 的 “inertia” 可以是 “make small incremental changes” 这样的 prompt。

适用条件：

- 输出结构化（可 parse、可 evaluate）。
- fitness 自动化（test runs、arithmetic evaluation）。
- population 小（约 10-30 particles），总 LLM calls 可控。

不适合需要 human review 的 fitness，因为每轮成本会过高。

### Model Swarms / 模型群体

arXiv:2410.11163 把 PSO 从输出层挪到 *model* 层。每个 “particle” 是一个 expert LLM（参数）。swarm 通过 gradient-free update 把参数朝 collective best 移动。报告结果：在 9 个数据集、12 个 baseline 上平均提升 13.3%，每轮只用 200 个 instances。

关键洞察是：LLM expert models 已经位于共享 parameter manifold 的邻近位置（adapter weights、LoRA deltas）。在这个低维子空间上做 PSO 便宜且有效。

### ACO refresher (Dorigo 1992) / ACO 速记

Ant Colony Optimization：蚂蚁穿越图，路径上有 pheromone trail。蚂蚁移动概率按 pheromone strength 加权。完成任务的蚂蚁按 solution quality 沉积 pheromone。pheromone 随时间衰减。

### AMRO-S — ACO for agent routing / AMRO-S：面向 Agent routing 的 ACO

arXiv:2603.12933 使用 ACO 做 multi-agent routing。每种 task-type 是 “destination”；每个 Agent 是可能 route。产生好输出的 route pheromone 会增强。关键贡献：

- **Interpretable routing evidence.** pheromone strength 是人类可读信号。
- **Quality-gated asynchronous update.** 只有 quality check 通过后才更新 pheromone，把 inference 与 learning 解耦。
- **4.7x speedup** on multi-agent routing benchmark。

quality gate 很关键：没有 gate，fast-but-wrong Agent 会积累 pheromone，系统会锁定坏路线。

### When to use PSO / ACO for LLMs / 何时使用

**Use PSO when:**

- 搜索空间连续或可映射到连续参数（prompt embeddings、LoRA weights、numeric generation parameters）。
- fitness 便宜且自动化。
- population 可以小（10-30）。

**Use ACO when:**

- 你有 routing 或 path-selection 问题。
- 决策会随时间强化（同类 task 重复出现）。
- 你需要 routing decision 的可解释 evidence。

**Do not use either when:**

- fitness 需要 human review（每轮太贵）。
- 搜索空间是 PSO 不擅长的离散组合空间（改用 genetic algorithms）。
- 实时决策需要严格 latency（PSO/ACO 相对 single-pass heuristic 收敛慢）。

### Why bio-inspired still wins / 仿生方法为什么仍然有效

基于梯度的方法需要可微的信号。LLM 输出和 routing decisions 不容易可微。伪梯度方法（reinforcement-learned routers、DPO-style prompt tuners）可以工作，但需要昂贵训练。

PSO 和 ACO 只需要一个 *evaluator* function。如果你能给 candidate output 或 routing decision 打分，就能优化。这让适用门槛低很多。

### Practical limits / 实用限制

- **Population budget.** N particles × T iterations × per-eval cost。如果 LLM eval 约 $0.02/call，20-particle PSO 跑 50 iterations 约 $20。提前规划。
- **Exploration vs exploitation.** pheromone decay rate 和 PSO inertia 都是取舍；decay 太快会忘记解，太慢会卡在早期 local optimum。
- **Catastrophic drift.** fitness landscape 变化（新数据分布）时，两种算法都可能先收敛再发散。监控 best-fitness stability。

## Build It / 动手构建

`code/main.py` 实现：

- `LMPSO` — 在数值 prompt parameters（temperature、top_k weights）上做 PSO。每个 particle 的 “LLM generation” 用脚本化 fitness function 模拟。运行 30 iterations，展示 g_best 收敛。
- `AMRO_S` — ACO-style routing。3 个 Agent、4 种 task types、pheromone matrix、100 个 routed tasks。打印 (task_type → agent choices) 随时间的分布，展示 trail formation。
- Comparison：同一 task stream 上 random routing vs ACO routing。测量 quality 和 latency。

运行：

```
python3 code/main.py
```

预期输出：

- LMPSO：g_best fitness 从随机提升到接近最优，约 30 iterations。
- AMRO-S：pheromone table 稳定到每类 task 的正确 Agent；ACO routing 在 quality 上比 random 高约 30-40%，同时减少 latency（更少 retries）。

## Use It / 应用它

`outputs/skill-swarm-optimizer.md` 帮助你在 LLM / Agent 优化问题中选择 PSO、ACO、genetic algorithms 或 gradient-based optimizers。

## Ship It / 交付它

- **Start small.** 10-20 particles，20-50 iterations。只有 convergence curve 显示明确收益时才扩大。
- **Log pheromones or g_best per iteration.** 没有 trail 的 swarm optimizer 很难调试。
- **Quality-gate updates.** 尤其是 ACO routing：fast-and-wrong Agent 不得积累 pheromone。
- **Reset decay on distribution shift.** eval distribution 变化时，旧 pheromone 过期；reset 或临时加倍 decay rate。
- **Cap the per-iteration cost.** 输出 cost-per-iteration metric。每轮 $500、提升 0.5% 的 PSO 不可交付。

## Exercises / 练习

1. 运行 `code/main.py`。观察 LMPSO 收敛。改变 population size：5、10、20、50。time-to-converge 在什么 size 后饱和？
2. 实现 “catastrophic drift” 实验：第 30 iteration 后改变 fitness function。PSO 多快适应？reset `p_best` 有帮助吗？
3. 给 AMRO-S 增加 quality gate：eval score > 0.7 时才 deposit pheromone。相比无 gate，收敛如何变化？
4. 阅读 LMPSO（arXiv:2504.09247）。把论文中的 “velocity as a prompt” 映射回你的 numeric velocity。simulation 丢失了什么，保留了什么？
5. 阅读 AMRO-S（arXiv:2603.12933）。实现 decoupled “inference fast-path” 与 asynchronous pheromone update。在持续负载下系统 latency 如何变化？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| PSO | “Particle Swarm Optimization” | Kennedy-Eberhart 1995。population-based gradient-free optimizer。 |
| ACO | “Ant Colony Optimization” | Dorigo 1992。通过 pheromone trails 做 path/route optimization。 |
| LMPSO | “PSO with LLM generation” | arXiv:2504.09247。velocity 是 prompt；LLM 产出 candidates。 |
| Model Swarms | “PSO on expert weights” | arXiv:2410.11163。在 model parameter subspace 上做 gradient-free update。 |
| AMRO-S | “ACO for agent routing” | arXiv:2603.12933。task-type × agent 上的 pheromone matrix。 |
| p_best / g_best | “Personal / global best” | 单 particle 和整个 swarm 已找到的最佳解。 |
| Pheromone | “Routing memory” | edge 上的强度；随时间衰减；按 quality 沉积。 |
| Quality-gated update | “只从好 run 学习” | pheromone deposit 以 quality check 为条件。 |
| Catastrophic drift | “Distribution shift” | fitness landscape 改变；旧 p_best 和 pheromone 过期。 |

## Further Reading / 延伸阅读

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — 1995 PSO paper
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — 1992 ACO foundations
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — 面向结构化 LLM 输出的 PSO
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — model-weight subspace 上的 PSO
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — 带 quality gate 的 pheromone-driven routing
