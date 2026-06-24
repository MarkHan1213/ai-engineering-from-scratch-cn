# Sim-to-Real Transfer / 仿真到现实迁移

> 在 simulator 中训练、到硬件上失败的 policy，本质上只是记住了 simulator。Domain randomization、domain adaptation 和 system identification，是让 learned controllers 跨过 reality gap 的三件工具。

**类型：** 学习
**语言：** Python
**前置知识：** 第 09 阶段 · 08（PPO）, 第 02 阶段 · 10（Bias/Variance）
**时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 reality gap 为什么是机器人 RL 部署的核心问题。
- 区分 domain randomization、system identification 与 domain adaptation。
- 用带 slip 参数的 toy GridWorld 演示 DR 训练与 zero-shot evaluation。
- 识别 randomization 范围过宽、过窄或参数选错的风险。
- 设计包含 safety envelope 的 sim-to-real pipeline。

## The Problem / 问题

在真实机器人上训练又慢、又危险、又昂贵。双足机器人要学会走路可能需要数百万个 training episodes；真实双足机器人摔一次就可能损坏硬件。Simulation 给你无限 reset、确定性复现、parallel environments，并且不会造成物理损伤。

但 simulator 是错的。轴承摩擦比 MuJoCo model 更大。摄像头有 lens distortion，而 simulator 里没有。电机有延迟、backlash 和 saturation，99% 的 sim model 会跳过这些。风、灰尘、可变光照会毁掉在干净渲染中训练出的 policy。**Reality gap**——sim distribution 和 real distribution 的系统性差异——是 robotics 中 deployed RL 的核心问题。

你需要一个对 *sim-to-real distribution shift* 鲁棒的 policy。三种历史路径：随机化 simulator（domain randomization），用少量真实数据适配 policy（domain adaptation / fine-tuning），或者识别真实系统参数并匹配它们（system identification）。2026 年主流 recipe 会把三者与大规模 parallel simulation（Isaac Sim、Isaac Lab、Mujoco MJX on GPU）结合起来。

## The Concept / 概念

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Domain Randomization (DR).** Tobin et al. 2017, Peng et al. 2018。训练时随机化所有可能与真实机器人不同的 sim 参数：mass、friction coefficients、motor PD gains、sensor noise、camera position、lighting、textures、contact models。policy 学会条件化地判断“今天在什么样的 sim 里”，并在整个范围内泛化。如果真实机器人落在 training envelope 内，policy 就能工作。

- **Upside:** 不需要真实数据。一套 recipe，多个机器人。
- **Downside:** randomization 过度会产生“通用但过于谨慎”的 policy。噪声太多 ≈ regularization 太强。

**System Identification (SI).** 训练前把 simulator 参数拟合到真实世界数据。如果你能测量真实机器人 arm-joint friction，就把它塞进 sim。然后训练一个期待这些值的 policy。需要访问真实系统，但直接缩小 reality gap。

- **Upside:** 精确、低噪声的 training target。
- **Downside:** policy 看不到 residual model error；未识别的小效应（例如 motor deadband）仍然会破坏部署。

**Domain Adaptation.** 先在 sim 中训练，再用少量真实数据 fine-tune。两种形式：

- **Real2Sim2Real:** 用真实 rollouts 学一个 residual simulator `f(s, a, z) - f_sim(s, a)`，在 corrected sim 中训练。用很少真实数据缩小 gap。
- **Observation adaptation:** 训练一个 policy，通过 learned feature extractor 把 real obs → sim-like obs（例如 GAN pixel-to-pixel）。controller 仍然留在 sim 域。

**Privileged learning / teacher-student.** Miki et al. 2022（ANYmal quadruped）。在 simulation 中训练一个 *teacher*，它能访问 privileged information（ground truth friction、terrain height、IMU drift）。再蒸馏一个只看 real-sensor observations 的 *student*。student 从 history 中学会推断 privileged features，从而对 physical parameters 鲁棒。

**Massively parallel simulation.** 2024–2026。Isaac Lab、Mujoco MJX、Brax 都能在单张 GPU 上跑几千个 parallel robots。PPO 用 4,096 个 parallel humanoids，几小时内收集数年的经验。当训练分布变宽时，“reality gap” 会缩小；每个 env 都有不同随机参数时，DR 几乎是免费的。

**2026 年真实 recipe（四足行走示例）：**

1. 使用 massively parallel sim，并对 gravity、friction、motor gains、payload 做 domain randomization。
2. 训练可访问 privileged info（terrain map、body velocity ground truth）的 teacher policy。
3. 用只有 proprioception（leg joint encoders）的输入，把 teacher 蒸馏成 student policy。
4. 可选：用真实 IMU 上的 autoencoder 做 observation adaptation。
5. 部署。Zero-shot 到 10+ 个环境。如果失败，用带 safety constraints 的 PPO 做几分钟真实 fine-tuning。

## Build It / 动手构建

本课代码是在带 *noisy* transitions 的 GridWorld 上演示 domain randomization。我们训练一个在 “sim” 中经历随机 slip probabilities 的 policy，然后在一个训练时从没见过的 slip level 上评估 “real”。这个形状可以直接映射到 MuJoCo-to-hardware transfer。

### Step 1: parameterized sim / 参数化 simulator

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是 simulator 暴露的一个参数。在真实机器人中，它可能是 friction、mass、motor gain，或任何 sim 与 real 之间会漂移的因素。

### Step 2: train with DR / 使用 DR 训练

每个 episode 开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q-learning / 任意算法。重复很多 episodes。

### Step 3: evaluate zero-shot on "real" slips / 在 “real” slip 上 zero-shot 评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在 training support 内；`0.5` 和 `0.7` 在外。DR-trained policy 应该在 support 内接近最优，在 support 外优雅退化。fixed-slip-trained policy 在超出 training slip 后会非常脆弱。

### Step 4: compare to narrow training / 与窄分布训练比较

再训练一个只使用 `slip = 0.0` 的 policy。在同样的 `slip` sweep 上评估。你应该看到 real slip > 0 后性能灾难性下降。

## Pitfalls / 常见陷阱

- **Too much randomization.** 在 `slip ∈ [0, 0.9]` 上训练，policy 会过于风险规避，连 optimal path 都不敢走。要匹配 *expected* real-world distribution，不是“任何事都可能发生”。
- **Too little randomization.** 只在很薄的一片分布上训练，policy 完全不能泛化。使用 adaptive curriculum（Automatic Domain Randomization），随着 policy 变强逐步拓宽分布。
- **Misidentified parameter space.** 随机化错东西（真实 gap 是 motor delay，却随机化 camera hue），DR 不会有帮助。先 profile 真实机器人。
- **Privileged info leakage.** 如果 teacher 用 global state 而不只是 observations 来行动，student 可能永远追不上。确保 teacher 的 policy 在 student 给定 observation history 时是可实现的。
- **Sim-to-sim transfer failure.** 如果 policy 对更难的 sim variant 都不鲁棒，它也不会对真实世界鲁棒。部署前总是在 held-out sim variant 上测试。
- **No real-world safety envelope.** 一个在 sim 有效、在 real “也有效”的 policy，如果没有低层 safety shield，仍然能损坏硬件。加入 rate limits、torque limits、joint limits，并放在非学习 controller 中。

## Use It / 应用它

2026 年 sim-to-real stack：

| Domain | Stack |
|--------|-------|
| Legged locomotion (ANYmal, Spot, humanoid) | Isaac Lab + DR + privileged teacher / student |
| Manipulation (dexterous hands, pick-and-place) | Isaac Lab + DR + DR-GAN for vision |
| Autonomous driving | CARLA / NVIDIA DRIVE Sim + DR + real fine-tune |
| Drone racing | RotorS / Flightmare + DR + online adaptation |
| Finger/in-hand manipulation | OpenAI Dactyl (DR at unprecedented scale) |
| Industrial arms | MuJoCo-Warp + SI + small real fine-tune |

所有尺度的 control 工作流都一致：尽力拟合 simulator，拟合不了的就随机化，训练大 policy，蒸馏，带 safety shield 部署。

## Ship It / 交付它

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Exercises / 练习

1. **Easy.** 在 fixed-slip GridWorld（slip=0.0）上训练 Q-learning agent。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。画 return vs slip。
2. **Medium.** 训练一个 DR Q-learning agent，采样 `slip ~ Uniform[0, 0.3]`。在同样 sweep 上评估。DR 在 slip=0.5（out-of-distribution）时带来多少收益？
3. **Hard.** 实现 curriculum：从 slip=0.0 开始，每当 policy 达到 90% optimal 时拓宽 DR range。比较它与 fixed DR baseline 达到 slip=0.3 zero-shot 所需的总 environment steps。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Reality gap | “Sim-to-real difference” | 训练和部署之间的 physics/sensing distribution shift。 |
| Domain randomization (DR) | “Train across random sims” | 训练时随机化 sim parameters，让 policy 泛化。 |
| System identification (SI) | “Measure real and fit sim” | 估计真实物理参数，并设置 sim 与之匹配。 |
| Domain adaptation | “Fine-tune on real data” | sim 训练后用少量真实数据 fine-tune；可能适配 observation 或 dynamics。 |
| Privileged info | “Ground truth for teacher” | 只有 sim 拥有的信息；student 必须从 obs history 中推断。 |
| Teacher/student | “Distill privileged -> observable” | teacher 使用捷径训练；student 学会在没有捷径时模仿。 |
| ADR | “Automatic Domain Randomization” | 随着 policy 变强而拓宽 DR ranges 的 curriculum。 |
| Real2Sim | “Close the gap with real data” | 学一个 residual，让 sim 模仿真实 rollouts。 |

## Further Reading / 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 原始 DR 论文（机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — dynamics DR，四足 locomotion。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，大规模 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 的 teacher-student。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 推动 2025–2026 部署的大规模 parallel sim。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR curriculum method。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna framing（用 model 做 planning + rollouts），现代 sim-to-real pipeline 的基础。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — sim-to-real 方法 taxonomy 与 benchmark results。
