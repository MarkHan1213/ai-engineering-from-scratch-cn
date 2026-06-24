# Deep Q-Networks (DQN) / 深度 Q 网络

> 2013 年，Mnih 在原始像素上训练一个 Q-learning 网络，在七个 Atari 游戏上击败所有经典 RL Agent。2015 年扩展到 49 个游戏并发表于 Nature，点燃了 deep-RL 时代。DQN 就是 Q-learning 加上三个让 function approximation 稳定下来的技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 第 03 阶段 · 03（Backpropagation）, 第 09 阶段 · 04（Q-learning, SARSA）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 说明 tabular Q-learning 为什么无法处理高维状态。
- 实现 replay buffer、manual MLP Q-network 和 DQN update。
- 解释 experience replay、target network 与 reward clipping 的稳定作用。
- 理解 Double DQN 如何修正 maximization bias。
- 判断哪些任务适合 DQN，哪些应转向 PPO、SAC、TD3 或 offline RL。

## The Problem / 问题

Tabular Q-learning 需要给每个 (state, action) pair 单独存一个 Q-value。国际象棋棋盘大约有 `10⁴³` 个状态。Atari frame 是 210×160×3 = 100,800 个特征。Tabular RL 在几千个状态时就撑不住，更别说数十亿状态。

事后看，修复方式很明显：用神经网络 `Q(s, a; θ)` 替换 Q-table。但这个“明显”的想法花了几十年才真正稳定。朴素地把 function approximation 接到 Q-learning 上，会因为 “deadly triad” 发散：function approximation + bootstrapping + off-policy learning。Mnih et al. (2013, 2015) 找到了三个让学习稳定的工程技巧：

1. **Experience replay** 去相关化 transitions。
2. **Target network** 冻结 bootstrap target。
3. **Reward clipping** 归一化梯度量级。

Atari DQN 第一次证明：同一个 architecture、同一套 hyperparameters，可以从原始像素解决几十个控制问题。后面所有 “deep-RL” 方法——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都叠在这个三技巧基础上。

## The Concept / 概念

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**目标函数。** DQN 在 neural Q-function 上最小化 one-step TD loss：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` 是 online network，每一步用 gradient descent 更新。`θ^-` 是 target network，周期性从 `θ` 拷贝（约每 10,000 步）。`D` 是存储过去 transitions 的 replay buffer。

**三个技巧，按重要性排序：**

**Experience replay.** 一个容量约 `~10⁶` transitions 的 ring buffer。每个 training step 从中均匀随机采样 minibatch。它打破 temporal correlation（连续帧几乎相同），让网络多次学习稀有的 rewarding transitions，并去相关化连续 gradient updates。没有它，带神经网络的 on-policy TD 在 Atari 上会发散。

**Target network.** 在 Bellman equation 两边都用同一个网络 `Q(·; θ)` 会让 target 每次 update 都移动，等于“追着自己的尾巴跑”。修复方式：保留第二个网络 `Q(·; θ^-)`，权重冻结。每隔 `C` 步，拷贝 `θ → θ^-`。这样 regression target 在数千个 gradient steps 内保持稳定。Soft updates `θ^- ← τ θ + (1-τ) θ^-`（DDPG、SAC 使用）是更平滑的变体。

**Reward clipping.** Atari 的奖励量级从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 可以避免某个游戏独占梯度。当 reward magnitude 本身有意义时这么做是错的；但在 Atari 中只看符号通常足够。

**Double DQN.** Hasselt (2016) 修复 maximization bias：用 online net *选择* action，用 target net *评估* 它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

这是 drop-in replacement，效果稳定更好。默认使用。

**Other improvements (Rainbow, 2017):** prioritized replay（更多采样 high-TD-error transitions）、dueling architecture（分离 `V(s)` 与 advantage heads）、noisy networks（学习式探索）、n-step returns、distributional Q（C51/QR-DQN）、multi-step bootstrapping。每项带来几个百分点，收益大致可叠加。

## Build It / 动手构建

这里的代码是 stdlib-only、numpy-free：我们在一个很小的 continuous GridWorld 上手写单隐藏层 MLP，所以每个 training step 都只要微秒级。算法形状和大规模 Atari DQN 完全一致。

### Step 1: replay buffer / 回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 大约使用 50,000 容量；我们的 toy env 用 5,000 就够。

### Step 2: a tiny Q-network (manual MLP) / 一个极小的 Q-network（手写 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Forward pass：linear → ReLU → linear。这就是整个网络。

### Step 3: the DQN update / DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

形状和 Lesson 04 的 Q-learning 一样，只多两点：(a) 我们对可微 `Q(·; θ)` 做 backprop，而不是索引 table；(b) target 使用 `Q(·; θ^-)`。

### Step 4: the outer loop / 外层循环

每个 episode：用 `Q(·; θ)` 做 ε-greedy 行动，把 transitions 放进 buffer，采样 minibatch，做一次 gradient step，并周期性同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的 16-dim one-hot state tiny GridWorld 上，Agent 约 500 个 episode 就能学到近似最优 policy。换到 Atari，就把它扩展到 200M frames，并加入 CNN feature extractor。

## Pitfalls / 常见陷阱

- **Deadly triad.** Function approximation + off-policy + bootstrapping 可能发散。DQN 用 target net + replay 缓解；不要移除任何一个。
- **Exploration.** ε 必须衰减，通常从 1.0 在训练前 `~10%` 的步骤内衰减到 0.01。早期探索不足会让 Q-net 收敛到局部 basin。
- **Overestimation.** 对 noisy Q 取 `max` 有向上偏差。生产环境默认使用 Double DQN。
- **Reward scale.** 裁剪或归一化 rewards；梯度量级与 reward magnitude 成正比。
- **Replay buffer coldstart.** buffer 里有几千条 transitions 前不要训练。早期只在 `~20` 个样本上做梯度会过拟合。
- **Target sync frequency.** 太频繁 ≈ 没有 target net；太不频繁 ≈ target 过旧。Atari DQN 使用 10,000 env steps。经验法则：大约每训练 horizon 的 `~1/100` 同步一次。
- **Observation preprocessing.** Atari DQN 堆叠 4 帧，让 state 满足 Markov。任何缺少 velocity 信息的环境都需要 frame-stacking 或 recurrent state。

## Use It / 应用它

2026 年，DQN 很少是 state-of-the-art，但仍是标准 off-policy 参考算法：

| Task | Method of choice | Why not DQN? |
|------|------------------|--------------|
| Discrete-action Atari-like | Rainbow DQN or Muesli | Same framework, more tricks. |
| Continuous control | SAC / TD3 (Phase 9 · 07) | DQN has no policy network. |
| On-policy / high-throughput | PPO (Phase 9 · 08) | No replay buffer; easier to scale. |
| Offline RL | CQL / IQL / Decision Transformer | Conservative Q targets, no bootstrapping blowups. |
| Large discrete action spaces (recommender) | DQN with action embedding, or IMPALA | Fine; decoration matters. |
| LLM RL | PPO / GRPO | Sequence-level, not step-level; different loss. |

这些课程内容仍然能迁移。Replay 和 target networks 出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的 self-play buffer，以及每个 offline RL 方法里。Reward clipping 以 PPO 中 advantage normalization 的形式继续存在。这个 architecture 是蓝图。

## Ship It / 交付它

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Exercises / 练习

1. **Easy.** 运行 `code/main.py`。画出 per-episode return 曲线。running mean 超过 -10 需要多少个 episode？
2. **Medium.** 禁用 target network（在 Bellman target 两边都用 online net）。测量训练不稳定性：return 会震荡还是发散？
3. **Hard.** 加入 Double DQN：online net 选择 `argmax a'`，target net 负责评估。在 noisy-reward GridWorld 上，训练 1,000 个 episode 后，比较有无 Double DQN 时 `Q(s_0, best_a)` 相对 true `V*(s_0)` 的 bias。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| DQN | “Deep Q-learning” | 带 neural Q-function、replay buffer 和 target network 的 Q-learning。 |
| Experience replay | “Shuffled transitions” | 每个 gradient step 从 ring buffer 均匀采样；去相关化数据。 |
| Target network | “Frozen bootstrap” | 用在 Bellman target 中的周期性 Q copy；稳定训练。 |
| Deadly triad | “Why RL diverges” | Function approximation + bootstrapping + off-policy = 没有收敛保证。 |
| Double DQN | “Fix for maximization bias” | Online net 选 action，target net 评估它。 |
| Dueling DQN | “V and A heads” | 分解 Q = V + A - mean(A)；输出相同，梯度流更好。 |
| Rainbow | “All the tricks” | DDQN + PER + dueling + n-step + noisy + distributional 的组合。 |
| PER | “Prioritized Replay” | 按 TD-error magnitude 成比例采样 transitions。 |

## Further Reading / 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 引爆 deep RL 的 2013 NeurIPS workshop paper。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 论文，49-game DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — stacked-tricks 论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代讲解。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 教材中对 “deadly triad”（function approximation + bootstrapping + off-policy）的处理，DQN 的 target network 和 replay buffer 正是为压住它而设计的。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 用于 ablation studies 的单文件 DQN 参考实现；适合和本课 from-scratch 版本一起读。
