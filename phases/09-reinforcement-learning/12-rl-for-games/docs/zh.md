# RL for Games — AlphaZero, MuZero, and the LLM-Reasoning Era / 游戏 RL：AlphaZero、MuZero 与 LLM 推理时代

> 1992 年：TD-Gammon 用纯 TD 在西洋双陆棋上击败人类冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零掌握国际象棋、将棋与围棋。2024 年：DeepSeek-R1 证明，同一套 recipe 用 GRPO 替代 PPO 后，也能用于推理。游戏是推动本 phase 每次突破的 benchmark。

**类型：** 构建
**语言：** Python
**前置知识：** 第 09 阶段 · 05（DQN）, 第 09 阶段 · 08（PPO）, 第 09 阶段 · 09（RLHF）, 第 09 阶段 · 10（MARL）
**时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 用 self-play + search + policy improvement 统一理解 AlphaZero、MuZero 与 GRPO。
- 解释 AlphaZero 如何用 MCTS visit distribution 训练 policy-value network。
- 说明 MuZero 如何把 search 移到 learned latent dynamics 中。
- 实现 toy GRPO bandit，理解 group-relative advantage。
- 判断游戏、推理、组合优化等任务应使用 AlphaZero、MuZero、GRPO、CFR 或 league play。

## The Problem / 问题

游戏拥有 RL 想要的一切。干净 reward（win/loss）。无限 episodes（self-play resets）。完美 simulation（游戏规则本身就是 simulator）。离散或小型连续 action spaces。Multi-agent 结构迫使策略具备对抗鲁棒性。

每个重大 RL 突破也几乎都先在游戏里验证。TD-Gammon（backgammon, 1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2, 2019）。AlphaStar（StarCraft II, 2019）。MuZero（learned model, 2019）。AlphaTensor（matrix multiplication, 2022）。AlphaDev（sorting algorithms, 2023）。DeepSeek-R1（math reasoning, 2025）——最新证明 game-RL techniques 可以迁移到文本。

这个 capstone 通过一个统一视角讲三种里程碑架构：AlphaZero、MuZero、GRPO。统一视角是：**self-play + search + policy improvement**。每一代都泛化上一代；尤其 GRPO，可以看作把 AlphaZero recipe 应用到 LLM reasoning 上，只是 actions 变成 tokens，win signal 变成数学 verification。

## The Concept / 概念

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Silver et al. 给定一个规则已知的游戏（国际象棋、将棋、围棋）：

- Policy-value network：一个 tower `f_θ(s) → (p, v)`。`p` 是 legal moves 上的 prior。`v` 是 expected game outcome。
- Monte Carlo Tree Search (MCTS)：每一步展开可能 continuation 的树。用 `(p, v)` 作为 prior + bootstrap。通过 UCB（PUCT）选节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- Self-play：Agent vs Agent 玩游戏。第 `t` 步的 MCTS visit distribution `π_t` 成为 policy training target。
- Loss：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是 game outcome（+1 / 0 / -1）。

零人类知识。零 handcrafted heuristics。一套 recipe，在每个游戏上通过几千万局 self-play 掌握国际象棋、将棋和围棋。

**MuZero (2019).** Schrittwieser et al. 去掉“规则已知”这个要求。

- 不使用固定 environment，而是学习一个 *latent dynamics model* `(h, g, f)`：
  - `h(s)`：把 observation 编码成 latent state。
  - `g(s_latent, a)`：预测 next latent state + reward。
  - `f(s_latent)`：预测 policy prior + value。
- MCTS 在 *learned latent space* 中运行。同样的 search，同样的 training loop。
- 能跑 Go、chess、shogi，也能跑 Atari：一个算法，不需要 rule knowledge。

**Stochastic MuZero (2022).** 加入 stochastic dynamics 与 chance nodes；扩展到 backgammon 类游戏。

**Muesli, Gumbel MuZero (2022-2024).** 改进 sample efficiency 与 deterministic search。

**GRPO (2024-2025).** DeepSeek-R1 recipe。同样是 AlphaZero-shaped loop，只是应用到 language-model reasoning：

- “Game”：回答数学 / 编码 / 推理题。“Win” = verifier（test case passes、numerical answer matches）返回 1。
- Policy：LLM。Actions：tokens。State：prompt + response-so-far。
- 没有 critic（PPO-style V_φ）。对每个 prompt，从 policy 采样 `G` 个 completions。计算每个 completion 的 reward。使用 **group-relative advantage** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE-style update 的信号。
- KL penalty to reference policy 防止漂移（类似 RLHF）。
- 完整 loss：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

没有 reward model，没有 critic，没有 MCTS。Group-relative baseline 替代了三者。在 reasoning benchmarks 上，它以一小部分 compute 匹配或超过 PPO-RLHF 质量。

**完整 R1 recipe。** DeepSeek-R1（DeepSeek 2025）论文里其实有两个模型：

- **R1-Zero.** 从 DeepSeek-V3 base model 开始。没有 SFT。直接用两个 reward components 做 GRPO：*accuracy reward*（rule-based：最终答案能否 parse 到正确数字 / 代码是否通过 unit tests）和 *format reward*（completion 是否把 chain-of-thought 包在 `<think>…</think>` 标签里）。经过数千步，平均 response length 从约 100 增长到约 10,000 tokens，math benchmark 分数爬到接近 o1-preview。模型从零学会推理。缺点是：chain of thought 经常不可读、混合语言、风格不够干净。
- **R1.** 用四阶段 pipeline 修复 R1-Zero 的可读性问题：
  1. **Cold-start SFT.** 收集几千条格式干净的 long-CoT demonstrations。对 base model 做 supervised-finetune，得到可读的起点。
  2. **Reasoning-oriented GRPO.** 使用 accuracy+format rewards，加一个 *language-consistency* reward 防止 code-switching。
  3. **Rejection sampling + SFT round 2.** 从 RL checkpoint 采样约 600K 条 reasoning trajectories，只保留 final answers 正确且 CoT 可读的样本，再与约 200K 条 non-reasoning SFT examples（writing、QA、self-cognition）合并。再次 fine-tune base。
  4. **Full-spectrum GRPO.** 再做一轮 RL，覆盖 reasoning（rule-based rewards）与 general alignment（helpfulness/harmlessness preference-based rewards）。

结果是在 open weights 下，AIME 与 MATH-500 上匹配 o1，并且足够小，可以蒸馏。同一论文还发布了六个 distilled dense models（Qwen-1.5B 到 Llama-70B），通过在 R1 reasoning traces 上做 SFT 得到，student 不做 RL。强 RL teacher 的 distillation，在 student scale 上稳定优于从零 RL。

**为什么 reasoning 用 GRPO 而不是 PPO。** DeepSeekMath paper（2024 年 2 月）给出三个原因：(1) 不训练 value network，显存减半；(2) group baseline 天然适配 reasoning tasks 的 sparse end-of-trajectory reward；(3) per-prompt normalization 让不同难度问题之间的 advantages 可比较，而 PPO 的单个 critic 很难做到。

**Search-free vs search-based.** 游戏已经分叉：

- *Perfect-information games with long horizons*（Go、chess）：仍是 search-based。AlphaZero / MuZero 占主导。
- *LLM reasoning*：生产中还没有 MCTS；用 full rollouts 上的 GRPO，inference compute 用 best-of-N。Process reward models（PRMs）暗示 step-level search 可能会被加回来。

## Build It / 动手构建

`code/main.py` 实现的是 **GRPO 的微缩版**：一个带多组 samples 的 bandit。算法和 LLM 上的完全相同，只是 policy 和 environment 更简单。它教授的是 *loss* 与 *group-relative advantage*，也就是 2025 年的关键创新。

### Step 1: a tiny verifier environment / 一个极小的 verifier 环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

真实 GRPO 中，verifier 会运行 unit tests 或检查数学等价性。

### Step 2: policy: softmax over K answer tokens per prompt / policy：每个 prompt 上对 K 个 answer tokens 做 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于 LLM 在给定 prompt 后最后一层输出。

### Step 3: group sampling and group-relative advantage / group sampling 与 group-relative advantage

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

group-relative advantage 是 2024 年 DeepSeek 的关键技巧。不需要 critic。“baseline” 是 group mean，normalization 使用 group std。

### Step 4: compare to REINFORCE baseline (value-free) / 与 REINFORCE baseline（无 value）比较

同样 setup、同样 compute，用 plain REINFORCE。GRPO 收敛更快也更稳定。

### Step 5: observe entropy and KL / 观察 entropy 与 KL

诊断与 RLHF 相同：mean KL to reference、policy entropy、reward-over-time。它们稳定后，训练完成。

## Pitfalls / 常见陷阱

- **Reward hacking via verifier gaming.** GRPO 继承 RLHF 的风险：如果 verifier 错误或可被利用，LLM 会找到 exploit。robust verifiers（多 test cases、formal proofs）很关键。
- **Group size too small.** group baseline 的方差大约按 `1/√G` 缩小。低于 `G = 4` 时，advantage signal 很噪；标准选择是 `G = 8` 到 `64`。
- **Length bias.** 不同长度的 LLM completions 有不同 log-probabilities。按 token count 归一化，或使用 sequence-level log-prob，或截断到 max length。
- **Pure self-play cycles.** AlphaZero-style training 在 general-sum games 上可能陷入 dominance loops。用 diverse opponent pools（league play，Lesson 10）缓解。
- **Search-policy mismatch.** AlphaZero 训练 policy 模仿 search output。如果 policy net 太小，无法表示 search distribution，训练会停滞。
- **Compute floor.** MuZero / AlphaZero 需要巨大算力。一次 ablation 往往是数百 GPU-hours。学习用 miniature demos（例如 Connect Four 上的 AlphaZero）。
- **Verifier coverage.** 如果 unit tests 对 buggy solution 也通过，训练会强化 bug。设计能捕捉 edge cases 的 verifiers。

## Use It / 应用它

2026 年 game-RL landscape：

| Domain | Dominant method |
|--------|-----------------|
| Two-player zero-sum board games (Go, chess, shogi) | AlphaZero / MuZero / KataGo |
| Imperfect info card games (poker) | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / pixel games | Muesli / MuZero / IMPALA-PPO |
| Large multiplayer strategy (Dota, StarCraft) | PPO + self-play + league (OpenAI Five, AlphaStar) |
| LLM math/code reasoning | GRPO (DeepSeek-R1, Qwen-RL, open replications) |
| LLM alignment | DPO / RLHF-PPO (not GRPO; verifier is preference not verifiable) |
| Robotics | PPO + DR (not game-RL, but uses same policy-gradient tools) |
| Combinatorial problems | AlphaZero variants (AlphaTensor, AlphaDev) |

这个 *recipe*——self-play、search-augmented improvement、policy distillation——跨越文本、像素和物理控制。GRPO 是最新实例；后面还会有更多。

## Ship It / 交付它

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Exercises / 练习

1. **Easy.** 在 `code/main.py` 中实现 GRPO bandit。对 2 个 prompts × 每个 4 个 answer tokens 训练。使用 `G=8`，在 < 1,000 updates 内收敛。
2. **Medium.** 接入 PPO（clipped）和 vanilla REINFORCE。与同一 bandit 上的 GRPO 比较 sample efficiency 和 reward variance。
3. **Hard.** 扩展到长度为 2 的 “reasoning chain”：Agent 发出两个 tokens，verifier 对 token pair 给奖励。测量 GRPO 如何处理两步序列上的 credit assignment。（Hint: 对 *full sequence* 计算 group advantage，并传播到两个 token positions。）

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| MCTS | “Tree search with learned net” | Monte Carlo Tree Search；使用 learned `(p, v)` priors 的 UCB1/PUCT selection。 |
| AlphaZero | “Self-play + MCTS” | policy-value net 训练目标是 MCTS visits 和 game outcome。 |
| MuZero | “Learned-model AlphaZero” | 同一循环，但通过 learned dynamics 在 latent space 中运行。 |
| GRPO | “Critic-free PPO” | Group Relative Policy Optimization；带 group-mean baseline + KL 的 REINFORCE。 |
| PUCT | “AlphaZero's UCB” | `Q + c · p · √N / (1 + N_a)`，在 value estimate 与 prior 之间平衡。 |
| Self-play | “Agent vs past self” | zero-sum 的标准方法；训练信号对称。 |
| League play | “Population-based self-play” | 采样 past + current + exploiters 作为 opponents。 |
| Verifier reward | “Verifiable RL” | reward 来自 deterministic checker（tests pass、answer matches）。 |
| Process reward | “PRM” | 对每个 reasoning step 打分，而不是只看 final answer。 |

## Further Reading / 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 提出 GRPO 与 group-relative baseline 的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整四阶段 R1 recipe 与 R1-Zero ablation。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + deep-learning at scale。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 一切开始的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 使用 custom reward functions 应用 GRPO 的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 多尺度 R1 recipe 开源复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — self-play、search、designed reward 的教材 framing，R1 可以看作它在 LLM scale 上的实例。
