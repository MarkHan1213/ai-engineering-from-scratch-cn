# Consensus and Byzantine Fault Tolerance for Agents / Agent 的共识与拜占庭容错

> 经典分布式系统 BFT 遇上随机性的 LLM。2025-2026 年出现三条研究方向：**CP-WBFT**（arXiv:2511.10400）用 confidence probe 给每票加权；**DecentLLMs**（arXiv:2507.14928）去 leader，worker 并行提案并用 geometric-median 聚合；**WBFT**（arXiv:2505.05103）把 weighted voting 与 Hierarchical Structure Clustering 结合，将节点分成 Core 和 Edge。来自 "Can AI Agents Agree?"（arXiv:2603.01213）的诚实经验结论是：今天就连 scalar agreement 都很脆弱，一个 deceptive agent 就能破坏 Mixture-of-Agents。BFT 必要但不充分。本课实现最小 BFT 协议，注入三种 Agent 特有攻击（byzantine lie、sycophantic conformity、correlated-error monoculture），并测量不同共识变体如何应对。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 07（Society of Mind and Debate）, 第 16 阶段 · 13（Shared Memory）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释经典 PBFT 给出的保证，以及 LLM Agent 违反了哪些假设
- 区分 byzantine lie、sycophantic conformity、correlated-error monoculture 三类攻击
- 比较 majority vote、CP-WBFT、DecentLLMs、WBFT 的缓解思路
- 为 Agent ensemble 设计带 semantic clustering、confidence weighting、threshold 和 escalation 的共识协议

## The Problem / 问题

你有 N 个 LLM Agent，各自产生答案。它们不同意。多数投票选错了，因为两个 Agent 高度相关（同 base model、同训练数据、同失败模式）。第三个 Agent 以一种新方式错误，于是多数变成假多数。

再加一个 deceptive agent：故意说谎。或者一个 sycophantic agent：同意最后说话的人。在经典 BFT 中，假设 Byzantine nodes 占比 `f < n/3`，并可任意行为。2026 年现实是：LLM node 即使诚实也随机，模型之间错误相关，还会被彼此输出影响。你不能把它们当独立 Bernoulli voters。

经典 BFT（PBFT, 1999）不是错，而是不完整。它处理任意 bit-flipping，不处理“三个诚实 Agent 因为同训练数据共享同一个 hallucination”。本课从 PBFT 基础开始，叠加三个 2025-2026 适配。

## The Concept / 概念

### What classical BFT gives you / 经典 BFT 给你什么

Practical Byzantine Fault Tolerance（Castro & Liskov, OSDI 1999）容忍 `f < n/3` Byzantine nodes。协议有三阶段（pre-prepare、prepare、commit）和两个 primitive（signed messages、quorum certificates）。在 `n >= 3f + 1` 个诚实或恶意节点中，对单个值达成一致。

保证很强，但假设：

1. **Independent faults.** Byzantine 不协作。
2. **Honest nodes are truly honest.** 诚实输出的正确性不是问题；协议只对齐分歧。
3. **The question has a ground-truth answer.** 对错误事实达成共识仍然只是共识。

LLM Agent 三条都违反。两个跑同一 base model 的 Agent 共享故障。“诚实”的 LLM 仍会 hallucinate。模糊问题上，“truth” 可能只是 Agent 决定的东西，没有外部 oracle。

### The three LLM-specific attacks / 三种 LLM 特有攻击

**Byzantine lie.** 一个 Agent 故意输出错误答案。经典 BFT 在 `f < n/3` 时能处理。

**Sycophantic conformity.** 一个 Agent 投票前读取其他答案，并向最后/最自信说话者靠拢。不是恶意，但会与最大声观点相关。经典 BFT 阻止不了，因为它能通过所有签名检查。

**Correlated-error monoculture.** 三个 Agent 共享同一 base model。它们 hallucinate 同一个错误答案。多数错了。经典 BFT 无能为力，因为三者“诚实”同意。

### The 2025-2026 responses / 2025-2026 的应对

**CP-WBFT**（arXiv:2511.10400）— Confidence-Probed Weighted BFT。每个 voter 为答案附带 confidence probe（自报概率，或独立校准模型预测）。vote weight 随 confidence 缩放。在 complete graphs 上报告 +85.71% BFT improvement。缓解：sycophantic conformity（随声附和的 Agent 对自愿立场通常 confidence 较低）。

**DecentLLMs**（arXiv:2507.14928）— leaderless。worker Agent 并行提 proposal，evaluator Agent 评分，最终答案是 scored positions 的 geometric median。当 `f < n/2` 时鲁棒。缓解：Byzantine lie 和 correlated errors（geometric median 对 outlier 鲁棒，趋向 dense cluster，而不是 model-biased average）。

**WBFT**（arXiv:2505.05103）— Weighted BFT with Hierarchical Structure Clustering。vote weight 由 response quality 加历史 trust score 决定。把 Agent 聚成 Core 和 Edge；Core 先达成共识，Edge 跟随。缓解：scalability（Core consensus 小且快），部分缓解 monoculture（Core 可以按多样性选择）。

### Empirical: "Can AI Agents Agree?" / 经验结果

"Can AI Agents Agree?"（arXiv:2603.01213）测量多个 frontier model 的 scalar agreement（LLM Agent 对单个数值达成一致）。结果并不好看：

- 没有对手时，LLM Agent 在许多 benchmark 的 scalar questions 上 disagreement rate 仍高于 30%。
- 一个采用 deceptive persona 的 Agent 能把 Mixture-of-Agents 共识拉离 honest baseline 40+ 个百分点。
- disagreement rate 与 model diversity 相关：异质 ensemble 分歧更多（好处：uncorrelated errors），但 drift 更慢（坏处：time-to-agreement 更长）。

结论：BFT 给你对齐输出的机械结构，但不告诉你对齐后的输出是否正确。要结合 verification（Phase 16 · 08 role specialization）、diversity（Phase 16 · 15 debate variants）和 evaluator agents（Phase 16 · 24 benchmarks）。

### The core protocol, stripped down / 极简核心协议

最小 LLM Agent BFT round：

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

semantic clustering 是 LLM 特有扭转。两个答案 “the study reports 4.2%” 和 “4.2% improvement” 是同一 cluster。朴素字符串相等检查会漏掉。生产里用便宜 embedding model 或显式 canonicalization。

### Threshold tuning / 阈值调优

`threshold` 参数决定何时接受、何时重试。太低会接受弱多数；太高则永远不接受。经验范围：`n=5-7` Agent 时为 0.5-0.67，小 N 时更高。低于阈值时，升级给人或换一个 Agent ensemble。

### Where consensus does not help / 共识没有帮助的地方

- **Ambiguous questions.** 问题没有 ground truth 时，共识只是意见，要这样标注。
- **Compound questions.** “写代码并解释它”是两个答案。应分别投票。
- **Adversarial multi-round.** 如果 Agent 能观察先前轮次并模仿（Du 2023 debate），它们会不管真相地互相趋同。限制轮数（通常 2-3）。

## Build It / 动手构建

`code/main.py` 实现：

- `AgentVoter` — 带 `(answer, confidence)` 的 scripted policy。
- `MajorityVote` — 经典 plurality。
- `CPWBFT` — 带 semantic clustering 的 confidence-weighted voting。
- `DecentLLMs` — scored proposals 上的 geometric-median aggregation。
- `Scenario` — 在三种 attack pattern 下运行每种 aggregator。

实现的攻击模式：

1. `byzantine`：一个 Agent 高 confidence 地撒谎。
2. `sycophancy`：一个 Agent 复制它看到的第一个答案，并匹配 confidence。
3. `monoculture`：三个 Agent 共享 moderate confidence 的错误答案（correlated error）。

运行：

```
python3 code/main.py
```

预期输出：一张 (attack, aggregator) -> final answer 表，并标出正确答案。plurality 在 monoculture case 失败。CPWBFT 的 confidence weighting 缓解 sycophancy。DecentLLMs 的 geometric-median 在 monoculture 少于半数时拉向 honest cluster。

## Use It / 应用它

`outputs/skill-consensus-designer.md` 为 multi-agent ensemble 设计共识协议：clustering method、weighting、threshold，以及 sub-threshold rounds 的 escalation policy。

## Ship It / 交付它

上线任何 consensus mechanism 前：

- **Attack-test with at least the three patterns** above。协议应可预测地失败，而不是静默失败。
- **Log every minority cluster** with provenance。minority cluster 是 correlated errors 的早期预警。
- **Enforce bounded rounds.** 不要“辩到同意为止”，这奖励 sycophancy。
- **Separate agreement from correctness.** 共识输出交给 verifier；verifier 独立于 ensemble。
- **Monitor the agreement rate.** sharp rise 说明 conformity bias；sharp fall 说明 model drift。

## Exercises / 练习

1. 运行 `code/main.py`。确认 plurality 在 monoculture attack 下失败，但当 monoculture confidence 低于 0.7 时，CPWBFT 能部分缓解。
2. 增加第四种攻击：**silent abstention**，一个 Agent 拒答（"I don't know"）。每个 aggregator 应如何处理 abstention？实现你的选择。
3. 把 semantic clustering 从 string canonicalization 换成 embedding-similarity（使用任意开源 embedding model）。sycophancy attack 会发生什么变化？
4. 阅读 CP-WBFT（arXiv:2511.10400）。实现 confidence-probe calibration step（独立校准模型检查每个 Agent 自报 confidence）。测量 monoculture scenario 上的 accuracy gain。
5. 阅读 "Can AI Agents Agree?"（arXiv:2603.01213）。复现一个简化 scalar-agreement 实验：三个 Agent，一个 scalar question，一个 deceptive-persona prompt。CPWBFT 或 DecentLLMs 能抓住吗？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| BFT | “Byzantine fault tolerance” | Castro-Liskov 1999 共识协议，容忍 `f < n/3` 任意故障。 |
| Byzantine | “任何坏行为” | 节点可以撒谎、丢消息、静默失败，除了安全 crash 以外的任意行为。 |
| Confidence probe | “你有多确定？” | 附加到 vote 的自报或校准器预测概率。 |
| Semantic clustering | “同答案，不同说法” | 在计票前把等价答案分组。 |
| Geometric median | “鲁棒中心” | 使到样本点距离和最小的点。相比 mean，对 outlier 更鲁棒。 |
| Monoculture | “同模型，同失败” | Agent 共享训练数据或 base model 造成 correlated errors。 |
| Sycophantic conformity | “跟最大声的人走” | Agent vote 偏向最先/最大声 speaker。 |
| Core/Edge | “Hierarchical BFT” | WBFT 拆分：小 Core 先 consensus，Edge 跟随。限制 latency。 |

## Further Reading / 延伸阅读

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — 基础论文
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — 按 confidence 给 vote 加权
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — geometric-median aggregation
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — Core/Edge split 控制 latency
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — scalar-agreement fragility 与 deceptive-persona attack
