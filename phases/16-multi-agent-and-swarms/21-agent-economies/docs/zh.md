# Agent Economies, Token Incentives, Reputation / Agent 经济、Token 激励与声誉

> 长程 autonomous agents（METR 的 1 小时到 8 小时 work-curve）需要经济主体能力。正在形成的 **5-layer stack** 是：**DePIN**（physical compute）→ **Identity**（W3C DIDs + reputation capital）→ **Cognition**（RAG + MCP）→ **Settlement**（account abstraction）→ **Governance**（Agentic DAOs）。生产级 Agent 激励网络包括 **Bittensor**（TAO subnets 奖励 task-specific models）、**Fetch.ai / ASI Alliance**（ASI-1 Mini LLM + FET token）和 **Gonka**（transformer-based PoW，将 compute 重分配到有生产价值的 AI tasks）。学术工作：AAMAS 2025 decentralized LaMAS 使用 **Shapley-value credit attribution** 公平奖励贡献 Agent；Google Research "Mechanism design for large language models" 提出 **token auctions**，在 monotone aggregation 下使用 second-price payment。本课构建最小 Agent marketplace，把 Shapley-value credit attribution 应用到多 Agent pipeline，并运行 second-price token auction，让博弈论机制具体落地。

**类型：** 学习
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 16（Negotiation and Bargaining）, 第 16 阶段 · 09（Parallel Swarm Networks）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 Agent economy 的 5-layer stack：DePIN、Identity、Cognition、Settlement、Governance
- 用 Shapley value 做多 Agent credit attribution，并理解其可扩展性限制
- 实现 second-price auction 和 DID-bound reputation update
- 判断何时需要 token economics，何时只需要内部 reputation / routing metrics

## The Problem / 问题

当多个 Agent 共同产生价值却需要分别获得奖励时，多 Agent 系统会变复杂。经典机制（equal split、last-contributor-takes-all）要么不公平，要么可被操纵。基于 coalition 的 Shapley value 奖励在构造上公平，但计算昂贵。2025-2026 文献提出了一些有用近似：Shapley sampling、monotone aggregation auctions，以及从已确认贡献中累积的 on-chain reputation。

除了 credit attribution，领域已经转向真实经济 Agent：Bittensor TAO 奖励 mining compute 微调 subnet-specific models，Fetch.ai/ASI 用 FET tokens 奖励 ASI-1 Mini LLM usage，Gonka 把 transformer proof-of-work 重新分配到有生产价值的 AI tasks。能自主交易的 Agent 已经存在；问题是如何对齐激励。

本课把 Agent economies 作为一个具体问题族：credit attribution、mechanism design 和 reputation，并用最小数学实现它们。

## The Concept / 概念

### The 5-layer agent-economy stack / Agent economy 五层栈

1. **DePIN (physical compute).** 去中心化基础设施，用 token 激励 GPU、storage、bandwidth 出租。Bittensor subnets、Render Network、Akash。不特指 Agent，Agent 使用它。
2. **Identity.** W3C Decentralized Identifiers（DIDs）给每个 Agent 一个不依赖平台的 durable ID。reputation 绑定到 DID。Agent Network Protocol（ANP）用 DID 作为 discovery layer。
3. **Cognition.** Agent reasoning loop：LLM + RAG + MCP。这是其他 phase 构建的部分。
4. **Settlement.** Account abstraction（ERC-4337）允许 Agent 用自己的余额支付 gas，而不直接持有 ETH。Agent 可以为服务、彼此或 compute 付款。
5. **Governance.** Agentic DAOs：人和 Agent 都可以对 protocol changes 投票，voting power 绑定 reputation。

不是每个生产系统都使用五层。Bittensor 使用 1、2，部分 3、部分 4，没有 5。OpenAI agents 除 3 外不使用。这个 stack 是参考地图，不是必选清单。

### Bittensor, Fetch.ai, Gonka — what runs / 已在运行的网络

**Bittensor (TAO).** subnets 是专门任务（language modeling、image generation、forecasting）。miners 提交 model outputs。validators 排名；stake-weighted scoring 分配 TAO 奖励。每个 subnet 有自己的 evaluation。经济教训：按 task-specific output quality 支付，而不是按 compute used 支付。

**Fetch.ai / ASI Alliance.** ASI-1 Mini LLM 运行在 Fetch.ai 网络上；用户用 FET tokens 为 inference 付费。agents-as-peers 叙事更强：Fetch 上的 Agent 可以调用另一个 Agent 完成任务并用 FET 付款。

**Gonka.** Transformer proof-of-work：“work” 是 transformer forward passes。miners 通过运行有已知正确输出（来自训练数据）的 inference tasks 获得收益。用资源产生价值的 PoW，而不是 hash-based PoW。

截至 2026 年 4 月，三者都是 production-grade。payoff distribution 不同：Bittensor 按 subnet validators 评估的质量奖励；Fetch 按 paying users 衡量的 utility 奖励；Gonka 奖励可验证 inference work。

### Shapley-value credit attribution / Shapley 值信用归因

三个 Agent 协作完成任务，输出得分 0.8。谁贡献了多少？

Shapley value 是唯一满足四个公理（efficiency、symmetry、linearity、null）的 credit allocation。对 Agent `i`：

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

其中 `S_i_O` 是 ordering `O` 中排在 `i` 前面的 Agent 集合。实践中：枚举所有 permutations，记录每个 Agent 在每个 permutation 中的 marginal contribution，再取平均。

N=3 时有 6 个 permutations。N=10 时有 3.6M，所以生产里一般 sample orderings 而不是枚举。

### Second-price auction for aggregation / 用 second-price auction 做聚合

Google Research（"Mechanism design for large language models"）提出用 second-price token auctions 聚合 LLM outputs。设置：N 个 Agent 各自提出 completion；每个 Agent 对被选中有 private value。auctioneer 选择 highest-value proposal，并支付 *second-highest* value。在 monotone aggregation（value 取决于哪个 proposal 被选中，而不是 bid 数量）下，这是 truthful 的：Agent 有动力报真实 value。

对 LLM 系统的意义：你可以把 completion tasks 外包给多个不同定价的 Agent；auction 选择最好结果并公平支付，Agent 没有动机误报。

### Reputation capital / 声誉资本

DID-bound reputation score 从已确认贡献中累积。简单 update rule：

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

`alpha` 接近 1，表示 decay factor。Reputation：

- 对 routing decisions 便宜可读（“hard tasks 发给 high-rep agents”）。
- 难以伪造（随时间累积，绑定 DID）。
- 可以 slash：贡献未通过 verification 时扣减。

### AAMAS 2025 decentralized LaMAS / 去中心化 LaMAS

LaMAS proposal（AAMAS 2025）组合了 DID identity、Shapley-value credit attribution 和简单 auction mechanism。关键 claim：把 credit attribution step 去中心化，让系统可审计，并免于单点操纵。

### Where the economics falls apart / 经济机制会在哪里崩

- **Price oracle manipulation.** 如果 credit function 可被操纵，Agent 就会操纵它。每个机制都需要 adversarial test。
- **Sybil attacks.** 一个 operator 启动 N 个假 Agent 来放大自己的贡献。DIDs 只能减缓不能阻止；reputation cost-to-forge 是缓解。
- **Verification cost.** credit attribution 的公平性取决于 verifier。verification 太便宜（小 LLM）会被 game；太贵（human panel）不可扩展。
- **Regulatory overhang.** Agent economies 与金融监管相交。截至 2026 年，Bittensor、Fetch、Gonka 在某些司法辖区都处于法律灰区。

### When agent economies make sense / 什么时候需要 Agent economy

- **Open networks with heterogeneous operators.** 没有一个团队控制所有 Agent。
- **Verifiable outputs.** 没有 verification，credit attribution 只是猜测。
- **Long-horizon workflows.** one-shot tasks 不太受益于 reputation accumulation。
- **Tokenized payments are legally viable** in your jurisdiction。

在封闭企业系统中，economics 通常让位于更简单的 allocation（manager 分配工作，metrics 内部化）。经济学文献主要适用于开放网络。

## Build It / 动手构建

`code/main.py` 实现：

- `shapley(value_fn, agents)` — 小 N 精确 Shapley enumeration。
- `second_price_auction(bids)` — truthful mechanism；winner 支付 second-highest。
- `Reputation` — 带 exponential decay 和 slashing 的 DID-bound reputation。
- Demo 1：三个 Agent 协作，用 exact Shapley 分配 credit。
- Demo 2：五个 Agent 为一个 task slot bid；second-price auction 选择 winner + payment。
- Demo 3：100 轮 task assignment 给不同 rep 的 Agent；warmup 后，rep-weighted routing 比 random 更好。

运行：

```
python3 code/main.py
```

预期输出：每个 Agent 的 Shapley values；auction result 展示 truthful-bid equilibrium；rep-weighted routing 在 warmup 后比 random 提升 10-20% quality。

## Use It / 应用它

`outputs/skill-economy-designer.md` 设计最小 Agent economy：identity layer、credit attribution mechanism、payment mechanism、reputation rule。

## Ship It / 交付它

2026 年运行 Agent economy：

- **Start with reputation, not tokens.** reputation 实现便宜且本身有价值；tokens 增加法律和经济复杂性。
- **Verify before you reward.** 没有 independent verification，不要分配 credit。self-reported quality 会带来 sybil games。
- **Shapley-sample, not Shapley-exact.** sample 100-1000 orderings；exact enumeration 不扩展。
- **Cap decay factor and floor reputation.** 无界 decay 会擦除合法贡献者；太慢 decay 会奖励过期 high-rep agents。
- **Audit mechanisms adversarially.** 开放网络前先运行 red-team scenarios。每个机制都有博弈论漏洞，最好由你先找到。

## Exercises / 练习

1. 运行 `code/main.py`。确认 Shapley values 之和等于 total value（efficiency axiom）。改变 value function；Shapley allocation 是否按预期变化？
2. 实现 Shapley *sampling*（Monte Carlo over K orderings）。K 如何影响近似准确率？与 N=4 的 exact 比较。
3. 在 auction 前实现 coalition-forming step：Agent 可以合并成 teams 并作为单位 bid。哪些 coalition 会形成？结果是否 Pareto-better than individual bidding？
4. 阅读 Google Research mechanism-design post。识别一个如果被违反就会破坏 truthfulness 的假设。该失败在 LLM setting 中长什么样？
5. 阅读 AAMAS 2025 decentralized LaMAS paper。在 synthetic task 上对 10 个 Agent 实现 Shapley step。exact computation 需要多久？100 draws sampling 有多接近？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| DePIN | “Decentralized physical infrastructure” | token-incentivized compute/storage/bandwidth。Bittensor、Akash、Render。 |
| DID | “Decentralized identifier” | W3C portable IDs。Agent reputation 绑定 DID，而非平台。 |
| ERC-4337 | “Account abstraction” | 可 sponsor gas 的 contract accounts，使 Agent payments 成为可能。 |
| Shapley value | “Fair credit attribution” | 满足 efficiency、symmetry、linearity、null 的唯一 allocation。 |
| Second-price auction | “Vickrey auction” | truthful mechanism：winner 支付 second-highest bid。兼容 monotone aggregation。 |
| Reputation capital | “Accumulated quality score” | 来自已确认贡献的 DID-bound score；随时间 decay。 |
| Agentic DAO | “Agents + humans govern” | Agent voters 作为一等公民、voting power 绑定 reputation 的 DAO。 |
| TAO / FET / GPU credits | “Token denominations” | Bittensor TAO、Fetch.ai FET、各类 DePIN tokens。 |

## Further Reading / 延伸阅读

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026 年 5-layer agent-economy stack 综述
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — monotone aggregation 下的 token auctions
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley-value credit attribution
- [Bittensor TAO documentation](https://docs.bittensor.com/) — subnet structure 与 reward distribution
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM 与 FET token
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — identity foundation
