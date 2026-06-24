# Negotiation and Bargaining / 协商与议价

> Agent 会协商资源、价格、任务分配和条款。2026 年 benchmark 已经很清楚：NegotiationArena（arXiv:2402.05863）显示 LLM 可以通过 persona manipulation（“desperation”）把 payoff 提升约 20%；"Measuring Bargaining Abilities"（arXiv:2402.15813）显示 buyer 比 seller 更难，scale 并不能解决问题，而他们的 **OG-Narrator**（确定性 offer generator + LLM narrator）把 deal rate 从 26.67% 推到 88.88%；Large-Scale Autonomous Negotiation Competition（arXiv:2503.06416）跑了约 180k 次谈判，发现 **chain-of-thought-concealing** agents 通过隐藏 reasoning 胜出；Bhattacharya et al. 2025 基于 Harvard Negotiation Project metrics 认为 Llama-3 最有效、Claude-3 更强势、GPT-4 最公平。本课实现 Contract Net Protocol（FIPA 祖先，Lesson 02），接入 LLM-style buyer/seller，运行 OG-Narrator 风格拆解，并测量 deal rate 如何随结构选择变化。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 02（FIPA-ACL Heritage）, 第 16 阶段 · 09（Parallel Swarm Networks）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 用 Contract Net Protocol 表达 task-market negotiation
- 解释 OG-Narrator 为什么把 offer generation 和 natural-language narration 分开
- 识别 persona manipulation、CoT concealment、private scratchpad 泄漏等谈判风险
- 为 Agent bargaining 设计 round bound、schema validation、ZOPA check 和 payoff monitoring

## The Problem / 问题

两个 Agent 需要就价格达成一致。让它们只靠自然语言 prompt 自由谈判时，2024-2026 的 LLM 在严格参数化 bargain 上 deal rate 出奇地低（arXiv:2402.15813 中约 27%）。规模不能解决问题：GPT-4 结构上不比 GPT-3.5 更会 bargaining；它只是更会说 bargaining 的话。

根因是 LLM 把两件事混在一起：决定 offer 和叙述 offer。OG-Narrator 把它们拆开：确定性 offer generator 计算数值动作；LLM 只负责叙述。deal rate 跳到约 89%。

这呼应了经典多 Agent 发现：把 mechanism 和 communication layer 解耦会赢。Contract Net Protocol（FIPA, 1996; Smith, 1980）是任务市场机制参考。把 LLM 放进 narration slot，就得到现代 LLM-powered task market。

## The Concept / 概念

### Contract Net, in one paragraph / 一段话讲 Contract Net

Smith 1980 年的 Contract Net Protocol：**manager** 广播 **call for proposals (cfp)**；**bidders** 返回包含 offer 的 **propose** 消息；manager 选择 winner，并向 winner 发送 **accept-proposal**，向 loser 发送 **reject-proposal**。winner 执行工作。可选消息：**refuse**（bidder 拒绝提案）。FIPA 将其编码为 `fipa-contract-net` interaction protocol。

### Why OG-Narrator wins / OG-Narrator 为什么赢

"Measuring Bargaining Abilities of Language Models"（arXiv:2402.15813）观察到：

- LLM 经常破坏 bargaining rules（给出荒谬价格、忽视对方 ZOPA）。
- 它们 anchoring 很差（接受糟糕 first offers；用象征性而非战略性数额 counter-offer）。
- scale alone 无法修复。更大模型语言更像样，但战略错误类似。

OG-Narrator decomposition：

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

offer generator 是经典 negotiation strategy：Rubinstein bargaining model、Zeuthen strategy，或价格上的简单 tit-for-tat。LLM 负责叙述。消息包含确定性 price 和 natural-language framing。

deal rate 上升，因为：

- 价格留在 bargaining zone。
- anchors 是战略性的，不是情绪性的。
- LLM 做自己擅长的事：写。

### NegotiationArena findings / NegotiationArena 发现

arXiv:2402.05863 提供标准 benchmark。关键发现：

- LLM 可以通过 persona（“I am desperate to sell this by Friday”）把 payoff 提升约 20%；persona manipulation 是真实策略。
- fair/cooperative agents 会被 adversarial agents 利用；防御需要显式 counter-posturing。
- symmetric pair-ups 在约 40% benchmark scenarios 上收敛到不公平结果。

这不是“LLM 不会谈判”，而是“LLM 谈判太像人类，包括人类可被利用的部分”。

### Chain-of-thought concealment / 隐藏思维链

Large-Scale Autonomous Negotiation Competition（arXiv:2503.06416）跑了约 180k 次谈判。胜出者隐藏自己的 reasoning：

- 如果 Agent 把 “I will only go to $75; my reservation price is $70” 写进公开 scratchpad，对手会读到。
- 胜者私下计算策略；输出通道只包含 offer 和最低限度叙述。

这是 2026 年对经典博弈论（Aumann 1976 关于 rationality and information）的回应：公开 private valuation 会损害 payoff。LLM 不会本能理解这一点，常把 reservations 写进对手可见的 reasoning trace。

工程结论：把 private-scratchpad context 和 public-message context 分开。这不是可选项。

### Bhattacharya et al. 2025 — model rankings / 模型风格差异

基于 Harvard Negotiation Project metrics（principled negotiation、BATNA respect、interest reciprocity）：

- **Llama-3** 最有效地达成 bargain（deal rate + payoff）。
- **Claude-3** 是最 aggressive negotiator（高 anchor、晚让步）。
- **GPT-4** 最公平（pairings 之间 payoff 方差最小）。

这是 2025 年快照。重点不是 2026 年 4 月哪个模型赢，而是不同 base model 具有稳定 negotiation styles。heterogeneous ensembles（Lesson 15）可以把这当作 diversity source。

### Task allocation via Contract Net + LLM / 用 Contract Net + LLM 做任务分配

现代 LLM 多 Agent 复用 Contract Net 的方式：

1. manager agent 把任务拆成 units。
2. 广播带任务描述的 `cfp` 给 worker agents。
3. 每个 worker 返回 offer：`(price, eta, confidence)`，其中 price 可以是 tokens、compute units 或 dollars。
4. manager 选择 winner（单个或多个，取决于任务）并 award。
5. 被 reject 的 worker 可以继续 bid 其他任务。

这比同步 chat 更能扩展到 100+ workers，因为协调是 broadcast-and-respond。生产中可见于 Microsoft Agent Framework orchestration patterns 和部分 LangGraph implementations。

### LLM-Stakeholders Interactive Negotiation / 多方利益相关者谈判

NeurIPS 2024 引入带 **secret scores** 和 **minimum-acceptance thresholds** 的 multi-party scorable games。每个 stakeholder 有私有 utility；LLM 必须从消息中推断它们。这是两方 bargaining 到 N 方 coalition formation 的推广。它对具备异构 worker capabilities 的生产 task market 很相关。

### The narration-vs-mechanism rule / 叙述与机制分离规则

2024-2026 所有 negotiation benchmark 的一致工程规则：

> Let the LLM narrate. Do not let the LLM compute the offer.

如果 offer 需要是数字（price、ETA、quantity），从 negotiation state 确定性计算，再让 LLM 生成 framing。如果 offer 需要 proposal structure（task decomposition、role assignment），可以让 LLM draft，但发送前必须 schema validate 和 constraint-check。

## Build It / 动手构建

`code/main.py` 实现：

- `ContractNetManager`, `ContractNetTask`, `Bid` — manager + bidders，broadcast cfp、collect proposals、award。
- `og_narrator_bargain(state, rng)` — OG-Narrator buyer：确定性 Zeuthen-style concession，向 midpoint 让步。
- `seller_response(state, rng)` — 确定性 seller counter-offer policy（两种风格的 structural ground truth）。
- `naive_llm_bargain(state, rng)` — 模拟 all-LLM bargainer：高方差选价，经常超出 ZOPA。
- Measurement：1000 trials 的 deal rate，每轮随机采样 reservation prices。

运行：

```
python3 code/main.py
```

预期输出：naive-LLM deal rate 约 65-75%；OG-Narrator deal rate 约 85-95%；15-25 个百分点差距来自 offer-generation 和 narration 解耦的结构优势。另有一个三 bidder、一任务的 Contract Net task-market allocation 示例。

## Use It / 应用它

`outputs/skill-bargainer-designer.md` 设计 bargaining protocol：谁生成 offers（deterministic 或 LLM），谁 narrate，private scratchpad 如何与 public messages 分离，以及如何监控 deal rate。

## Ship It / 交付它

生产 bargaining checklist：

- **Separate scratchpad.** private state 永远不进入 counterpart context。这条不可妥协。
- **Deterministic offer generation.** price、quantity、ETA：计算，不要 prompt。
- **Validate all incoming offers** against a schema。在协议边界拒绝 out-of-ZOPA offers。
- **Bound rounds.** 最多 3-5 轮；deadlock 时升级给 mediator。
- **Measure deal rate and payoff variance** continuously。deal rate 下降是症状，常见原因是 prompt drift 或 counterpart-side attack。
- **Log all rejected proposals** with deterministic rationale。Contract Net manager 要让 losing bidders 明白原因。

## Exercises / 练习

1. 运行 `code/main.py`。确认 OG-Narrator 在 deal rate 上超过 naive-LLM。差距多大？
2. 实现 **persona-based payoff improvement**（arXiv:2402.05863）：buyer 只在 narration 中采用 “desperate to buy this week” persona，offer generator 不变。deal rate 或 payoff 变化了吗？
3. 实现 chain-of-thought **concealment**：维护一个不传给 counterpart 的 private scratchpad string。如果不小心泄漏（模拟为交换通道），会发生什么？
4. 扩展 Contract Net 到 N-bidder auction，带 reserve price。当所有 bid 都超过 reserve 时，manager 如何在 lowest-price 和 highest-quality 之间选择？你选哪个 award rule，为什么？
5. 阅读 Bhattacharya et al. 2025 的 Harvard Negotiation Project metrics。实现两个风格不同的 bargainers（aggressive vs fair）。测量 symmetric 与 asymmetric pairings 下的 payoff variance。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Contract Net | “Task market” | Smith 1980、FIPA 1996。cfp + propose + accept/reject。标准任务市场。 |
| ZOPA | “Zone of possible agreement” | buyer max 与 seller min 的重叠区。区外 offer 无法成交。 |
| BATNA | “Best alternative to a negotiated agreement” | deal 失败时的 fallback。决定 reservation price。 |
| OG-Narrator | “Offer generator + narrator” | 拆解：确定性 offer，LLM narration。 |
| Zeuthen strategy | “Risk-minimizing concession” | 根据风险限度让步的经典 offer-generator。 |
| Rubinstein bargaining | “Alternating-offer equilibrium” | 带 discounting 的无限期交替报价博弈模型。 |
| CoT concealment | “隐藏 reasoning” | arXiv:2503.06416 的胜者保留 private scratchpads；public channel 只显示 offer。 |
| Persona manipulation | “情绪姿态” | arXiv:2402.05863：desperation/urgency persona 带来约 20% payoff gain。 |

## Further Reading / 延伸阅读

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — benchmark；persona manipulation 与 exploitation findings
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator 与 buyer-harder-than-seller 结果
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — 约 180k 次谈判；chain-of-thought concealment 胜出
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — 带 secret utilities 的多方 scorable games
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — 经典机制，IEEE Transactions on Computers
