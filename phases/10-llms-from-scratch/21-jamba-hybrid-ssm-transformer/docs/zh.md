# Jamba — Hybrid SSM-Transformer / Jamba 混合 SSM-Transformer

> State space models（SSMs）和 transformers 想要的东西不同。Transformers 用 quadratic cost 的 attention 换质量。SSMs 用 recurrence 换 linear-time inference 和 constant memory，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）把二者放进同一个模型：每 7 个 Mamba layers 配 1 个 Transformer layer，隔一个 block 用一次 MoE，并把 256k context window 放进单张 80GB GPU。Mamba-3（ICLR 2026）用 complex-valued state spaces 和 MIMO projections 收紧 SSM 侧。本课端到端阅读这两类架构，并解释为什么 pure-SSM 和 pure-Transformer 的 long-context 尝试没有活下来，而 hybrid recipe 经过三年 scaling 仍然存在。

**类型：** Learn
**语言：** Python (stdlib, layer-mix calculator)
**前置要求：** Phase 10 · 14（open-model architectures），Phase 10 · 17（native sparse attention）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 Jamba block 中的三个 primitives：Transformer layers、Mamba layers、MoE，以及 1:7:even interleaving recipe
- 从高层说明 SSM 的 recurrence 长什么样，以及为什么它支持 constant-memory inference
- 计算 Jamba model 在 256k context 下的 KV cache footprint，并与 pure-Transformer model 的需求对比
- 说出 Mamba-3 的三个 innovations（exponential-trapezoidal discretization、complex-valued state update、MIMO），以及每个解决的问题

## The Problem / 问题

Attention 对 sequence length 是二次复杂度。State space models 是线性的。这个差异会被放大：到 256k tokens 时，一个 Transformer attention map 每个 head 有 65B entries；SSM 的 recurrent state 不随 sequence length 变化，是固定大小。

Pure-SSM models（Mamba、Mamba-2）在小 scale 上能匹配 Transformer perplexity，但在 state-tracking tasks 上落后，并且在某些 in-context retrieval 类别上失败。直觉是：SSM 会把历史压进固定 state；历史很长时，信息会泄漏。Attention 精确记住所有东西，但付出 quadratic cost。

显然的修复是：两个都用。在需要 exact recall 的地方放 Transformer layers，其他地方用 SSM layers。调比例。Jamba 是第一个在 scale 上发布这种 hybrid recipe 的 production-grade model（52B total，12B active，256k context，单张 80GB GPU）。Jamba 1.5 把这个家族扩展到 398B total / 94B active。Mamba-3（ICLR 2026）是当前最好的 pure-SSM baseline，未来的 hybrid 可能会围绕它重建。

本课阅读三篇论文，并产出“如何选择正确比例”的 mental model。

## The Concept / 概念

### An SSM in one page / 一页理解 SSM

State space model 通过固定大小 state `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步中，state 通过 linear dynamics `A` 演化，接收输入 `B x_t`，并输出 `C h_t`。`A, B, C` 都可以学习。注意关键性质：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。内存是常数。Inference 每 token 是 O(1)。

建模质量的关键在于 `A` 的结构。S4（Gu 2021）使用一种高度结构化矩阵，训练时可以高效作为长 convolution 评估。Mamba（Gu, Dao 2023）把固定 `A, B, C` 换成 data-dependent ones（也就是 "selective" 部分）。Mamba-2（2024）进一步简化结构。Mamba-3（2026）在特定位置重新加入复杂性。

核心性质：对 decoder LLM 来说，SSM layer 可以替代 attention layer；它使用固定大小 per-layer state，而不是持续增长的 KV cache。

### The Jamba block / Jamba block

Jamba block 用两个数字交错 layers：

- `l`：attention-to-Mamba ratio。Jamba 使用 `l = 8`，表示每 7 个 Mamba layers 配 1 个 Transformer layer（7 Mamba + 1 Attention = 每组 8 layers）。
- `e`：MoE frequency。Jamba 使用 `e = 2`，表示每隔一层应用 MoE。

block 内的 layer sequence：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba block 有 8 层。4 个 blocks（总计 32 层）时，你得到 28 个 Mamba layers 和 4 个 Attention layers。其中 16 层使用 MoE。

### Why the 1:7 ratio / 为什么是 1:7 比例

AI21 做了 ablations：什么 attention-to-Mamba ratio 能在 perplexity-per-parameter 和 long-context evals 上的 in-context recall 之间取得最好平衡？

- 太多 attention（1:1）：质量上升，但内存和速度变差。
- 太少 attention（1:15）：内存很好，但 in-context retrieval 失败。
- sweet spot：1:7 或 1:8。

直觉是：Transformer layers 负责 exact recall 和 state tracking。Mamba layers 负责便宜的大量处理。

### Positional encoding / 位置编码

Mamba layers 本身通过 recurrence 具备位置感知能力。原始 Mamba-based hybrids 中的 attention layers 没有使用 RoPE；SSM layers 提供了 position info。Jamba 1.5 为 attention layers 加入 RoPE，以提升 long-context generalization，这是基于经验 long-context evaluation 的事后改进。

### The memory budget / 内存预算

以 Jamba-1 形状为例（32 layers：28 Mamba + 4 Attention，hidden 4096，32 attention heads）：

- KV cache（仅 attention layers）：在 256k BF16 下，`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。只有 4 个 attention layers 贡献 KV cache。
- SSM state：`28 * hidden * state_size` per token prefix，但这是固定大小 per layer，不随 sequence length 增长。典型 Mamba state 是每 feature 16，hidden 4096：`28 * 4096 * 16 * 2 = 3.7 MB` total。

与同样 32 layers、同样 hidden、32 heads 的 pure Transformer full MHA 对比：在 256k BF16 下是 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV cache 降低 8x。即使和多数 2024 模型使用的 GQA(8) baseline 对比（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 hybrid 在 16 GB 下仍然小 2x。

这就是 AI21 所说 “single 80GB GPU 上的 256k context”。full-MHA pure Transformer 的 KV cache 放不下；即使 GQA baseline，也几乎不给 weights 和 activations 留空间；Jamba 可以。

### Mamba-3: the pure-SSM baseline in 2026 / Mamba-3：2026 年 pure-SSM baseline

Mamba-3（ICLR 2026, arXiv:2603.15569）在 pure-SSM 侧引入三个 innovations：

1. **Exponential-trapezoidal discretization。** 用更有表达力的 recurrence 替代 Mamba-2 中的 Euler-method discretization。convolution-like operation 作用在 core recurrence 内部的 state-input 上，而不是作为 `x_t` 上的外部 convolution。

2. **Complex-valued state update。** 早期 Mambas 从 S4 的 complex state matrix 简化到 Mamba 的 real diagonal，再到 Mamba-2 的 scaled identity。Mamba-3 重新加入 complex values，等价于 state 上的 data-dependent rotary embedding。这恢复了此前 real-valued simplifications 损失的 state-tracking capabilities。

3. **Multi-input multi-output (MIMO) projections。** 使用 matrix-valued projections，而不是 per-feature scalar projections。在不增加 decode latency 的情况下提升 modeling power 和 inference-time hardware utilization。

在 1.5B 参数规模上，Mamba-3 相比 Gated DeltaNet 平均 downstream accuracy 提升 0.6 points；MIMO variant 再增加 1.2，总计 1.8 points。相同 state size 下，Mamba-3 用一半 state 就能匹配 Mamba-2。

Mamba-3 尚未在 production hybrid 中大规模发布，但它显然是下一代 Jamba-class model 中 SSM 侧的候选。

### When to reach for a hybrid / 什么时候使用 hybrid

Hybrids 在这些情况下胜出：

- context 足够长，pure Transformer KV cache 开始痛苦（64k+）。
- task 混合了 short-range structure（适合 SSM）和 long-range recall（需要 Transformer）。
- 希望部署在 single-GPU memory budgets 下，而 Transformer KV cache 本身就放不下。

Hybrids 在这些情况下劣势明显：

- context 较短（16k 以下）。SSM overhead 被浪费；pure Transformer 足够。
- task 需要 everywhere-to-everywhere attention（深度推理、多文档交叉引用）。hybrid 中 attention layers 稀疏会伤害表现。
- 你在 scale 到 trillion-parameter frontier models。目前 pure-Transformer + MLA + MoE（DeepSeek-V3 风格）正在赢 capability race。

### The competitive landscape / 竞争格局

| Model | Family | Scale | Unique claim |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | linear time, constant memory |
| Jamba | hybrid | 52B/12B | 256k on 80GB |
| Jamba 1.5 Large | hybrid | 398B/94B | enterprise-grade long-context |
| Mamba-3 | pure SSM | 1.5B (paper) | state-tracking restored |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | frontier capability |

2026 年格局：pure-Transformer MoE 主导 frontier，但 hybrids 占据 256k+ context niche。Mamba-3 的 state-tracking 改进，可能会让下一代 hybrid ratio 更低（更多 SSM、更少 attention）。

```figure
swiglu-ffn
```

## Build It / 动手构建

本课配套的 `code/main.py` 是 hybrid architecture memory calculator。它把 layer mix 变成可计算对象：给定 attention-to-Mamba ratio、layer count、hidden size、context length、attention heads 和 state size，计算 attention-only KV cache、SSM state memory，以及 pure-Transformer baseline。

动手时先复现 Jamba-1 的 1:7 配比，再改成 1:3 与 1:15。你会看到 memory curve 随 attention layers 数量近似线性变化，而 SSM state 基本是常数项。这比只看论文表格更容易形成直觉。

## Use It / 使用它

`code/main.py` 是 hybrid architectures 的 memory calculator。给定 SSM-Transformer ratio 和 hidden-size / layer-count config，它会计算：

- target context 下的 KV cache。
- SSM state memory。
- 一系列 model shapes 在 context N 下的 total memory。

calculator 支持：

- Pure-Transformer baseline（KV cache 随 N 增长）。
- Jamba-style 1:7 hybrid。
- Pure-SSM（完全没有 KV cache）。

已发布 shapes 的数字直接来自 Jamba-1 和 Jamba-1.5 论文；hypothetical variants 则做外推。

真实部署的集成注意事项：

- 大多数 production inference servers（vLLM、SGLang）支持 Jamba 和 Mamba。检查具体版本。
- 在 256k context 下，Jamba 的内存优势会体现在 concurrent-request throughput 上。同样 VRAM 可以容纳更多 Jamba sequences，而不是 Transformer sequences。
- Mamba-3 作为 standalone model 尚未在 production 发布，当前还是 1.5B 的 research preview。

## Ship It / 交付

本课会产出 `outputs/skill-hybrid-picker.md`。给定 workload specification（context length profile、task mix、memory budget），它会在 pure Transformer、Jamba-style hybrid 和 pure SSM 之间给出推荐，并明确说明 memory 与 quality tradeoffs。

## Exercises / 练习

1. 运行 `code/main.py`，计算 32-layer pure Transformer（hidden 4096, 32 heads）和同形状 Jamba-1 hybrid 在 256k context 下的 KV cache。验证 AI21 论文声称的约 8x memory reduction。

2. 修改 calculator，建模 1:3 hybrid（4 Mamba : 1 Attention）和 1:15 hybrid（14 Mamba : 1 Attention）。画出 KV cache vs ratio。在什么比例下，KV cache 等于 SSM state memory？

3. 阅读 Jamba paper（arXiv:2403.19887）的 Section 3。解释为什么 AI21 使用 Mamba-1 而不是更快的 Mamba-2。提示：hybrid ablation section 记录了这一点。

4. 计算 Jamba 1.5 Large（398B total, 94B active）中 MoE-every-other-layer 的 parameter overhead。把 active ratio 与 DeepSeek-V3（37B/671B）对比，并解释为什么 Jamba 架构会把 active ratio 推高。

5. 阅读 Mamba-3 paper（arXiv:2603.15569）的 Section 3。用三句话解释为什么 complex-valued state update 等价于 data-dependent rotary embedding。把答案联系到 Phase 7 · Lesson 04 的 RoPE 推导。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| State space model (SSM) | “Recurrence with a fixed state” | 带 learned recurrence `h_t = A h_{t-1} + B x_t` 的层；每 token 常数内存 |
| Selective SSM | “Mamba's trick” | data-dependent A、B、C parameters，让模型在线性时间下获得类似 gating 的选择性 |
| Attention-to-Mamba ratio | “How many attention layers” | 在 Jamba 中，`l = 8` 表示每 7 个 Mamba layers 配 1 个 attention layer |
| Jamba block | “The 8-layer group” | 一个 attention 加七个 Mamba，并在交替位置使用 MoE |
| SSM state | “The hidden buffer” | 替代 Mamba layers 的 KV cache 的 fixed-size per-layer state |
| 256k context | “Jamba's flagship number” | Jamba-1 能放进单张 80GB GPU 的 sequence length；pure Transformer 在该长度下不行 |
| Mamba-3 | “2026 pure SSM” | 当前最佳 pure-SSM architecture，包含 complex state + MIMO；hybrids 会围绕它重建 baseline |
| MIMO | “Multi-input multi-output” | Mamba-3 innovation，使用 matrix-valued projections 而不是 scalar per-feature |
| Exponential-trapezoidal discretization | “Mamba-3's recurrence” | 更有表达力的 recurrence，包含 Mamba-2 的 Euler-method discretization |
| Hybrid architecture | “Mix attention and SSM” | 任何交错 Transformer 和 SSM layers 的模型；Jamba 是 production archetype |

## Further Reading / 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887)：原始 Jamba paper，包含 ratio ablations 和 256k context claim
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570)：扩展后的家族，398B/94B 和 12B/52B public releases
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752)：Jamba 构建其上的 selective SSM paper
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060)：简化后的 structured-state-space successor
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569)：complex-valued state、MIMO、2026 年 pure-SSM frontier
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396)：S4 paper，LLM 中 SSM genealogy 的起点
