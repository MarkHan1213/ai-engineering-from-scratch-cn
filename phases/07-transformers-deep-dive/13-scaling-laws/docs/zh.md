# Scaling Laws / Scaling Laws：缩放定律

> 2020 年 Kaplan 论文说：model 越大，loss 越低。2022 年 Hoffmann 论文说：你们训练得不够。Compute 会落入两个桶：parameters 和 tokens，而分配比例并不直观。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 理解 training compute 如何在 parameters `N` 与 tokens `D` 之间分配
- 使用 Chinchilla/Hoffmann loss equation 估算 compute-optimal `(N, D)`
- 解释为什么 2026 年会为了降低 inference cost 而 over-train 小模型
- 区分 continuous loss scaling 与 benchmark 上的 apparent emergence

## The Problem / 问题

当你有 C FLOPs 的 training compute，并且想训练最好的 model 时，你面对两个旋钮：

1. **How many parameters (N)? / 多少参数量？** 模型越大，capacity 越高。
2. **How many training tokens (D)? / 多少训练 token？** 数据越多，capacity 使用得越充分。

FLOPs 近似按 `6 × N × D` 增长。你可以提高 N、降低 D，也可以提高 D、降低 N。哪种更好？

2022 年之前，答案是 “push N hard”。GPT-3（2020）是 175B parameters，在约 300B tokens 上训练。比例约 1.7 tokens per parameter。Kaplan scaling laws 支持这个结论。

Hoffmann et al.（2022）训练了一组名为 Chinchilla 的小模型，发现另一件事：optimal ratio 更接近 **20 tokens per parameter**。GPT-3 undertrained 了 10×。Chinchilla（70B params、1.4T tokens）以 2.5× 更低 inference cost，在每个 benchmark 上击败 GPT-3（175B、300B tokens）。

2026 年是 Chinchilla 的世界，但有一个重要 twist。Llama 3 8B 在 15 trillion tokens 上训练，比例是 1,875 tokens per parameter。比 Chinchilla-optimal 多 94 倍。对于会被大规模使用的模型，inference cost 比 training cost 更重要，所以为了更小的 deployable footprint 而 over-training（超过 Chinchilla）是 2026 年默认做法。

## The Concept / 概念

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### The Hoffmann law / Hoffmann law

Chinchilla 论文中，loss 服从：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = parameters（non-embedding）。
- `D` = training tokens。
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，irreducible loss ceiling。
- `A ≈ 406`，`B ≈ 411`。

随着 scale 增长，两项会彼此 trade off。在 fixed compute（C = 6ND）下对 `N` 求导并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Compute-optimal：20 tokens per parameter。

### Why over-training anyway / 为什么仍然 over-train

Chinchilla-optimal 最小化的是每 training FLOP 的 training loss。但 training cost 只付一次；inference cost 会一直付。

对于每月服务一万亿 tokens 的 chatbot，inference 会主导总成本。Llama 的做法是：train smaller, longer。15T tokens 上的 8B model 已经深度 inference-optimized：

- 适合 consumer GPUs。
- Latency 只是 70B Chinchilla-optimal 的一小部分。
- 对多数任务质量已经足够接近。

DeepMind 2024 年论文（"Over-training is the new optimal"）形式化了这一点。对于 inference-dominated workloads，正确比例更接近 100–500 tokens per parameter，取决于 serving volume。

### Emergence vs smoothness / 涌现与平滑性

有一种说法认为：某些能力（arithmetic、multi-step reasoning、chain-of-thought following）会在某个 scale 突然 “emerge”。

Schaeffer et al.（2023）认为这是 measurement artifact：emergent metrics 使用 discontinuous scoring（exact match、accuracy at threshold），隐藏了 logits 底层的 smooth improvement。Continuous metrics（cross-entropy）显示的是 smooth curves。

2026 年的共识是：通过 continuous loss 做预测是可靠的。Benchmark jumps 常常是 scorer artifacts。预算规划应依赖 continuous metrics。

### The 2026 picture / 2026 年图景

Scaling laws 仍然有效，但：

| Factor | Changed how |
|--------|-------------|
| Data quality | Curating "good" tokens (Phi-style) shifts curves by >2× effective compute |
| MoE | Total params decouple from active FLOPs; scaling laws per-active-FLOP |
| Post-training | Some capabilities (instruction following, code) shift with SFT+RLHF more than pretraining |
| Multimodality | Image + text tokens scale together; separate curves per modality |
| Synthetic data | Models generate training data; effective compute can compound |

Muon optimizer（Kimi Moonlight, 2024）展示了相比 AdamW，在 matched data 下约 2× effective-compute gain。一些 2026 training runs 默认使用 Muon。它改变 scaling law 的 absolute constant，不改变 shape。

```figure
scaling-laws
```

## Build It / 动手构建

见 `code/main.py`。我们实现 Chinchilla loss equation，并在多个 compute budgets 下求解 compute-optimal `(N, D)`。

### Step 1: Chinchilla loss / 第 1 步：Chinchilla loss

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在 fixed `C = 6ND` 下，把 `L` 作为 `(N, D)` contour 画出来，找到 minimum。

### Step 2: compute-optimal frontier / 第 2 步：compute-optimal frontier

对 `1e17` 到 `1e25` FLOPs 的 compute budgets，找到满足 `6ND = C` 时 loss 最小的 `(N, D)`。验证比例 `D/N ≈ 20`。

### Step 3: over-training cost / 第 3 步：over-training cost

计算训练一个 10× smaller model（optimal N 的 1/10、optimal D 的 10×）会多付多少 loss。报告换来的 inference FLOP savings（与 N 成正比）。

### Step 4: compare to real models / 第 4 步：与真实 models 对比

填入 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（active params）等已知 `(N, D)` pairs，比较 predicted loss 与 reported loss。

## Use It / 应用它

你大概率不会亲自训练 frontier model。但 scaling laws 会告诉你：

1. **Whether your fine-tune has enough data / 你的 fine-tune 数据是否够。** 如果 task-specific data 低于 base model 的 20 tokens per param，要预期它会在某个 loss floor 饱和。
2. **Whether to pick a bigger base model / 是否选更大的 base model。** 如果预算大多花在 inference，优先选更小、训练更久的模型。
3. **Where the returns diminish / 收益何处变小。** 超过 Chinchilla-optimal 1000× 后，log-loss 变化会接近噪声。

**The research trajectory in 2026 / 2026 年研究方向：**

- **Data-constrained regime.** Web 上 high-quality tokens 数量有限（过滤后英文约 5–10 trillion）。Frontier pretraining 正接近这个上限。Synthetic data、multilingual、multimodal 和 RLHF-scaled fine-tuning 是下一批 levers。
- **Compute-multiplier tricks.** Muon optimizer、MoE、更好的 data curation；每项都改变 absolute constants，不改变 asymptote。
- **Scaling laws for RL.** 仍是 open question。早期证据显示 RL samples 上有 power-law，但 exponents 与 pretraining 很不同。

## Ship It / 交付它

见 `outputs/skill-training-budget-estimator.md`。这个 skill 会在给定 compute budget、deployment constraints 和 target loss 时，为新 training run 选择 `(N, D, hours, GPU)`。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。打印 compute budgets `1e20`、`1e22`、`1e24` 下的 Chinchilla-optimal `(N, D)`。与真实 model table 对比。
2. **Medium / 中等。** 实现 Hoffmann loss-as-function-of-compute curve。对 compute-optimal frontier 画 loss vs `log10(C)`。识别 scaling law 预测下一次 cross-entropy 降低 0.1 需要 `>10^28` FLOPs 的位置。
3. **Hard / 困难。** 在同一 dataset 上训练 5 个 tiny models（100K 到 10M params），拟合你自己的 scaling law。估计 `α` 和 `E`。你的 exponents 与公开论文匹配得如何？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Parameters (N) | “Model size” | Non-embedding weight count；决定 capacity。 |
| Tokens (D) | “Training data” | 训练中见过的 token 数量；决定 parameters 被使用得多充分。 |
| Compute (C) | “FLOPs spent” | 对标准 transformer 约为 `6 × N × D`。 |
| Chinchilla-optimal | “D/N ≈ 20” | 最小化 pretraining 每 FLOP loss 的比例。 |
| Over-training | “Past Chinchilla” | 多花 training FLOPs 来省 inference FLOPs；D/N >> 20。 |
| Irreducible loss | “The floor” | Scaling law 中的 `E` 项；数据本身的 entropy。 |
| Emergent capability | “Sudden jumps at scale” | 常常是 scorer artifact；continuous loss 是平滑的。 |
| Effective compute | “Training-efficiency multiplier” | 更好的 data / optimizer / architecture 会放大每个 FLOP 的价值。 |

## Further Reading / 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 第一篇 scaling law 论文，但 undertrained。
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla。
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — emergence 作为 measurement artifact。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — 为什么 Llama 的 over-training 对其 workload 是正确的。
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2× compute multiplier。
