# Why Transformers — The Problems with RNNs / 为什么是 Transformer：RNN 的问题

> RNN 一次处理一个 token。Transformer 一次处理所有 token。2017 年之后，深度学习里几乎所有 scaling curve 都被这个架构选择改写了。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释为什么 RNN 的 sequential computation 会限制训练并行度和 wall-clock scaling
- 区分 operation count 与 serial depth，并说明为什么 GPU 更关心依赖深度
- 理解 self-attention 如何把 token 间交互改写成大规模矩阵运算
- 判断 2026 年何时仍应选择 RNN、SSM 或 transformer

## The Problem / 问题

2017 年之前，地球上几乎所有最强序列模型，不管是语言、翻译还是语音，都是 recurrent neural network。LSTM 和 GRU 在相当于 ImageNet 地位的翻译 benchmark 上统治了半个十年。那时大家手里几乎只有这把工具。

它们有三个致命弱点。首先是 sequential computation：你无法沿时间轴并行，因为 token `t+1` 需要 token `t` 的 hidden state。一个 1,024-token sequence 意味着 1,024 个串行步骤，而 GPU 每个周期可以做 1,000,000 次 floating-point ops。训练 wall-clock time 在硬件最擅长并行时，却随 sequence length 线性增长。

其次是 vanishing gradients。50 个 token 之前的信息已经穿过 50 层非线性压缩。Gated recurrent units（LSTM、GRU）缓解了这种挤压，但从未真正消除。长程依赖，例如 “the book I read last summer on a plane to Kyoto was…” 这种结构，仍然经常失败。

第三是 fixed-width hidden states。Encoder 在 decoder 看到任何东西之前，必须把整个 source sequence 压进一个单一 vector。source 是 5 个 token 还是 500 个 token 都一样，瓶颈 shape 不变。

2017 年论文 "Attention Is All You Need" 提出了一个激进方案：完全去掉 recurrence。让每个位置并行 attend 到其他所有位置。用一次大的 matrix multiplication 训练，而不是 1,024 次串行计算。

到 2026 年，结果已经主导所有模态：语言（GPT-5、Claude 4、Llama 4）、视觉（ViT、DINOv2、SAM 3）、音频（Whisper）、生物学（AlphaFold 3）、机器人（RT-2）。同一个 block，换不同输入。

## The Concept / 概念

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence as a bottleneck / 作为瓶颈的 recurrence。** RNN 计算 `h_t = f(h_{t-1}, x_t)`。每一步都依赖前一步。不能先算 `h_5` 再算 `h_4`。在有 10,000+ parallel cores 的现代 GPU 上，长 sequence 会让 99% 的 silicon 闲着。

**Attention as a broadcast / 作为 broadcast 的 attention。** Self-attention 为每一对 `(i, j)` 同时计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N attention matrix 通过一次 batched matmul 填满。没有一步依赖另一步。GPU 很喜欢这种形态。

**The speedup is not a constant / 加速不是一个常数。** 它是 `O(N)` serial depth 与 `O(1)` serial depth 的差异。实践中，在 N=512 且硬件相同时，transformer 每个 epoch 训练通常快 5–10×；sequence length 越长差距越大，直到撞上 attention 的 `O(N²)` memory wall（后来 Flash Attention 缓解了常数问题，见 Lesson 12）。

**What transformers cost / Transformer 的代价。** Attention memory 按 `O(N²)` 增长。2K context 没问题；128K context 就需要 sliding windows、RoPE extrapolation、Flash Attention tiling 或 linear attention variants。Recurrence 的 time 和 memory 都是 `O(N)`；transformer 用 memory 换 time，再通过 parallelism 把 time 赢回来。

**The inductive bias shift / Inductive bias 的转移。** RNN 假设 locality 和 recency。Transformer 什么都不假设，每一对位置都可能互相关注。这也是为什么 transformer 需要更多数据才能训练好，但一旦有足够数据就能扩展得更远。Chinchilla（2022）形式化了这一点：在 token 充足时，同参数量 transformer 总会击败 RNN。

## Build It / 动手构建

这里不写神经网络。我们用数值模拟核心瓶颈，让你在自己的 laptop 上直观看到差距。

### Step 1: measure serial depth / 第 1 步：测量 serial depth

见 `code/main.py`。我们构造两个函数。一个把 sequence 编码成一条 addition chain（串行，像 RNN）。另一个把它编码成 parallel reduction（broadcast，像 attention）。数学相同，dependency graph 不同。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

我们会在最高 100,000 个元素的 sequences 上给两者计时。RNN 版本是 O(N)，而且走单条 CPU pipeline。即使在 pure Python 里，length ≥ 1,000 时 attention-style reduction 也会超过它，因为 Python 的 `sum()` 在 C 中实现，不会让每一步都付 interpreter overhead。

### Step 2: count theoretical operations / 第 2 步：计算理论操作量

两个算法都做 N 次 add。差别在于 *dependency depth*：下一批操作开始前，必须串行完成多少步。RNN depth = N。Attention depth 在 tree reduction 下是 log(N)，在 parallel scan 下是 1。决定 GPU time 的是 depth，不是 op count。

### Step 3: empirical scaling on long sequences / 第 3 步：观察长 sequence 上的经验 scaling

我们会打印一张 timing table，让 O(N) 差距变得可见。在 2026 年的 Mac laptop 上，1,000 以下的 sequence 太快，几乎测不准。100,000 的 sequence 会呈现清晰 linear scan。把它扩展到 16,384-token transformer，再类比 12-layer LSTM，你就能看到为什么训练 wall-clock 在 2016 年是 blocker。

## Use It / 应用它

2026 年什么时候仍然选 RNN：

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| Streaming inference, one token at a time, constant memory | RNN or state-space model (Mamba, RWKV) |
| Very long sequences (>1M tokens) where attention memory explodes | Linear attention, Mamba 2, Hyena |
| Edge device with no matmul accelerator | Depthwise-separable RNN still wins on FLOPs/watt |
| Anything else (training, batched inference, context up to 128K) | Transformer |

像 Mamba 这样的 state-space models（SSMs）本质上是带结构化参数化的 RNN，拿到了两边的好处：`O(N)` scan memory，以及通过 selective scan 做 parallel training。它们能恢复 transformer 质量的 90%，并有更好的 long-context scaling。2026 年多数 frontier labs 都在训练 hybrid SSM+transformer models（例如 Jamba、Samba）。Recurrence 没死，它成了一个组件。

## Ship It / 交付它

见 `outputs/skill-architecture-picker.md`。这个 skill 会根据 length、throughput 和 training-budget constraints，为新的 sequence problem 选择架构。对于超过 1B tokens 的训练 run，它必须拒绝推荐 pure RNN，除非同时说明 trade-off。

## Exercises / 练习

1. **Easy / 简单。** 取 `code/main.py` 里的 `rnn_style`，把 scalar hidden state 替换成 length-64 vector hidden states。重新测量。serial overhead 会随 hidden-state dimension 增长多少？
2. **Medium / 中等。** 用 pure Python 实现 parallel prefix-sum（Hillis-Steele scan）。验证它在 length 1024 上与 serial scan 数值一致，并计算 depth。
3. **Hard / 困难。** 把 attention-style reduction 移植到 GPU 上的 PyTorch。sequence length 从 64 sweep 到 65,536，分别计时并画图解释 curve shape。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Recurrence | “RNNs are sequential” | step `t` 依赖 step `t-1` 的计算，因此时间轴上必须串行执行。 |
| Serial depth | “How deep the graph is” | 最长依赖操作链；即使有无限硬件也会限制 wall-clock。 |
| Attention | “Let tokens look at each other” | 加权求和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 和 j 的相似度分数。 |
| Context window | “How much the model sees” | attention layer 可接收的 position 数；quadratic memory cost 就在这里增长。 |
| Inductive bias | “Assumptions baked into the architecture” | 对数据形态的先验；CNN 假设 translation invariance，RNN 假设 recency。 |
| State-space model | “RNN with algebra behind it” | 通过结构化 state-space matrices 参数化的 recurrence，可支持 parallel training。 |
| Quadratic bottleneck | “Why context costs so much” | Attention memory = sequence length 上的 `O(N²)`；Flash Attention 隐藏常数，不改变 scaling。 |

## Further Reading / 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 让主流 NLP 放弃 recurrence 的论文。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — attention 的诞生地，当时它仍挂在 RNN 上。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — 原始 LSTM 论文，作为历史记录。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — transformer 时代的现代 recurrent 答案。
