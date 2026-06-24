# Attention Mechanism — The Breakthrough / 注意力机制：突破点

> Decoder 不再盯着压缩摘要硬猜，而是开始查看整个 source。此后的所有东西，基本都是 attention 加工程化。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 Bahdanau attention 如何解除 seq2seq 的 context-vector bottleneck
- 推导 query、key、value、score、weights、context vector 的 shape
- 从零实现 additive attention、dot attention 和 general attention
- 识别 attention weights 被误当作解释的风险，并理解它如何通向 transformer

## The Problem / 问题

Lesson 09 以一个可测的失败收尾：在 toy copy task 上训练的 GRU encoder-decoder，长度 5 时准确率 89%，长度 80 时接近随机。原因是结构性的，不是训练 bug：encoder 捕捉到的每一比特信息都必须塞进一个固定大小 hidden state，而 decoder 看不到其他东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行修复。不要只把最终 encoder state 给 decoder，而是保留每个 encoder state。每个 decoder step 都计算 encoder states 的加权平均，其中权重表示“decoder 此刻需要看 encoder 位置 `i` 的程度”。这个加权平均就是 context，而且每个 decoder step 都会变化。

核心想法就是这样。Transformer 扩展了它。Self-attention 把它应用到单个序列上。Multi-head attention 并行运行它。但 2014 年的版本已经打破了瓶颈；一旦你理解它，转向 transformer 就主要是工程问题，而不是概念问题。

## The Concept / 概念

![Bahdanau attention：decoder 查询所有 encoder states](../assets/attention.svg)

在每个 decoder step `t`：

1. 使用上一个 decoder hidden state `s_{t-1}` 作为 **query**。
2. 把它与每个 encoder hidden state `h_1, ..., h_T` 打分。每个 encoder position 一个 scalar。
3. 对 scores 做 softmax，得到和为 1 的 attention weights `α_{t,1}, ..., α_{t,T}`。
4. Context vector `c_t = Σ α_{t,i} * h_i`。也就是 encoder states 的加权平均。
5. Decoder 接收 `c_t` 和上一个 output token，生成下一个 token。

加权平均是关键。当 decoder 需要把 "Je" 翻译成 "I" 时，它会给 "Je" 上方的 encoder state 高权重，其他位置低权重。当它需要 "not" 时，会给 "pas" 高权重。Context vector 每一步都会重塑。

## Shapes (the thing that bites everyone) / Shapes：最容易踩坑的部分

第一次写 attention，几乎每个人都会在这里出错。慢慢读。

| Thing / 对象 | Shape | Notes / 说明 |
|-------|-------|-------|
| Encoder hidden states `H` | `(T_enc, d_h)` | 如果是 BiLSTM，`d_h = 2 * d_hidden` |
| Decoder hidden state `s_{t-1}` | `(d_s,)` | 一个 vector |
| Attention score `e_{t,i}` | scalar | 每个 encoder position 一个 |
| Attention weight `α_{t,i}` | scalar | 对所有 `i` softmax 之后得到 |
| Context vector `c_t` | `(d_h,)` | 与 encoder state shape 相同 |

**Bahdanau (additive) score.** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` shape 是 `(d_s,)`，`h_i` shape 是 `(d_h,)`。
- `W_a` shape 是 `(d_attn, d_s)`。`U_a` shape 是 `(d_attn, d_h)`。
- tanh 内部两者相加后的 shape 是 `(d_attn,)`。
- `v_α` shape 是 `(d_attn,)`。与 `v_α` 做 inner product 会压成 scalar。**这就是 `v_α` 的作用。** 它不是魔法，只是把 attention-dim vector 投影成 scalar score。

**Luong (multiplicative) score.** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬约束。如果 encoder 是 bidirectional，就跳过。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` shape 是 `(d_s, d_h)`。它移除了等维约束。
- `concat`：本质上是 Bahdanau 形式。自从前两者更便宜后就很少用。

**一个值得点名的 Bahdanau / Luong 坑。** Bahdanau 使用 `s_{t-1}`（生成当前词 *之前* 的 decoder state）。Luong 使用 `s_t`（生成当前词 *之后* 的 state）。混用会产生很难调试的细微错误梯度。选一篇论文，并坚持它的约定。

```figure
attention-heatmap
```

## Build It / 动手构建

### Step 1: additive (Bahdanau) attention / 第 1 步：additive（Bahdanau）attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上表检查 shape。`encoder_states` shape 是 `(T_enc, d_h)`。`projected_enc` shape 是 `(T_enc, d_attn)`。`projected_dec` shape 是 `(d_attn,)` 并会 broadcast。`combined` shape 是 `(T_enc, d_attn)`。`scores` shape 是 `(T_enc,)`。`weights` shape 是 `(T_enc,)`。`context` shape 是 `(d_h,)`。可以交付。

### Step 2: Luong dot and general / 第 2 步：Luong dot 与 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每个三行。这就是 Luong 论文能落地的原因。多数任务上准确率相近，代码少很多。

### Step 3: a worked numerical example / 第 3 步：一个数值例子

给定三个 encoder states（大致表示 "cat"、"sat"、"mat"）和一个最接近第一个 state 的 decoder state，attention distribution 会集中在位置 0。如果 decoder state 改到更接近第三个 encoder state，attention 就会移动到位置 2。Context vector 会随之变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后把 decoder state 移得更接近第三个 encoder state，观察 weights 如何移动。Attention 就是显式 alignment。

### Step 4: why this is the bridge to transformers / 第 4 步：为什么它是通往 transformer 的桥

把上面的语言翻译成 Q/K/V：

- **Query** = decoder state `s_{t-1}`
- **Key** = encoder states（用于打分的对象）
- **Value** = encoder states（用于加权求和的对象）

在经典 attention 中，keys 和 values 是同一个东西。Self-attention 会把它们分开：你可以让一个序列查询自身，并为 K 和 V 使用不同 learned projections。Multi-head attention 会用不同 learned projections 并行运行。Transformer 把整个阶段堆叠多次，并去掉 RNN。

数学是一样的，shape 是一样的。从 Bahdanau attention 跳到 scaled dot-product attention，主要是记号变化。

## Use It / 应用它

PyTorch 和 TensorFlow 直接提供 attention。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 transformer attention layer。Query batch 有 5 个位置，key/value batch 有 10 个位置，每个 128 维，8 个 heads。`output` 是新的 context-augmented queries。`weights` 是可以可视化的 5x10 alignment matrix。

### When classical attention still matters / 经典 attention 仍然重要的场景

- 教学。Single-head、single-layer、RNN-based 版本能让每个概念都可见。
- Transformer 放不进去的 on-device sequence tasks。
- 2014-2017 年的任何论文。不知道 Bahdanau 的约定就会读错。
- MT 中的细粒度 alignment analysis。即便在 transformer models 中，raw attention weights 仍是解释性工具；要读懂它们，必须知道它们是什么。

### The attention-weight-as-explanation trap / 把 attention weights 当解释的陷阱

Attention weights 看起来很可解释。它们是在位置上求和为 1 的权重；可以画图；高权重看起来像“看了这里”。审稿人喜欢它。

它们没有看起来那么可解释。Jain and Wallace (2019) 证明，在某些任务上，attention distributions 可以被打乱，甚至替换成任意替代分布，而模型预测不变。不要把 attention weights 当成 reasoning 证据，除非你做了 ablation 或 counterfactual check。

## Ship It / 交付它

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## Exercises / 练习

1. **Easy / 简单。** 实现 `softmax` masking，让 encoder 中的 padding tokens attention weight 为零。在包含可变长度序列的 batch 上测试。
2. **Medium / 中等。** 给 Luong `general` 形式增加 multi-head attention。把 `d_h` 拆成 `n_heads` 组，每个 head 单独运行 attention，再 concatenate。验证 single-head 情况与你之前的实现一致。
3. **Hard / 困难。** 在 lesson 09 的 toy copy task 上训练带 Bahdanau attention 的 GRU encoder-decoder。绘制 accuracy vs sequence length。与 no-attention baseline 对比。你应该会看到长度越长差距越大，从而确认 attention 解除了瓶颈。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Attention | 看某些东西 | 对 value sequence 做加权平均，权重由 query-key similarity 计算。 |
| Query, Key, Value | QKV | 三种投影：Q 发问，K 用来匹配，V 是返回的内容。 |
| Additive attention | Bahdanau | Feed-forward score：`v^T tanh(W q + U k)`。 |
| Multiplicative attention | Luong dot / general | Score 是 `q^T k` 或 `q^T W k`。更便宜，多数任务准确率相同。 |
| Alignment matrix | 漂亮图 | Attention weights 组成的 `(T_dec, T_enc)` 网格。用它查看模型关注了哪里。 |

## Further Reading / 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 原论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种 score variants 及其对比。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 解释性 caveat。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 带 PyTorch 的可运行 walkthrough。
