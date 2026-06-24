# Self-Attention from Scratch / 从零实现 Self-Attention

> Attention 像一张 lookup table：每个词都在问“谁对我重要？”然后通过学习得到答案。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 3 (Deep Learning Core), Phase 5 Lesson 10 (Sequence-to-Sequence)
**Time / 时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 只用 NumPy 从零实现 scaled dot-product self-attention，包括 query/key/value projections 和 softmax-weighted sum
- 构建一个 multi-head attention layer，完成 heads 拆分、并行 attention 和结果拼接
- 追踪 attention matrix 如何捕捉 token 关系，并解释为什么除以 sqrt(d_k) 能避免 softmax saturation
- 使用 causal masking，把 bidirectional attention 转成 autoregressive（decoder-style）attention

## The Problem / 问题

RNN 一次处理一个 token。当你到达 token 50 时，token 1 的信息已经穿过 50 次压缩步骤。长程依赖被挤进固定大小 hidden state，这个瓶颈不是加多少 LSTM gating 就能完全解决的。

2014 年 Bahdanau attention 论文给出修复：让 decoder 回看每个 encoder position，并决定哪些 position 对当前 step 重要。但它仍然挂在 RNN 上。2017 年 "Attention Is All You Need" 提出一个更尖锐的问题：如果 attention 是*唯一*机制呢？没有 recurrence，没有 convolution，只有 attention。

Self-attention 让 sequence 中每个位置在一次并行步骤中 attend 到其他所有位置。这就是 transformer 快、可扩展并最终占据主导地位的原因。

## The Concept / 概念

### The Database Lookup Analogy / 数据库查询类比

把 attention 想成一次 soft database lookup：

```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

每个 token 都生成三个 vectors：
- **Query (Q)**： “What am I looking for?”
- **Key (K)**： “What do I contain?”
- **Value (V)**： “What information do I provide if selected?”

一个 query 与所有 keys 的 dot product 会产生 attention scores。高分表示“这个 key 匹配我的 query”。这些 scores 再用来加权 values。输出就是 values 的加权和。

### Q, K, V Computation / Q、K、V 计算

每个 token embedding 会通过三个 learned weight matrices 做投影：

```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

对单个 token 来看：

```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### The Attention Matrix / Attention Matrix

一旦所有 tokens 都有了 Q、K、V，attention scores 就形成一个 matrix：

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

一次看一个 query 如何扫过所有 keys：每一行都给每个 token 打分，softmax 把 scores 变成 weights，context vector 则是 values 的加权混合。

```figure
attention-matrix
```

### Why Scale? / 为什么要 Scale？

Dot products 会随 dimension dk 增大而变大。如果 dk = 64，dot products 可能到几十，直接把 softmax 推到梯度消失的区域。修复方式：除以 sqrt(dk)。

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

这样 values 会保持在 softmax 能产生有效 gradients 的范围内。

### Softmax Turns Scores into Weights / Softmax 把 Scores 变成 Weights

Softmax 会把 raw scores 转成每一行上的 probability distribution：

```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

现在每个 token 都有一组 weights，表示它应该多大程度上 attend 到其他 token。

### Weighted Sum of Values / Values 加权求和

每个 token 的最终输出都是所有 value vectors 的加权和：

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### Full Pipeline / 完整 Pipeline

```mermaid
flowchart LR
  X["X (input)"] --> Q["Q = X · Wq"]
  X --> K["K = X · Wk"]
  X --> V["V = X · Wv"]
  Q --> S["Q · Kᵀ / √dk"]
  K --> S
  S --> SM["softmax"]
  SM --> WS["weighted sum"]
  V --> WS
  WS --> O["output"]
```

一行公式：

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

```figure
softmax-attention-scaling
```

## Build It / 动手构建

### Step 1: Softmax from scratch / 第 1 步：从零实现 Softmax

Softmax 把 raw logits 转成 probabilities。为了 numerical stability，要先减去最大值。

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### Step 2: Scaled dot-product attention / 第 2 步：Scaled dot-product attention

核心函数。接收 Q、K、V matrices，并返回 attention output 与 weight matrix。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### Step 3: Self-attention class with learned projections / 第 3 步：带 learned projections 的 Self-attention class

一个完整 self-attention module，包含 Wq、Wk、Wv weight matrices，并用类似 Xavier 的 scaling 初始化。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### Step 4: Run it on a sentence / 第 4 步：在一个句子上运行

为一句话创建 fake embeddings，并观察 attention weights。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### Step 5: Visualize attention with ASCII heatmap / 第 5 步：用 ASCII heatmap 可视化 attention

把 attention weights 映射成字符，快速看出图案。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## Use It / 应用它

PyTorch 的 `nn.MultiheadAttention` 做的正是我们刚刚构建的东西，外加 multi-head splitting 和 output projection：

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

关键区别是：multi-head attention 会并行运行多个 attention functions，每个都有自己的 Q、K、V projections，大小为 dk = d_model / n_heads，然后再拼接结果。这让模型能同时 attend 到不同关系类型。

## Ship It / 交付它

本课产出：
- `outputs/prompt-attention-explainer.md` — 一个用 database lookup 类比解释 attention 的 prompt

## Exercises / 练习

1. 修改 `scaled_dot_product_attention`，让它接受 optional mask matrix，在 softmax 之前把某些位置设为 negative infinity（这就是 causal/decoder masking 的做法）。
2. 从零实现 multi-head attention：把 Q、K、V 拆成 `n_heads` chunks，在每个 head 上运行 attention，concatenate，再通过最终 weight matrix Wo 做投影。
3. 取两句长度相同但内容不同的句子，送入同一个 SelfAttention instance，比较 attention patterns。哪些变化了？哪些保持不变？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Query (Q) | “The question vector” | 输入的 learned projection，表示当前 token 正在寻找什么信息。 |
| Key (K) | “The label vector” | learned projection，表示当前 token 包含什么信息，并用于和 queries 匹配。 |
| Value (V) | “The content vector” | 携带实际信息的 learned projection，会根据 attention scores 被聚合。 |
| Scaled dot-product attention | “The attention formula” | softmax(QK^T / sqrt(dk)) @ V；scaling 防止高维下 softmax saturation。 |
| Self-attention | “The token looks at itself and others” | Q、K、V 都来自同一个 sequence，让每个位置 attend 到其他所有位置。 |
| Attention weights | “How much focus” | 对 scaled dot products 做 softmax 后得到的 position probability distribution。 |
| Multi-head attention | “Parallel attention” | 用不同 projections 并行运行多个 attention functions，再拼接结果以获得更丰富的表示。 |

## Further Reading / 延伸阅读

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — 原始 transformer 论文。
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) — 最好的完整架构可视化 walkthrough。
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 逐行 PyTorch 实现和解释。
