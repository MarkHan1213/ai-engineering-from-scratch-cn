# Multi-Head Attention / 多头注意力

> 一个 attention head 一次学习一种关系。八个 heads 就能学八种。Heads 很便宜，多用一些。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 02 (Self-Attention from Scratch)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释为什么 single-head attention 会把多种关系压进同一个 softmax distribution
- 实现 multi-head attention 的 split、parallel attention、concatenate 和 output projection
- 理解 MHA、MQA、GQA、MLA 在 Q heads 与 K/V heads 上的差异
- 根据模型尺寸、context length 和部署目标选择合理的 head count 与 kv-head strategy

## The Problem / 问题

单个 self-attention head 只计算一个 attention matrix。这个 matrix 捕捉一种关系，通常是训练信号下最能降低 loss 的那一种。如果你的数据里 subject-verb agreement、co-reference、long-range discourse 和 syntactic chunking 混在一起，单个 head 会把它们糊进同一个 soft-max distribution，损失一半信号。

2017 年 Vaswani 论文给出的修复是：并行运行多个 attention functions，每个都有自己的 Q、K、V projections，然后 concatenate 输出。每个 head 在维度 `d_model / n_heads` 的较小 subspace 中工作。总参数量保持不变，表达能力上升。

Multi-head attention 是 2026 年每个 transformer 的默认配置。真正争论的只是 *how many* heads，以及 keys 和 values 是否共享 projections（Grouped-Query Attention、Multi-Query Attention、Multi-head Latent Attention）。

## The Concept / 概念

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split / 拆分。** 取 shape 为 `(N, d_model)` 的 `X`。投影得到 Q、K、V，shape 都是 `(N, d_model)`。Reshape 成 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。再 transpose 成 `(n_heads, N, d_head)`。

**Attend in parallel / 并行 attention。** 在每个 head 内运行 scaled dot-product attention。每个 head 产生 `(N, d_head)`。这些 heads 在 embedding 的不同 subspaces 上工作，在 attention computation 本身期间互不通信。

**Concatenate and project / 拼接并投影。** 把 heads stack 回 `(N, d_model)`，再乘以 learned output matrix `W_o`，shape 为 `(d_model, d_model)`。`W_o` 是 heads 开始混合的位置。

**Why it works / 为什么有效。** 每个 head 可以 specialization，不需要和其他 heads 争夺同一份 representation budget。2019–2024 的 probing studies 展示过不同 head roles：positional heads、attend 到 previous token 的 head、copy heads、named-entity heads、induction heads（它们支撑 in-context learning）。

**The 2026 lineage of variations / 2026 年常见变体谱系：**

| Variant | Q heads | K/V heads | Used by |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA 是现代默认选择，因为它能按 `N/G` 的比例削减 KV-cache memory，同时几乎保持完整质量。MLA 更进一步，把 K/V 压到 latent space，再在 compute time 投影回来，代价是 FLOPs，收益是大幅节省 memory。

```figure
multihead-split
```

## Build It / 动手构建

### Step 1: split heads from the single-head attention we already have / 第 1 步：从已有 single-head attention 拆出 heads

取 Lesson 02 的 `SelfAttention`，在外面加一对 split/concat。`code/main.py` 里有 numpy 实现，核心逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape 和一次 transpose，没有 loop。这正是 PyTorch 在 `nn.MultiheadAttention` 下面做的事。

### Step 2: run scaled-dot-product attention per head / 第 2 步：在每个 head 上运行 scaled-dot-product attention

每个 head 拿到自己的 Q、K、V slice。Attention 变成 batched matmul：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上，`Qh @ Kh.transpose(...)` 是一个 `bmm`。GPU 看到的是一个 shape 为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的 batched matmul。增加 heads 很便宜。

### Step 3: Grouped-Query Attention variant / 第 3 步：Grouped-Query Attention 变体

只有 key 和 value projections 会变。Q 有 `n_heads` 组；K 和 V 有 `n_kv_heads < n_heads` 组，然后 repeat 到匹配 Q：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推理时这会节省 memory，因为 KV cache 里只需要保存 `n_kv_heads` 份，而不是 `n_heads` 份。Llama 3 70B 使用 64 个 query heads 和 8 个 KV heads，也就是 8× cache shrink。

### Step 4: probe what each head learned / 第 4 步：探查每个 head 学到了什么

用 4 个 heads 在短句上运行 MHA。对每个 head 打印 `(N, N)` attention matrix。即使是 random initialization，你也会看到不同 heads 选出不同结构；这部分来自信号，也部分来自 subspaces 中的 rotational symmetry。

## Use It / 应用它

PyTorch 的一行版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 中的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads? / 多少 heads 合适？** 2026 年 production models 的经验规则：

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` 几乎总是 64 或 128。它是一个 head 能“看见”多少信息的单位。低于 32 时，heads 会开始被 scaling factor `sqrt(d_head)` 牵制；高于 256 时，会失去“许多小专家”的好处。

## Ship It / 交付它

见 `outputs/skill-mha-configurator.md`。这个 skill 会根据 parameter budget、sequence length 和 deployment target，为新 transformer 推荐 head count、kv-head count 和 projection strategy。

## Exercises / 练习

1. **Easy / 简单。** 取 `code/main.py` 中的 MHA，在固定 `d_model=64` 时把 `n_heads` 从 1 改到 16。画出 tiny one-layer model 在 synthetic copy task 上的 loss。更多 heads 是有帮助、进入平台期，还是有害？
2. **Medium / 中等。** 实现 MQA（所有 query heads 共享一个 KV head）。测量相对 full MHA，parameter count 降低多少。计算 N=2048 推理时 KV-cache size 缩小多少。
3. **Hard / 困难。** 实现一个 tiny 版 Multi-head Latent Attention：把 K,V 压缩到 rank-`r` latent，把 latent 存进 KV cache，在 attention time 解压。`r` 到多少时，cache memory 会低于 full MHA 的 1/8，同时质量仍保持在 validation ppl 的 1 bit 以内？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Head | “A single attention circuit” | 一个维度为 `d_head = d_model / n_heads` 的 Q/K/V projection，并有自己的 attention matrix。 |
| d_head | “Head dimension” | 每个 head 的 hidden width；production 中几乎总是 64 或 128。 |
| Split / combine | “Reshape tricks” | attention 前后的 `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose。 |
| W_o | “Output projection” | 拼接 heads 后应用的 `(d_model, d_model)` matrix；heads 在这里混合。 |
| MQA | “One KV head” | Multi-Query Attention：单个共享 K/V projection。KV cache 最小，但会有一些质量损失。 |
| GQA | “The default since Llama 2” | `n_kv_heads < n_heads` 的 Grouped-Query Attention；repeat 到匹配 Q。 |
| MLA | “DeepSeek's trick” | Multi-head Latent Attention：K,V 压到 low-rank latent，在 attend time 解压。 |
| Induction head | “The circuit behind in-context learning” | 一对 heads，用来检测过去出现过的模式，并复制其后续内容。 |

## Further Reading / 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始 multi-head specification。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 训练后如何把 MHA 转成 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 以及它为何在 cache memory 上优于 MHA/GQA。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 从 mechanistic 角度观察 heads 实际做了什么。
