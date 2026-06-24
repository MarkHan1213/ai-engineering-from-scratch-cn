# The Full Transformer — Encoder + Decoder / 完整 Transformer：Encoder + Decoder

> Attention 是主角。其余部分，包括 residuals、normalization、feed-forward、cross-attention，都是让它能堆深的脚手架。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 拆解完整 transformer block 的 embedding、attention、FFN、residual、normalization 和 cross-attention
- 区分 encoder block、decoder block 与 encoder-decoder stack 的信息流
- 解释 pre-norm、RMSNorm、SwiGLU、RoPE、GQA 为什么成为 2026 年默认组件
- 估算 transformer block 的参数量，并据此判断实现是否合理

## The Problem / 问题

单个 attention layer 是 feature extractor，不是完整 model。每层一次 matmul 没有足够 capacity 表示语言。你需要 depth，而 depth 如果没有正确的 plumbing 就会崩。

2017 年 Vaswani 论文把六个设计决策打包起来，把一个 attention layer 变成可堆叠的 block。此后的每个 transformer，不管是 encoder-only（BERT）、decoder-only（GPT），还是 encoder-decoder（T5），都继承了同一套 skeleton。到 2026 年，这些 blocks 已经被细化为 RMSNorm、SwiGLU、pre-norm、RoPE，但骨架没变。

这一课讲的就是骨架。接下来的课会把它 specialize：06 讲 encoders，07 讲 decoders，08 讲 encoder-decoder。

## The Concept / 概念

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### The six pieces / 六个部件

1. **Embedding + positional signal / Embedding + 位置信号。** Tokens → vectors。Position 通过 RoPE（现代）或 sinusoidal（经典）注入。
2. **Self-attention / 自注意力。** 每个位置 attend 到其他所有位置。Decoder 中会 masked。
3. **Feed-forward network (FFN) / 前馈网络。** Position-wise two-layer MLP：`W_2 · activation(W_1 · x)`。默认 expansion ratio 为 4×。
4. **Residual connection / 残差连接。** `x + sublayer(x)`。没有它，超过约 6 层后 gradients 会消失。
5. **Layer normalization / 归一化。** `LayerNorm` 或 `RMSNorm`（现代）。稳定 residual stream。
6. **Cross-attention (decoder only) / 交叉注意力（仅 decoder）。** Queries 来自 decoder，keys 和 values 来自 encoder output。

观察一个 vector 穿过一个 block：attention 跨 positions 混合信息，residual 把它向前带，FFN 转换它，norm 保持 stream 稳定。

```figure
transformer-block
```

### Encoder block (used by BERT, T5 encoder) / Encoder block（BERT、T5 encoder 使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder 是 bidirectional。没有 masking。所有 positions 都能看到所有 positions。

### Decoder block (used by GPT, T5 decoder) / Decoder block（GPT、T5 decoder 使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder 每个 block 有三个 sublayers。中间的 cross-attention 是信息从 encoder 流向 decoder 的唯一位置。在 pure decoder-only architecture（GPT）里，cross-attention 会被省略，只保留 masked self-attention + FFN。

### Pre-norm vs post-norm / Pre-norm 与 post-norm

原论文讨论的是 `x + sublayer(LN(x))` 与 `LN(x + sublayer(x))`。Post-norm 在 2019 年左右失宠，因为没有细致 warmup 时很难训练深层网络。Pre-norm（在 sublayer *之前* 做 `LN`）是 2026 年默认选择：Llama、Qwen、GPT-3+、Mistral 都使用它。

### The 2026 modernized block / 2026 年的现代化 block

Vaswani 2017 使用 LayerNorm + ReLU。现代 stack 替换了两者。Production blocks 实际长这样：

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm 去掉了 LayerNorm 的 mean-centering（少一次 subtraction），节省 compute，经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 论文中 consistently 优于 ReLU/GELU FFN，LM ppl 大约好 0.5 point。

### Parameter count / 参数量

对一个 `d_model = d` 且 FFN expansion 为 `r` 的 block：

- MHA：`4 · d²`（Q、K、V、O projections）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- Norms：可忽略

当 `d = 4096, r = 2.6, layers = 32`（大致对应 Llama 3 8B）时，单层约为 `4·4096² + 3·2.6·4096² ≈ 198M` parameters；32 层约 `6.3B`，再加 embeddings 和 head，接近公开参数规模。

## Build It / 动手构建

### Step 1: the building blocks / 第 1 步：基础 building blocks

使用 Lesson 03 的 tiny `Matrix` class（为保持独立性复制到本文件）：

- `layer_norm(x, eps=1e-5)` — 减去 mean，除以 std。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。不减 mean。
- `gelu(x)` 与 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 与 `decoder_block(x, enc_out, params)`。

完整 wiring 见 `code/main.py`。

### Step 2: wire a 2-layer encoder and a 2-layer decoder / 第 2 步：连接 2-layer encoder 与 2-layer decoder

把它们 stack 起来。把 encoder output 传给每个 decoder cross-attention。在 output projection 前添加 final LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Step 3: run forward on a toy example / 第 3 步：在 toy example 上跑 forward

送入一个 6-token source 和一个 5-token target。验证 output shape 是 `(5, vocab)`。不做 training，这一课关注 architecture，不关注 loss。

### Step 4: swap in RMSNorm + SwiGLU / 第 4 步：替换为 RMSNorm + SwiGLU

用 RMSNorm 和 SwiGLU 替换 LayerNorm 与 ReLU-FFN。确认 shapes 仍然匹配。这就是用一个函数替换完成的 2026 modernization。

## Use It / 应用它

PyTorch/TF 的 reference implementations 是 `nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但 2026 年多数 production code 会自己写 block，因为：

- Flash Attention 在 attention 内部调用，而不是通过 `nn.MultiheadAttention`。
- GQA / MLA 不在 stdlib reference 中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch defaults。

HF `transformers` 里有值得阅读的清晰 reference blocks：`modeling_llama.py` 是 2026 年 canonical decoder-only block。它大约 500 行，值得完整走读一次。

**Encoder vs decoder vs encoder-decoder — when to pick / 何时选择：**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

Decoder-only 在语言任务中胜出，因为它 scaling 最干净，并且同时处理 comprehension 和 generation。Encoder-decoder 在 input 有明确 “source sequence” 身份时仍是最佳选择，例如 translation、speech recognition、structured tasks。

## Ship It / 交付它

见 `outputs/skill-transformer-block-reviewer.md`。这个 skill 会按 2026 年默认实践 review 一个新的 transformer block implementation，并标记缺失组件（pre-norm、RoPE、RMSNorm、GQA、FFN expansion ratio）。

## Exercises / 练习

1. **Easy / 简单。** 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 时计算你的 encoder_block 参数量。实现 block 后用 `sum(p.numel() for p in block.parameters())` 验证。
2. **Medium / 中等。** 从 post-norm 切换到 pre-norm。初始化两者，在 random input 上堆 12 层后测量 activation norm。Post-norm 的 activations 应该会 explode；pre-norm 应该保持 bounded。
3. **Hard / 困难。** 在 toy copy task（copy `x` reversed）上实现一个 4-layer encoder-decoder。训练 100 steps，报告 loss。换成 RMSNorm + SwiGLU + RoPE 后，loss 是否下降？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Block | “One transformer layer” | norm + attention + norm + FFN 的 stack，并包在 residual connections 中。 |
| Residual | “Skip connection” | `x + f(x)` output；让 gradients 能流经 deep stacks。 |
| Pre-norm | “Normalize before, not after” | 现代形式：`x + sublayer(LN(x))`。不用复杂 warmup 也能训练更深。 |
| RMSNorm | “LayerNorm without the mean” | 除以 RMS；少一个 op，经验稳定性相同。 |
| SwiGLU | “The FFN everyone switched to” | `Swish(W1 x) ⊙ W3 x → W2`。在 LM ppl 上优于 ReLU/GELU。 |
| Cross-attention | “How the decoder sees the encoder” | Q 来自 decoder、K/V 来自 encoder outputs 的 MHA。 |
| FFN expansion | “How wide the middle MLP is” | hidden-size 与 d_model 的比例，通常是 4（LayerNorm）或 2.6（SwiGLU）。 |
| Bias-free | “Drop the +b terms” | 现代 stack 省略 linear layers 的 biases；ppl 略好，模型更小。 |

## Further Reading / 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 block specification。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么 pre-norm 在深层上优于 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — canonical 2026 decoder-only block。
