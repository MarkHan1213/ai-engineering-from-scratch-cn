# Positional Encoding — Sinusoidal, RoPE, ALiBi / 位置编码：Sinusoidal、RoPE 与 ALiBi

> Attention 对 permutation 不敏感。如果没有位置信号，"The cat sat on the mat" 和 "mat the on sat cat the" 会产生同样的输出。三个算法修复这个问题，每个都对“position”意味着什么做了不同押注。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释为什么 scaled dot-product attention 本身是 order-blind
- 实现 absolute sinusoidal encoding、RoPE rotation 和 ALiBi bias
- 理解 RoPE 如何让 dot product 依赖 relative distance
- 根据 context length、extrapolation 需求和训练预算选择 positional encoding strategy

## The Problem / 问题

Scaled dot-product attention 不懂顺序。Attention matrix `softmax(Q K^T / √d) V` 只由 pairwise similarities 计算。把 `X` 的 rows shuffle，输出 rows 也只会以同样方式 shuffle。Attention 内部没有任何东西关心 position。

这对 bag-of-words model 不算 bug。但对 language、code、audio、video 这类 order 承载语义的输入来说，是致命问题。

修复方式是以某种方式把 position 注入 embeddings。三个时代有三种答案：

1. **Absolute sinusoidal**（Vaswani 2017）。把 position 的 `sin/cos` 加到 embedding 上。简单、无需学习参数，但超出训练长度后 extrapolation 差。
2. **RoPE — Rotary Position Embeddings**（Su 2021）。按与 position 成比例的角度旋转 Q 和 K vectors。它会在 dot product 中直接编码 *relative* position。2026 年主流选择。
3. **ALiBi — Attention with Linear Biases**（Press 2022）。完全跳过 embeddings，对 attention scores 按距离加一个 per-head linear penalty。长度外推非常好。

截至 2026 年，几乎所有 frontier open model 都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数 long-context models 使用 ALiBi 或其现代变体。Absolute sinusoidal 主要是历史方案。

## The Concept / 概念

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Absolute sinusoidal / 绝对 sinusoidal

预先计算一个 shape 为 `(max_len, d_model)` 的固定 matrix `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在 attention 之前做 `X' = X + PE[:N]`。每个 dimension 是不同 frequency 的 sinusoid。模型会学习从 phase pattern 中读取 position。它在 `max_len` 之外会失败：如果模型只见过 positions 0–2047，就没人告诉它 position 2048 以后会发生什么。

### RoPE / RoPE

旋转 Q 和 K vectors（不是 embeddings）。对一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对 position `pos_k` 的 keys 应用同样 rotation。Dot product `q'_m · k'_n` 会变成只关于 `(m - n)` 的函数。也就是说：**attention score 只依赖 relative distance**，虽然 rotation 本身是由 absolute positions 驱动的。这个技巧非常漂亮。

扩展 RoPE 时，可以 scale `base`（NTK-aware、YaRN、LongRoPE），在不重新训练的情况下外推到更长 context。Llama 3 就这样从 8K context 扩展到 128K。

### ALiBi / ALiBi

跳过 embedding trick，直接 bias attention scores：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是 head-specific slope（例如 `1 / 2^(8·h/H)`）。更近的 tokens 会被 boost，更远的 tokens 会被 penalize。没有训练时开销。论文显示，其 length extrapolation 超过 sinusoidal，并在原训练长度内匹配 RoPE。

### What to pick in 2026 / 2026 年怎么选

| Variant | Extrapolation | Training cost | Used by |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | poor | free | original transformer, early BERT |
| Learned absolute | none | tiny | GPT-2, GPT-3 |
| RoPE | good with scaling | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | excellent | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | excellent | free | BLOOM, MPT, Baichuan |

RoPE 胜出的原因是它能无缝嵌入 attention，不改变 architecture；它编码 relative position；并且它的 `base` hyperparameter 为 long-context fine-tuning 提供了清晰旋钮。

```figure
rope-explorer
```

## Build It / 动手构建

### Step 1: sinusoidal encoding / 第 1 步：sinusoidal encoding

见 `code/main.py`。4 行核心计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

把它加到第一个 attention layer 之前的 embedding matrix 上。

### Step 2: RoPE applied to Q, K / 第 2 步：把 RoPE 应用到 Q、K

RoPE 对 Q 和 K 原地操作。对每对 dims：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键点：对 position `m` 的 Q 和 position `n` 的 K 应用同一个函数。它们的 dot product 会在每个 coordinate pair 上带上 `cos((m-n)·θ_i)` 因子。Attention 因此免费学到 relative position。

### Step 3: ALiBi slopes and bias / 第 3 步：ALiBi slopes 与 bias

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

把 `bias[h]` 加到 head `h` 的 `(seq_len, seq_len)` attention score matrix 上，然后做 softmax。

### Step 4: verify relative-distance property of RoPE / 第 4 步：验证 RoPE 的 relative-distance property

取两个 random vectors `a, b`。先按 `(pos_a, pos_b)` 旋转，再按 `(pos_a + k, pos_b + k)` 旋转。两个 dot products 必须在 floating-point error 内相等。这个性质就是 RoPE 的核心：它对 absolute offset 不敏感，只关心 relative gap。

## Use It / 应用它

PyTorch 2.5+ 在 `torch.nn.functional` 中提供 RoPE utilities。多数 production code 使用 `flash_attn` 或 `xformers`，在 attention kernel 内部应用 RoPE。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**Long-context tricks in 2026 / 2026 年的 long-context 技巧：**

- **NTK-aware interpolation.** 从 4K 扩展到 16K+ 时，把 `base` rescale 到 `base * (scale_factor)^(d/(d-2))`。
- **YaRN.** 更聪明的 interpolation，在 long context 上保持 attention entropy。Llama 3.1 128K 使用它。
- **LongRoPE.** Microsoft 2024 年方法，用 evolutionary search 选择 per-dimension scale factors。Phi-3-Long 使用它。
- **Position interpolation + fine-tuning.** 直接按 extension factor 缩小 positions，然后 fine-tune 1–5B tokens。效果出奇地好。

## Ship It / 交付它

见 `outputs/skill-positional-encoding-picker.md`。这个 skill 会根据 target context length、extrapolation needs 和 training budget，为新模型选择 encoding strategy。

## Exercises / 练习

1. **Easy / 简单。** 把 `max_len=512, d=128` 的 sinusoidal `PE` matrix 画成 heatmap。确认 “stripes get wider as dimension index grows” 的 pattern。
2. **Medium / 中等。** 实现 NTK-aware RoPE scaling。在 length 256 的 sequences 上训练一个 tiny LM，然后在 length 1024 上分别用 scaling 和不用 scaling 测试，测量 perplexity。
3. **Hard / 困难。** 在同一个 attention module 中实现 ALiBi 和 RoPE。训练一个 4-layer transformer，在 length 512 的 copy task 上训练，在 test time extrapolate 到 2048。比较退化程度。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Positional encoding | “Tells attention about order” | 添加到 embeddings 或 attention 中、用于编码 position 的任何信号。 |
| Sinusoidal | “The original one” | 几何 frequencies 上的 `sin/cos` 加到 embeddings；不能很好 extrapolate。 |
| RoPE | “Rotary embeddings” | 按 position-dependent angle 旋转 Q、K；dot product 编码 relative distance。 |
| ALiBi | “Linear bias trick” | 对 attention scores 加 `-m·\|i-j\|`；不需要 embedding，extrapolation 很强。 |
| base | “RoPE's knob” | RoPE 中的 frequency scaler；增大它可以在 inference 时扩展 context。 |
| NTK-aware | “A RoPE scaling trick” | Rescale `base`，避免 context 扩大时 high-frequency dims 被挤压。 |
| YaRN | “The fancy one” | 保持 attention entropy 的 per-dimension interpolation+extrapolation。 |
| Extrapolation | “Works beyond trained length” | position scheme 能否在超过训练中见过的 `max_len` 后仍给出正确输出。 |

## Further Reading / 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始 sinusoidal。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 当前最先进的 RoPE scaling。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 long-context 论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Phi-3-Long 使用并在 Use It 中引用的 Microsoft 方法。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 各类 RoPE scaling scheme（default、linear、dynamic、YaRN、LongRoPE、Llama-3）的 production-grade implementations。
