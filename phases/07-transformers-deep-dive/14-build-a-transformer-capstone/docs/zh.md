# Build a Transformer from Scratch — The Capstone / 从零构建 Transformer：Capstone

> 十三课。一个模型。不走捷径。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 01 through 13. Don't skip.
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 把前 13 课中的 attention、MHA、positional encoding、decoder blocks、causal loss、KV cache 等概念整合到一个可训练模型中
- 实现一个 small decoder-only transformer，并在 character-level language modeling task 上端到端训练
- 理解 nanoGPT-style training loop 的 data batching、shift-by-one loss、optimizer、sampling 流程
- 识别 RoPE、KV cache、Flash Attention、MoE 等工程扩展在 capstone 中的插入点

## The Problem / 问题

你已经读过每篇论文。你实现过 attention、multi-head splits、positional encodings、encoder and decoder blocks、BERT and GPT losses、MoE、KV cache。现在要让它们在一个真实任务中协同工作。

Capstone：在 character-level language modeling task 上端到端训练一个 small decoder-only transformer。它读 Shakespeare。它生成新的 Shakespeare。它小到能在 laptop 上 10 分钟内训练完，又正确到只要换成更大 dataset 和更长训练，就能得到真实 LM。

这是本课程的 “nanoGPT”。它不是原创；Karpathy 2023 年 nanoGPT tutorial 是每个学生至少会写一次的 reference implementation。我们借用其 shape，并围绕本阶段已经覆盖的内容重新组织。

## The Concept / 概念

![Transformer-from-scratch block diagram](../assets/capstone.svg)

架构标注如下：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### What we ship / 我们交付什么

- `GPTConfig` — 统一配置所有 hyperparameters。
- `MultiHeadAttention` — causal、batched，带 optional Flash-style pathway（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` — 现代 FFN。
- `Block` — pre-norm，residual-wrapped attention + FFN。
- `GPT` — embeddings、stacked blocks、LM head、generate()。
- Training loop：AdamW、cosine LR、gradient clipping。
- Shakespeare text 上的 char-level tokenizer。

### What we don't ship / 我们不交付什么

- RoPE — Lesson 04 已经概念实现。这里为了简单使用 learned positional embeddings。练习会让你换成 RoPE。
- Generation 中的 KV cache — 每个 generation step 重新在 full prefix 上计算 attention。更慢但更简单。练习会让你加 KV cache。
- Flash Attention — PyTorch 2.0+ 会在输入匹配时 auto-dispatch；我们使用 `F.scaled_dot_product_attention`。
- MoE — 每个 block 一个 FFN。Lesson 11 已经看过 MoE。

### Target metrics / 目标指标

在 Mac M2 laptop 上，一个 4-layer、4-head、d_model=128 的 GPT，在 `tinyshakespeare.txt` 上训练 2,000 steps：

- Training loss 约 6 分钟内从 ~4.2（random）收敛到 ~1.5。
- Sampled output 看起来像 Shakespeare：古旧词汇、换行、"ROMEO:" 这样的 proper names 会出现。
- Val loss（held-out final 10% text）紧跟 training loss；在这个 size/budget 下没有 overfitting。

## Build It / 动手构建

本课使用 PyTorch。安装 `torch`（CPU build 即可）。见 `code/main.py`。脚本会处理：

- 如果缺失则下载 `tinyshakespeare.txt`（或读取本地 copy）。
- Byte-level char tokenizer。
- 90/10 train/val split。
- 支持硬件上的 bf16 autocast training loop。
- 训练完成后 sampling。

### Step 1: data / 第 1 步：data

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个 unique characters。Tiny vocabulary。适合 4-byte vocab_size。没有 BPE，没有 tokenizer drama。

### Step 2: model / 第 2 步：model

见 `code/main.py`。Block 是 Lesson 05 的 textbook 版本：pre-norm、RMSNorm、SwiGLU、causal MHA。4/4/128 的参数量约 800K。

### Step 3: training loop / 第 3 步：training loop

随机取 length-256 token windows 的 batch。Forward。Shift-by-one cross-entropy。Backward。AdamW step。Log。Repeat。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Step 4: sample / 第 4 步：sample

给定 prompt，重复 forward，从 top-p logits 中 sample，append，再继续。500 tokens 后停止。

### Step 5: read the output / 第 5 步：阅读输出

2,000 steps 后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是 Shakespeare，但已经 Shakespeare-shaped。对 ~800K parameters 和 laptop 上 6 分钟训练来说，这是明确胜利。

## Use It / 应用它

这个 capstone 是 reference architecture。要把它推向真实应用，可以做三类扩展：

1. **Swap the tokenizer / 替换 tokenizer。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。Vocab size 会从 65 跳到约 50,000。Model capacity 也需要随之 scale。
2. **Train on a bigger corpus / 换更大语料。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。单张 A100 上训练 10B tokens、125M-param GPT 大约需要 24 小时。
3. **Add RoPE + KV cache + Flash Attention / 加入 RoPE、KV cache、Flash Attention。** 下面的练习会逐步带你做。

最终会得到一个 125M-parameter GPT，可以生成流畅英文。它不是 frontier model。但同一条 code path，只是更大，就是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年训练 research checkpoints 的方式。

## Ship It / 交付它

见 `outputs/skill-transformer-review.md`。这个 skill 会 review 一个 transformer-from-scratch implementation，检查前 13 课覆盖的正确性要点。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。验证 trained model final-step validation loss 低于 2.0。把 `max_steps` 从 2,000 改到 5,000，val loss 是否继续改善？
2. **Medium / 中等。** 用 RoPE 替换 learned positional embeddings。在 `MultiHeadAttention` 内部对 Q 和 K 应用 rotation。训练并验证 val loss 至少不更差。
3. **Medium / 中等。** 在 sampling loop 中实现 KV cache。分别用 cache 与不用 cache 生成 500 tokens。Laptop 上 wall-clock 应提升 5–20×。
4. **Hard / 困难。** 给模型加第二个 head，预测 next-plus-one token（MTP — DeepSeek-V3 的 Multi-Token Prediction）。联合训练。它有帮助吗？
5. **Hard / 困难。** 把每个 block 的 single FFN 替换为 4-expert MoE。Router + top-2 routing。在 matched active parameters 下观察 val loss 如何变化。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| nanoGPT | “Karpathy's tutorial repo” | Minimal decoder-only transformer training code，约 300 LOC；canonical reference。 |
| tinyshakespeare | “The standard toy corpus” | 约 1.1 MB text；2015 年以来每个 character-LM tutorial 都用它。 |
| Tied embeddings | “Share input/output matrix” | LM head weight = token embedding matrix 的 transpose；省参数，提高质量。 |
| bf16 autocast | “Training precision trick” | Forward/backward 用 bf16，optimizer state 保持 fp32；2021 年后成为标准。 |
| Gradient clipping | “Stops spikes” | 把 global grad norm capped at 1.0；防止 training blowups。 |
| Cosine LR schedule | “The 2020+ default” | LR 先 linear warmup，再按 cosine 形状 decay 到 peak 的 10%。 |
| MFU | “Model FLOP Utilization” | Achieved FLOPs / theoretical peak；2026 年 dense 40%、MoE 30% 已经很强。 |
| Val loss | “Held-out loss” | 模型从未见过的数据上的 cross-entropy；overfit detector。 |

## Further Reading / 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 经典 annotated implementation。
