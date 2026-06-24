# Sequence-to-Sequence Models / 序列到序列模型

> 两个 RNN 假装成翻译器。它撞上的瓶颈，正是 attention 存在的理由。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 encoder、decoder、context vector 如何组成 seq2seq 架构
- 用 PyTorch 实现 GRU encoder、decoder、teacher forcing 训练循环和 greedy decoding
- 理解 exposure bias、scheduled sampling、beam search 与 context-vector bottleneck
- 判断何时应直接使用现代 transformer encoder-decoder checkpoint

## The Problem / 问题

分类把可变长度序列映射成单个标签。翻译把可变长度序列映射成另一个可变长度序列。输入和输出位于不同 vocabulary，甚至可能是不同语言，而且长度不一定对应。

Seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一个刻意简单的 recipe 解决了这个问题。两个 RNN。一个读取 source sentence，产生固定大小的 context vector。另一个读取这个 vector，并逐 token 生成 target sentence。你在 lesson 08 写过的代码，只是换了一种粘合方式。

它值得学习有两个原因。第一，context-vector bottleneck 是 NLP 中最有教学价值的失败。它直接说明 attention 和 transformer 为什么擅长这些问题。第二，训练 recipe（teacher forcing、scheduled sampling、推理时 beam search）仍然适用于包括 LLM 在内的每个现代生成系统。

## The Concept / 概念

**Encoder.** 读取 source sentence 的 RNN。它的最终 hidden state 是 **context vector**，也就是整个输入的固定大小摘要。理论上，除了 source 本身，不丢任何东西。

**Decoder.** 另一个由 context vector 初始化的 RNN。每一步它把上一个生成 token 作为输入，并输出 target vocabulary 上的分布。Sample 或 argmax 选择下一个 token，再喂回去。直到产生 `<EOS>` token 或达到 max length。

**Training / 训练：** 每个 decoder step 上计算 cross-entropy loss，并在序列上求和。对两个网络一起做标准 backprop through time。

**Teacher forcing.** 训练时，decoder 在 step `t` 的输入是位置 `t-1` 的 *ground-truth* token，而不是 decoder 自己上一步的预测。这会稳定训练；没有它，早期错误会级联，模型学不起来。推理时必须使用模型自己的预测，所以训练/推理分布永远有缺口。这个缺口叫 **exposure bias**。

**The bottleneck / 瓶颈。** Encoder 对 source 学到的一切，都必须挤进那个 context vector。长句会丢细节。稀有词会被模糊掉。重排序（chat noir vs. black cat）只能靠记忆，而不是被计算出来。

Attention（lesson 10）通过让 decoder 查看 *每个* encoder hidden state，而不只是最后一个，直接修复这个问题。核心卖点就这一句。

```figure
lstm-gates
```

## Build It / 动手构建

### Step 1: an encoder / 第 1 步：encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的 shape 是 `[batch, seq_len, hidden_dim]`，每个输入位置一个 hidden state。`hidden` 的 shape 是 `[1, batch, hidden_dim]`，也就是最后一步。Lesson 08 说“分类时对 outputs 做 pool”。这里我们保留最后 hidden state 作为 context vector，并忽略 per-step outputs。

### Step 2: a decoder / 第 2 步：decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder 一次调用一步。输入：一批单 token 和当前 hidden state。输出：下一个 token 的 vocabulary logits 和更新后的 hidden state。

### Step 3: training loop with teacher forcing / 第 3 步：带 teacher forcing 的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个 knob 值得点名。`ignore_index=0` 会跳过 padding tokens 上的 loss。`teacher_forcing_ratio` 是每一步使用真实 token 而不是模型预测的概率。从 1.0（full teacher forcing）开始训练，再逐渐退火到约 0.5，以缩小 exposure-bias gap。

### Step 4: inference loop (greedy) / 第 4 步：推理循环（greedy）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Greedy decoding 每一步都选择最高概率 token。它可能走偏：一旦提交某个 token，就不能反悔。**Beam search** 会保留 top-`k` 个 partial sequences，并在最后选择分数最高的完整序列。Beam width 3-5 是常见设置。

### Step 5: the bottleneck, demonstrated / 第 5 步：演示瓶颈

在 toy copy task 上训练模型：source `[a, b, c, d, e]`，target `[a, b, c, d, e]`。逐渐增加序列长度，观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU hidden state 无法无损记住 40-token 输入。信息存在于每个 encoder step，但 decoder 只能看到最后状态。Attention 会直接修复这一点。

## Use It / 应用它

PyTorch 有 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face `transformers` 库提供在数十亿 tokens 上训练好的完整 encoder-decoder models（BART、T5、mBART、NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代 encoder-decoders 已经从 RNN 换成 transformer。高层形状（encoder、decoder、逐 token 生成）与 2014 seq2seq 论文完全一致。每个 block 内部机制不同。

### When to still reach for RNN-based seq2seq / 什么时候仍然选择 RNN-based seq2seq

新项目几乎永远不要。少数例外：

- Streaming translation：你需要一次消费一个输入 token，并保持有界内存。
- On-device text generation：transformer 内存成本太高。
- 教学。理解 encoder-decoder bottleneck，是理解 transformer 为什么胜出的最快路径。

### Exposure bias and its mitigations / Exposure bias 及其缓解

- **Scheduled sampling.** 训练中退火 teacher forcing ratio，让模型学会从自己的错误中恢复。
- **Minimum risk training.** 用 sentence-level BLEU score 训练，而不是 token-level cross-entropy。更接近你真正想要的目标。
- **Reinforcement learning fine-tuning.** 用指标奖励 sequence generator。现代 LLM RLHF 也使用这种思想。

这三者仍然适用于 transformer-based generation。

## Ship It / 交付它

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Exercises / 练习

1. **Easy / 简单。** 实现 toy copy task。训练一个 GRU seq2seq，其中 target 等于 source。测量长度 5、10、20 上的准确率，复现 bottleneck。
2. **Medium / 中等。** 增加 beam width 3 的 beam search decoding。在小型 parallel corpus 上对比 greedy 的 BLEU。记录 beam search 在哪里胜出（通常是最后几个 token），以及在哪里没有区别。
3. **Hard / 困难。** 在 10k-pair paraphrase dataset 上 fine-tune `facebook/bart-base`。比较 fine-tuned model 的 beam-4 output 与 base model 在 held-out inputs 上的输出。报告 BLEU，并挑选 10 个定性样例。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Encoder | 输入 RNN | 读取 source。产生 per-step hidden states 和最终 context vector。 |
| Decoder | 输出 RNN | 由 context vector 初始化。一次生成一个 target token。 |
| Context vector | 摘要 | 最终 encoder hidden state。固定大小。Attention 要解决的瓶颈。 |
| Teacher forcing | 使用真实 token | 训练时喂 ground-truth previous token。稳定学习。 |
| Exposure bias | 训练/测试缺口 | 模型训练时只看真实 token，从未练习从自己的错误中恢复。 |
| Beam search | 更好的 decoding | 每一步保留 top-k partial sequences，而不是 greedy 地一次提交。 |

## Further Reading / 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始 seq2seq 论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入 GRU 和 encoder-decoder framing。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — attention 论文。读完本课后立刻读它。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可构建的 seq2seq + attention 代码。
