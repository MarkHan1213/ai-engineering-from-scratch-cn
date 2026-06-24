# CNNs and RNNs for Text / 文本中的 CNN 与 RNN

> 卷积学习 n-grams。循环网络负责记忆。两者都被 attention 超越了。两者在受限硬件上仍然重要。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 TextCNN 如何把 1D convolution 用作可学习的 n-gram detector
- 使用 PyTorch 实现 TextCNN 与 LSTM text classifier
- 理解 vanishing gradient、LSTM/GRU gating 和 bidirectional RNN 的动机
- 判断 CNN/RNN 在 edge、streaming、baseline 和 sequence labeling 场景中的价值

## The Problem / 问题

TF-IDF 和 Word2Vec 产生的是忽略词序的扁平向量。基于它们的分类器无法区分 `dog bites man` 和 `man bites dog`。而词序有时就是信号。

Transformer 出现之前，有两类架构填补了这个缺口。

**Convolutional nets for text (TextCNN).** 在 word embeddings 序列上应用 1D convolutions。宽度为 3 的 filter 是一个可学习 trigram detector：它跨越三个词并输出一个分数。堆叠不同宽度（2、3、4、5）来检测多尺度模式。Max-pool 成固定长度表示。扁平、并行、快速。

**Recurrent nets (RNN, LSTM, GRU).** 一次处理一个 token，并维护携带前文信息的 hidden state。顺序处理、有记忆、输入长度灵活。它们在 2014 到 2017 年统治了 sequence modeling，直到 attention 出现。

这一课会构建两者，并指出推动 attention 出现的失败点。

## The Concept / 概念

**TextCNN**（Kim, 2014）。Tokens 先变成 embeddings。宽度为 `k` 的 1D convolution 会让 filter 滑过连续 `k`-grams 的 embeddings，产生 feature map。对 feature map 做 global max-pooling 取最强 activation。拼接多个 filter widths 的 max-pooled outputs。喂给 classifier head。

为什么有效：filter 是可学习的 n-gram。Max-pooling 与位置无关，所以 "not good" 不管出现在评论开头还是中间，都会触发同一个 feature。三个 filter widths、每个 100 filters，就得到 300 个学出来的 n-gram detectors。训练可以并行，没有顺序依赖。

**RNN.** 在每个时间步 `t`，hidden state `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b` 在时间维上共享。时间 `T` 的 hidden state 是整个 prefix 的摘要。做分类时，可以对 `h_1 ... h_T` 做 pooling（max、mean 或 last）。

Plain RNN 会遭遇 vanishing gradients。**LSTM** 增加 gates，决定忘记什么、存储什么、输出什么，从而稳定长序列上的梯度。**GRU** 把 LSTM 简化为两个 gates；参数更少，效果相近。

**Bidirectional RNNs** 会同时运行一个正向 RNN 和一个反向 RNN，再拼接 hidden states。每个 token 的表示都能看到左、右两侧上下文。对 tagging tasks 很关键。

```figure
rnn-unroll
```

## Build It / 动手构建

### Step 1: TextCNN in PyTorch / 第 1 步：用 PyTorch 实现 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 会把 `[batch, seq_len, embed_dim]` 改成 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 把中间轴当作 channels。无论输入长度如何，pooled output 都是固定大小。

### Step 2: LSTM classifier / 第 2 步：LSTM classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

这里对 sequence 做 max-pool，而不是 last-state pool。对分类任务来说，max-pooling 通常胜过最后 hidden state，因为长序列末尾的信息容易支配 last state。

### Step 3: the vanishing gradient demo (intuition) / 第 3 步：vanishing gradient 直觉演示

没有 gating 的 plain RNN 学不好长程依赖。考虑一个 toy task：预测 token `A` 是否在序列中出现过。如果 `A` 在位置 1，而序列长度是 100，loss 的梯度必须穿过 recurrent weight 的 99 次乘法才能回到开头。如果权重小于 1，梯度消失；如果大于 1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM 用 **cell state** 修复这个问题：它通过网络时主要是加性相互作用（forget gate 会乘性缩放它，但梯度仍然能沿着“高速通道”流动）。GRU 用更少参数做类似的事。两者都能在 100+ step sequences 上稳定训练。

### Step 4: why this still was not enough / 第 4 步：为什么这仍然不够

即使用 LSTM，仍然有三个问题。

1. **Sequential bottleneck / 顺序瓶颈。** 在长度 1000 的序列上训练 RNN，需要 1000 个串行 forward/backward steps。无法沿时间维并行。
2. **Encoder-decoder 中的 fixed-size context vector。** Decoder 只能看到 encoder 的最后 hidden state，也就是整个输入压缩后的结果。长输入会丢细节。Lesson 09 会直接讲这个问题。
3. **Distant-dependency accuracy ceiling / 远距离依赖准确率上限。** LSTM 胜过 plain RNN，但仍然难以把具体信息传播 200+ steps。

Attention 解决了这三个问题。Transformer 完全移除了 recurrence。Lesson 10 是转折点。

## Use It / 应用它

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 都是生产可用的。训练代码也很标准。

Hugging Face 提供预训练 embeddings，可以把它们接成输入层：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

适用约束 checklist：

- **Edge / on-device inference.** TextCNN + GloVe embeddings 比 transformer 小 10-100 倍。如果部署目标是手机，就选这套。
- **Streaming / online classification.** RNN 一次处理一个 token；transformer 需要完整序列。对实时进入的文本，LSTM 仍然占优。
- **Tiny models for baselines.** 新任务快速迭代。在 CPU 上 5 分钟就能训练一个 TextCNN。
- **Sequence labeling with limited data.** BiLSTM-CRF（lesson 06）在 1k-10k 标注句子的 NER 上仍是生产级架构。

其他情况交给 transformer。

## Ship It / 交付它

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## Exercises / 练习

1. **Easy / 简单。** 在一个 3-class toy dataset（你自己造数据）上训练 TextCNN。验证 filter widths (2, 3, 4) 在 average F1 上优于单一 width (3)。
2. **Medium / 中等。** 为 LSTM classifier 实现 max-pool、mean-pool 和 last-state pooling。在小数据集上比较；记录哪种 pooling 胜出，并推测原因。
3. **Hard / 困难。** 构建 BiLSTM-CRF NER tagger（结合 lesson 06 和本课）。在 CoNLL-2003 上训练。与 lesson 06 的 CRF-alone baseline 和 BERT fine-tune 对比。报告训练时间、内存和 F1。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| TextCNN | 文本 CNN | 在 word embeddings 上堆叠 1D convolutions，并做 global max-pool。Kim (2014)。 |
| RNN | 循环网络 | 每个时间步更新 hidden state：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 带门控的 RNN | 增加 input / forget / output gates 和 cell state。能在长序列上稳定训练。 |
| GRU | 更简单的 LSTM | 两个 gates，而不是三个。准确率相似，参数更少。 |
| Bidirectional | 双向 | 正向 + 反向 RNN 拼接。每个 token 都能看到上下文两侧。 |
| Vanishing gradient | 训练信号消失 | Plain RNN 中反复乘以 <1 的权重，会让早期 step 的梯度几乎为零。 |

## Further Reading / 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文。八页，可读。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文。意外地清楚。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 对所有人都变得可理解的图解。
