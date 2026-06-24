# BERT — Masked Language Modeling / BERT：Masked Language Modeling

> GPT 预测下一个词。BERT 预测缺失的词。只差一句话，却开启了半个十年的 embedding-shaped everything。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 理解 Masked Language Modeling 如何让 encoder 从双向上下文中学习表示
- 实现 BERT 的 15% token 选择和 80/10/10 masking rule
- 解释 NSP 为什么被 RoBERTa 后续工作移除
- 判断 2026 年何时仍应选择 encoder-only model，而不是 decoder-only LLM

## The Problem / 问题

2018 年，每个 NLP task，包括 sentiment、NER、QA、entailment，都会在自己的 labeled data 上从零训练自己的模型。没有一个预训练好的 “understand English” checkpoint 可以拿来 fine-tune。ELMo（2018）展示了可以用 bidirectional LSTM 预训练 contextual embeddings；它有帮助，但泛化有限。

BERT（Devlin et al. 2018）问了一个问题：如果我们取一个 transformer encoder，在互联网上所有句子上训练，并强迫它根据左右两侧 context 预测缺失词，会怎样？然后你只需要在 downstream task 上 fine-tune 一个 head。Parameter efficiency 当时是一次冲击。

结果是：18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了当时存在的每个 NLP leaderboard。到 2020 年，地球上每个 search engine、content moderation pipeline 和 semantic-search system 里都有一个 BERT。

到 2026 年，encoder-only models 仍然是 classification、retrieval 和 structured extraction 的正确工具。它们每 token 比 decoders 快 5–10×，而它们的 embeddings 是每个现代 retrieval stack 的 backbone。ModernBERT（2024 年 12 月）用 Flash Attention + RoPE + GeGLU 把 architecture 推到 8K context。

## The Concept / 概念

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### The training signal / 训练信号

取一句话：`the quick brown fox jumps over the lazy dog`。

随机 mask 15% tokens：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型在 masked positions 预测原始 tokens。因为 encoder 是 bidirectional，预测 position 1 的 `[MASK]` 时可以使用 positions 2+ 的 `brown fox jumps`。这是 GPT 做不到的。

### The BERT mask rules / BERT 的 mask 规则

在被选中用于预测的 15% tokens 中：

- 80% 被替换成 `[MASK]`。
- 10% 被替换成 random token。
- 10% 保持不变。

为什么不总是用 `[MASK]`？因为 `[MASK]` 在 inference time 不会出现。如果训练时 100% masked positions 都是 `[MASK]`，模型会在 pretraining 与 fine-tuning 之间出现 distribution shift。10% random + 10% unchanged 会让模型保持诚实。

### Next Sentence Prediction (NSP) — and why it was dropped / Next Sentence Prediction（NSP）以及为什么被移除

原始 BERT 还训练 NSP：给定两个句子 A 和 B，预测 B 是否跟在 A 后面。RoBERTa（2019）做 ablation 后发现 NSP 有害而不是有益。现代 encoders 会跳过它。

### What changed in 2026: ModernBERT / 2026 年的变化：ModernBERT

2024 年 ModernBERT 论文用 2026 primitives 重建了 block：

| Component | Original BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

并且不同于 2018 年 stack，它原生支持 Flash Attention。在 sequence length 8K 时，inference 比 DeBERTa-v3 快 2–3×，GLUE scores 也更好。

### Use cases that still pick an encoder in 2026 / 2026 年仍然选择 encoder 的场景

| Task | Why encoder beats decoder |
|------|---------------------------|
| Retrieval / semantic search embeddings | Bidirectional context = better embedding quality per token |
| Classification (sentiment, intent, toxicity) | One forward pass; no generation overhead |
| NER / token labeling | Per-position output, natively bidirectional |
| Zero-shot entailment (NLI) | Classifier head on top of encoder |
| Reranker for RAG | Cross-encoder scoring, 10x faster than LLM rerankers |

```figure
transformer-residual
```

## Build It / 动手构建

### Step 1: masking logic / 第 1 步：masking logic

见 `code/main.py`。函数 `create_mlm_batch` 接收 token IDs、vocab size 和 mask probability。返回 input IDs（已应用 masks）和 labels（只在 masked positions 填原始 token，其他位置为 -100，即 PyTorch ignore index convention）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Step 2: run MLM prediction on a tiny corpus / 第 2 步：在 tiny corpus 上运行 MLM prediction

在 20 个词的 vocabulary、200 个 sentences 上训练 2-layer encoder + MLM head。不做 gradient，只做 forward-pass sanity checks。完整 training 需要 PyTorch。

### Step 3: compare mask types / 第 3 步：比较 mask types

展示 three-way rule 如何让模型在没有 `[MASK]` 的情况下仍可用。分别在 unmasked sentence 和 masked sentence 上预测。因为模型在训练中见过两种 pattern，两者都应该产生合理 token distributions。

### Step 4: fine-tune head / 第 4 步：fine-tune head

在 toy sentiment dataset 上把 MLM head 替换成 classification head。只训练 head，encoder 冻结。这是每个 BERT application 都遵循的模式。

## Use It / 应用它

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding models are fine-tuned BERT / Embedding models 本质上是 fine-tuned BERT。** `sentence-transformers` 里的 `all-MiniLM-L6-v2` 这类模型，是用 contrastive loss 训练的 BERT。Encoder 相同，loss 变了。

**Cross-encoder rerankers are also fine-tuned BERT / Cross-encoder rerankers 也是 fine-tuned BERT。** 输入形式是 `[CLS] query [SEP] doc [SEP]` 上的 pair-classification。Query 与 doc 之间的 bidirectional attention 正是 cross-encoder 相比 biencoder 有质量优势的原因。

**When not to pick BERT in 2026 / 2026 年什么时候不选 BERT。** 任何 generative task 都不适合。Encoder 没有合理方式 autoregressively 生成 tokens。另外：在 1B params 以下，small decoder（Phi-3-Mini、Qwen2-1.5B）往往能以更灵活方式匹配质量。

## Ship It / 交付它

见 `outputs/skill-bert-finetuner.md`。这个 skill 会为新的 classification 或 extraction task 规划 BERT fine-tune（backbone choice、head spec、data、eval、stopping）。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`，打印 10,000 tokens 上的 mask distribution。确认约 15% 被选中，其中约 80% 变成 `[MASK]`。
2. **Medium / 中等。** 实现 whole-word masking：如果一个 word 被 tokenized 成 subwords，要么一起 mask 所有 subwords，要么都不 mask。在 500-sentence corpus 上测量它是否提高 MLM accuracy。
3. **Hard / 困难。** 在 public dataset 的 10,000 sentences 上训练一个 tiny（2-layer, d=64）BERT。用 `[CLS]` token fine-tune SST-2 sentiment。与 matched params 的 decoder-only baseline 对比，谁赢？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| MLM | “Masked language modeling” | 训练信号：随机把 15% tokens 替换成 `[MASK]`，预测原 token。 |
| Bidirectional | “Looks both ways” | Encoder attention 没有 causal mask，每个 position 都能看到其他所有 position。 |
| `[CLS]` | “The pooler token” | prepend 到每个 sequence 前面的 special token；它的最终 embedding 用作 sentence-level representation。 |
| `[SEP]` | “Segment separator” | 分隔 paired sequences（例如 query/doc、sentence A/B）。 |
| NSP | “Next sentence prediction” | BERT 的第二个 pretraining task；RoBERTa 证明无用，2019 年后被移除。 |
| Fine-tuning | “Adapt to a task” | Encoder 大多保持冻结；在上面训练一个 small head 适配 downstream task。 |
| Cross-encoder | “A reranker” | 同时接收 query 和 doc 作为输入、输出 relevance score 的 BERT。 |
| ModernBERT | “2024 refresh” | 用 RoPE、RMSNorm、GeGLU、alternating local/global attention 和 8K context 重建的 encoder。 |

## Further Reading / 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT，并移除 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 在相同 compute 下，replaced-token detection 优于 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — canonical encoder reference。
