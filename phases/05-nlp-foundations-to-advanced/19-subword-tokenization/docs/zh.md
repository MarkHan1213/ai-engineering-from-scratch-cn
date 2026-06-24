# Subword Tokenization — BPE, WordPiece, Unigram, SentencePiece / Subword Tokenization：BPE、WordPiece、Unigram、SentencePiece

> Word tokenizers 会卡在未见词上。Character tokenizers 会让序列长度爆炸。Subword tokenizers 折中解决。每个现代 LLM 都基于它交付。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 BPE、Byte-level BPE、Unigram、WordPiece、SentencePiece 与 tiktoken
- 从零实现 BPE merge 训练与编码流程
- 使用 SentencePiece 训练 tokenizer，并用 tiktoken 估算 OpenAI-compatible vocab token 数
- 识别 tokenizer drift、whitespace ambiguity、多语言 undertraining 与 emoji splits 等生产问题

## The Problem / 问题

你的 vocabulary 有 50,000 个词。用户输入 "untokenizable"。Tokenizer 返回 `[UNK]`。模型现在完全没有这个词的信号。更糟的是，你 corpus 中 90th-percentile document 有 40 个 rare words，这意味着每篇文档丢掉 40 bits 信息。

Subword tokenization 解决了这个问题。常见词保持单个 token。稀有词拆成有意义的片段：`untokenizable` → `un`, `token`, `izable`。训练数据可以覆盖一切，因为任何字符串最终都是 bytes 的序列。

2026 年每个 frontier LLM 都基于三种算法之一（BPE、Unigram、WordPiece），并由三类库之一封装（tiktoken、SentencePiece、HF Tokenizers）。不选 tokenizer，就无法交付 language model。

## The Concept / 概念

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** 从 character-level vocabulary 开始。统计每个相邻 pair。把最频繁 pair 合并成新 token。重复直到达到目标 vocabulary size。主流算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**Byte-level BPE.** 同样算法，但作用在 raw bytes（256 个 base tokens）上，而不是 Unicode characters。保证没有 `[UNK]` tokens：任何 byte sequence 都能编码。GPT-2 使用 50,257 个 tokens（256 bytes + 50,000 merges + 1 special）。

**Unigram.** 从一个巨大 vocabulary 开始。给每个 token 分配 unigram probability。迭代删除那些移除后最少增加 corpus log-likelihood 的 tokens。推理时是概率式的：可以采样 tokenizations（对 subword regularization 数据增强有用）。T5、mBART、ALBERT、XLNet、Gemma 使用它。

**WordPiece.** 合并最能最大化 training corpus likelihood 的 pairs，而不是 raw frequency。BERT、DistilBERT、ELECTRA 使用它。

**SentencePiece vs tiktoken.** SentencePiece 是直接在 raw Unicode text 上 *训练* vocabularies（BPE 或 Unigram）的库，用 `▁` 编码 whitespace。tiktoken 是 OpenAI 针对预构建 vocabularies 的快速 *encoder*；它不训练。

经验法则：

- **Training a new vocabulary / 训练新 vocabulary：** SentencePiece（多语言、无需 pre-tokenization）或 HF Tokenizers。
- **Fast inference against GPT vocab / 对 GPT vocab 做快速推理：** tiktoken（cl100k_base、o200k_base）。
- **Both / 两者都要：** HF Tokenizers，一个库负责 training + serving。

```figure
bpe-merge
```

## Build It / 动手构建

### Step 1: BPE from scratch / 第 1 步：从零实现 BPE

见 `code/main.py`。循环如下：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

算法编码了三个事实。`</w>` 标记词尾，让 "low"（suffix）和 "lower"（prefix）保持不同。Frequency weighting 让高频 pairs 更早胜出。Merge list 是有序的，推理时按训练顺序应用 merges。

### Step 2: encode with the learned merges / 第 2 步：用学到的 merges 编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素实现是 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）使用 merge-rank lookup 和 priority queues，接近线性时间。

### Step 3: SentencePiece in practice / 第 3 步：实践中使用 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：不需要 pre-tokenization，space 被编码成 `▁`，`character_coverage` 控制 rare characters 是被积极保留，还是映射到 `<unk>`。

### Step 4: tiktoken for OpenAI-compatible vocabs / 第 4 步：为 OpenAI-compatible vocabs 使用 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

只做 encoding。快（Rust backend）。对 GPT-4/5 tokenization 精确匹配，适合 byte-counting、cost estimation 和 context-window budgeting。

## Pitfalls that still ship in 2026 / 2026 年仍会进生产的坑

- **Tokenizer drift.** 训练用 vocab A，部署却用 vocab B。Token IDs 不同，模型输出垃圾。在 CI 中检查 `tokenizer.json` hash。
- **Whitespace ambiguity.** BPE 中 "hello" 和 " hello" 会产生不同 tokens。始终显式指定 `add_special_tokens` 和 `add_prefix_space`。
- **Multilingual undertraining.** 英语占主导的 corpus 会产生把非拉丁 scripts 切成 5-10 倍 tokens 的 vocabularies。同一个 prompt 在 GPT-3.5 上用日语/阿拉伯语会贵 5-10 倍。o200k_base 部分修复了这个问题。
- **Emoji splits.** 单个 emoji 可能占 5 个 tokens。做 context budgeting 时要 checkpoint emoji handling。

## Use It / 应用它

2026 stack：

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| 从零训练单语言模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 服务 OpenAI-compatible API | tiktoken（GPT-4+ 用 `o200k_base`） |
| 领域特定 vocab（code、math、protein） | 在领域 corpus 上训练 custom BPE，并与 base vocab 合并 |
| Edge inference，小模型 | Unigram（小 vocabulary 通常更好） |

Vocabulary size 是 scaling decision，不是常数。粗略经验：<1B 参数用 32k，1-10B 用 50-100k，多语言/frontier 用 200k+。

## Ship It / 交付它

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Exercises / 练习

1. **Easy / 简单。** 在 `code/main.py` 的 tiny corpus 上训练 500-merge BPE。编码三个 held-out words。多少词正好产生 1 个 token，多少词产生 >1 token？
2. **Medium / 中等。** 在 100 个 English Wikipedia sentences 上比较 `cl100k_base`、`o200k_base` 和你用 vocab=32k 训练的 SentencePiece BPE 的 token counts。报告每种 compression ratio。
3. **Hard / 困难。** 用 BPE、Unigram 和 WordPiece 在同一 corpus 上训练 tokenizer。分别用于一个小型 sentiment classifier，测量 downstream accuracy。选择 tokenizer 是否让 F1 变化超过 1 个点？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | 贪心合并最频繁 character pairs，直到达到目标 vocab size。 |
| Byte-level BPE | 永远没有 unknown tokens | 在 raw 256 bytes 上做 BPE；GPT-2 / Llama 使用这种方案。 |
| Unigram | 概率式 tokenizer | 从大候选集合按 log-likelihood 剪枝；T5、Gemma 使用。 |
| SentencePiece | 处理 whitespace 的那个 | 在 raw text 上训练 BPE/Unigram 的库；space 编码为 `▁`。 |
| tiktoken | 很快的那个 | OpenAI 的 Rust-backed BPE encoder，针对预构建 vocabs。不训练。 |
| Merge list | 魔法数字 | 有序的 `(a, b) → ab` merges 列表；推理时按顺序应用。 |
| Character coverage | 多稀有才算太稀有？ | Tokenizer 必须覆盖训练 corpus 中的字符比例；典型值约 0.9995。 |

## Further Reading / 延伸阅读

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — SentencePiece 库。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简明参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — cookbook + encoding list。
