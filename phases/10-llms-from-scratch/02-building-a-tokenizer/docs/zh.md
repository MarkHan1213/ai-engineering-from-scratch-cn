# Building a Tokenizer from Scratch / 从零构建 Tokenizer

> Lesson 01 给了你一个玩具。本课给你一件真正能用的工具。

**类型：** Build
**语言：** Python
**前置基础：** Phase 10, Lesson 01（Tokenizers: BPE, WordPiece, SentencePiece）
**时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 构建一个生产级 BPE tokenizer，能够处理 Unicode、空白规范化和 special tokens
- 实现字节级 fallback，使 tokenizer 能编码任意输入，包括 emoji、CJK 和代码，而不产生 unknown token
- 添加预分词 regex pattern，在应用 BPE merge 之前按词边界切分文本
- 在语料上训练自定义 tokenizer，并在多语言文本上与 `tiktoken` 比较压缩率

## The Problem / 问题

Lesson 01 里的 BPE tokenizer 可以处理英文文本。现在把日文扔进去，或者 emoji，或者混合 tabs 和 spaces 的 Python 代码。

它会坏掉。

不是因为 BPE 错了，而是因为实现不完整。生产 tokenizer 要处理任意编码的原始字节，在切分前规范化 Unicode，管理永远不能被 merge 的 special tokens，把预分词和子词切分串起来，并且速度要足够快，不能拖慢一个正在处理 15 万亿 tokens 的训练流水线。

GPT-2 的 tokenizer 有 50,257 个 token。Llama 3 有 128,256 个。GPT-4 约 100,000 个。这些不是玩具数字。支撑这些词表的 merge tables 是在数百 GB 文本上训练出来的；而外围机制，包括 normalization、pre-tokenization、special token 注入、chat template 格式化，决定了一个 tokenizer 是只能处理 `"hello world"`，还是能处理整个互联网。

你要构建的正是这套机制。

## The Concept / 概念

### The Full Pipeline / 完整流水线

生产 tokenizer 不是一个算法，而是一条五阶段流水线，每一阶段解决不同问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段都有明确职责：

| Stage | What It Does | Why It Matters |
|-------|-------------|----------------|
| Normalize | NFKC Unicode、可选 lowercase、可选 strip accents | `"fi"` 连字（U+FB01）变成 `"fi"` 两个字符。没有这一步，同一个词会得到不同 token。 |
| Pre-Tokenize | 在 BPE 前把文本切成 chunk | 防止 BPE 跨词边界 merge。`"the cat"` 不应该生成 `"e c"` 这种 token。 |
| BPE Merge | 对字节序列应用学到的 merge 规则 | 核心压缩步骤。把原始字节变成子词 token。 |
| Special Tokens | 注入 [BOS]、[EOS]、[PAD]、chat template markers | 这些 token 有固定 ID，永远不参与 BPE merge；模型用它们表示结构。 |
| ID Mapping | 把 token 字符串转成整数 ID | 模型看到的是整数，不是字符串。 |

### Byte-Level BPE / 字节级 BPE

Lesson 01 的 tokenizer 已经运行在 UTF-8 字节上，这是正确方向。但我们跳过了一个关键问题：如果这些字节本身不是合法 UTF-8，会发生什么？

字节级 BPE 把每个可能的字节值（0-255）都视为合法 token。因此基础词表恰好是 256 个条目。任何文件，无论是文本、二进制还是损坏内容，都可以被 tokenized，而不会产生 unknown token。

GPT-2 加了一个技巧：把每个字节映射到一个可打印 Unicode 字符，让词表更易读。字节 `0x20`（空格）在它们的映射里会变成字符 `"G"`。这只是展示层面的处理；算法本身并不关心。

真正强大的地方在于：字节级 BPE 能处理地球上所有语言。中文字符通常是 3 个 UTF-8 字节。日文可能是 3-4 个字节。阿拉伯文、天城文、emoji 都只是字节序列。BPE 在这些字节序列中寻找模式，方式和它在英文 ASCII 字节中寻找模式完全一样。

### Pre-Tokenization / 预分词

在 BPE 接触文本之前，你需要先把文本切成 chunk。这能防止 merge 算法创建跨越词边界的 token。

GPT-2 使用一个 regex pattern 来切分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个 pattern 会切分 contractions（`"don't"` 变成 `"don"` + `"'t"`）、带可选前导空格的单词、数字、标点和空白。前导空格会附着在单词上，所以 `"the cat"` 变成 `[" the", " cat"]`，而不是 `["the", " ", "cat"]`。

Llama 使用 SentencePiece，完全跳过 regex。它把原始字节流当作一个长序列，让 BPE 自己推断边界。这更简单，但也给了 BPE 更多自由，可能学出跨词 token。

这个选择很重要。GPT-2 的 regex 会阻止 tokenizer 学到某个词末尾的 `"the"` 和下一个词开头的 `"the"` 应该 merge。SentencePiece 允许这类 merge，有时压缩更高效，但 token 可解释性更差。

### Special Tokens / 特殊 Token

每个生产 tokenizer 都会为结构标记保留 token ID：

| Token | Purpose | Used By |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列开始 | Llama 3, GPT |
| `[EOS]` / `</s>` | 序列结束 | All models |
| `[PAD]` | batch 对齐 padding | BERT, T5 |
| `[UNK]` | unknown token（字节级 BPE 消除了它） | BERT, WordPiece |
| `<\|im_start\|>` | chat 消息边界开始 | ChatGPT, Qwen |
| `<\|im_end\|>` | chat 消息边界结束 | ChatGPT, Qwen |
| `<\|user\|>` | 用户轮次标记 | Llama 3 |
| `<\|assistant\|>` | assistant 轮次标记 | Llama 3 |

Special tokens 永远不会被 BPE 切开。它们会在 merge 算法运行前被精确匹配，替换成固定 ID；周围文本再按正常方式 tokenized。

### Chat Templates / Chat 模板

这是最容易让人困惑，也最容易让实现出错的地方。

当你向 chat model 发送消息时，API 接收的是消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看到的不是 JSON。它看到的是一个扁平 token 序列。chat template 使用 special tokens 把消息转换成这个扁平序列。每个模型的格式都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

模板一旦写错，模型输出就会崩。它是在某个精确格式上训练的。任何偏差，比如少一个换行、token 顺序反了、多一个空格，都会把输入推到训练分布之外。

### Speed / 速度

Python 对生产 tokenization 来说太慢。

`tiktoken`（OpenAI）用 Rust 编写并提供 Python 绑定。HuggingFace `tokenizers` 也是 Rust。SentencePiece 是 C++。这些实现比纯 Python 快 10-100 倍。

给一个直观量级：Llama 3 预训练需要 tokenizing 15 万亿 tokens。如果速度是每秒 100 万 tokens（已经算快的 Python），需要 174 天；如果是每秒 1 亿 tokens（Rust），只需要 1.7 天。

本课用 Python 是为了理解算法。在生产环境中，你会使用编译实现，只在 Python wrapper 层交互。

```figure
weight-tying
```

## Build It / 动手构建

### Step 1: Byte-Level Encoding / 步骤 1：字节级编码

地基是：把任意字符串转成字节序列，为展示把每个字节映射到可打印字符，并能反向恢复。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

用多语言文本测试字节数量：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

`"hello"` 是 5 个字节。`"你好"` 是 6 个字节，每个字符 3 个。火焰 emoji 是 4 个字节。字节级 tokenizer 不关心语言是什么，bytes are bytes。

### Step 2: Pre-Tokenizer with Regex / 步骤 2：用 Regex 做预分词

使用 GPT-2 的 regex pattern 把文本切成 chunk。每个 chunk 独立交给 BPE。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode property escapes，比如 `\p{L}` 表示字母，`\p{N}` 表示数字。标准库 `re` 不支持，所以我们 fallback 到 ASCII 字符类。生产级多语言 tokenizer 应该安装 `regex`。

试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格附着在词上。contractions 在 apostrophe 处分开。标点成为自己的 chunk。BPE 永远不会跨这些边界 merge。

### Step 3: BPE on Byte Sequences / 步骤 3：在字节序列上运行 BPE

沿用 Lesson 01 的核心算法，但现在对每个预分词 chunk 独立运行。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Step 4: Special Token Handling / 步骤 4：处理 Special Token

Special tokens 需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Step 5: Full Tokenizer Class / 步骤 5：完整 Tokenizer 类

把所有阶段串起来：normalize、按 special tokens 切分、pre-tokenize、BPE merge、映射到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Step 6: Multilingual Test / 步骤 6：多语言测试

真正的测试：把英文、中文、emoji 和代码都扔进去。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符各产生 3 个字节。emoji 产生 4 个字节。它们都不会让 tokenizer 崩溃，也不会产生 unknown token。这就是字节级 BPE 的力量。

## Use It / 应用它

### Comparing Real Tokenizers / 比较真实 Tokenizer

加载 Llama 3、GPT-4 和 Mistral 的实际 tokenizer，观察它们如何处理同一个多语言段落。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

同一文本会得到不同 token 数。Llama 3 的 128K 词表更积极地合并常见模式。GPT-4 的 100K 处在中间。Mistral 的 32K 会产生更多 token，但 embedding 层更小。

取舍始终一样：更大词表意味着更短序列，但参数更多。

## Ship It / 交付它

本课会产出一个用于构建和调试生产 tokenizer 的 prompt，见 `outputs/prompt-tokenizer-builder.md`。

## Exercises / 练习

1. **Easy:** 添加一个 `get_token_bytes(id)` 方法，展示任意 token ID 对应的原始字节。用它检查最常见的 merged tokens 实际代表什么。
2. **Medium:** 实现 Llama 风格的 pre-tokenizer：按空白和数字切分，但保留前导空格。在同一语料上比较它与 GPT-2 regex 方案得到的词表。
3. **Hard:** 添加一个 chat template 方法，接收 `{"role": ..., "content": ...}` 消息列表，并按 Llama 3 chat format 生成正确 token 序列。与 HuggingFace 实现做对照测试。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Byte-level BPE | “在字节上工作的 tokenizer” | 基础词表包含 256 个字节值的 BPE；能处理任意输入而无需 unknown token |
| Pre-tokenization | “BPE 前先切分” | 基于 regex 或规则的切分，防止 BPE 跨词边界 merge |
| NFKC normalization | “Unicode 清理” | 规范分解后做兼容组合；`"fi"` 连字变成 `"fi"`，全角 `"A"` 变成 `"A"` |
| Chat template | “消息如何变成 token” | 把 role/content 消息列表转成扁平 token 序列的精确格式；模型相关，必须匹配训练格式 |
| Special tokens | “控制 token” | 绕过 BPE 的保留 token ID，如 [BOS]、[EOS]、[PAD]、chat markers，在 merge 前精确匹配 |
| Fertility | “每词 token 数” | 输出 token 与输入词数的比例；GPT-4 英文约 1.3，韩文 2-3，更高表示浪费上下文 |
| tiktoken | “OpenAI tokenizer” | 带 Python 绑定的 Rust BPE 实现；比纯 Python 快 10-100 倍 |
| Merge table | “词表” | 训练期间学到的有序 byte-pair merge 列表；这就是 tokenizer 学到的知识 |

## Further Reading / 延伸阅读

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- GPT-3.5/4 使用的 Rust BPE 实现
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- 支持 BPE、WordPiece、Unigram 的 Rust tokenizer 库
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 128K 词表和 tokenizer 训练细节
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- 语言无关 tokenization
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 原始 byte-to-Unicode mapping
