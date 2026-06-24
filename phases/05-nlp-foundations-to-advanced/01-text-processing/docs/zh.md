# Text Processing — Tokenization, Stemming, Lemmatization / 文本处理：分词、词干提取与词形还原

> 语言是连续的，模型是离散的。预处理就是两者之间的桥。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 2 · 14 (Naive Bayes)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 tokenization、stemming 和 lemmatization 的职责与常见失败模式
- 从零实现一个 regex tokenizer、Porter step 1a stemmer 和查表式 lemmatizer
- 对比 NLTK 与 spaCy 在预处理 pipeline 中的取舍
- 识别训练与推理预处理不一致、库版本漂移带来的生产风险

## The Problem / 问题

模型读不懂 "The cats were running."，它读到的是整数。

每个 NLP 系统一开始都会面对同样三个问题：词从哪里开始？词根是什么？什么时候应该把 "run"、"running"、"ran" 当成同一个词，什么时候又必须保留它们的差异？

分词做错了，模型就会从垃圾输入里学习。如果 tokenizer 把 `don't` 当成一个 token，而另一个环境把它切成 `do n't` 两个 token，训练分布就被拆开了。如果 stemmer 把 `organization` 和 `organ` 压成同一个词干，topic modeling 会直接失真。如果 lemmatizer 需要词性上下文，而你没有传进去，动词就会被当成名词处理。

这一课会从零构建三个预处理步骤，然后展示 NLTK 和 spaCy 如何完成同样的工作，让你看清它们的取舍。

## The Concept / 概念

三个操作，各自有明确职责，也各自有失败模式。

**Tokenization / 分词** 把字符串切成 token。“Token” 这个词刻意保持宽泛，因为正确粒度取决于任务。经典 NLP 常用 word-level。Transformer 常用 subword。没有空格分隔的语言可能使用 character 级粒度。

**Stemming / 词干提取** 用规则砍掉后缀。快、激进、粗糙。`running -> run`。`organization -> organ`。第二个例子就是它的失败模式。

**Lemmatization / 词形还原** 利用语法知识把词还原成词典形式。更慢、更准确，需要查表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：如果速度重要，并且能接受噪声，就用 stemming（搜索索引、粗粒度分类）。如果语义重要，就用 lemmatization（问答、语义搜索、任何会展示给用户看的内容）。

```figure
edit-distance
```

## Build It / 动手构建

### Step 1: a regex word tokenizer / 第 1 步：一个 regex 词级 tokenizer

最简单可用的 tokenizer 会按非字母数字字符切分，同时把标点保留为独立 token。它不完美，也不是最终方案，但一行就能跑起来。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

这里有三个按优先级排列的 pattern：带可选内部撇号的单词（`don't`、`it's`）、纯数字、任何单个非空白且非字母数字字符（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

注意它的失败模式。`3pm` 会被切成 `['3', 'pm']`，因为 pattern 在字母串和数字串之间做了交替匹配。对多数任务来说够用。URL、email、hashtag 都会坏掉。生产环境里，要把这些 pattern 放在更通用的 pattern 前面。

### Step 2: a Porter stemmer (step 1a only) / 第 2 步：一个 Porter stemmer（只做 step 1a）

完整 Porter 算法有五个阶段的规则。只实现 step 1a 就能覆盖最常见的英文后缀，也足够看懂这种规则系统的写法。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

规则要从上往下读。`ies -> i` 这条规则就是为什么 `ponies -> poni`，而不是 `pony`。真实的 Porter 后续 step 1b 会修复它。规则之间会竞争，前面的规则获胜。顺序比任何单条规则都重要。

### Step 3: a lookup-based lemmatizer / 第 3 步：基于查表的 lemmatizer

真正的 lemmatization 需要形态学。一个适合教学的版本可以用小型 lemma 表，再加一个 fallback。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个例子是关键教学点。`watched` 不在表里，而我们的 fallback 只处理 `ing`。真实 lemmatization 会覆盖 `ed`、不规则动词、比较级形容词、发生音变的复数（`children -> child`）。这就是生产系统会使用 WordNet、spaCy morphologizer 或完整形态分析器的原因。

### Step 4: pipe them together / 第 4 步：把它们串起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的一块是 POS tagger。Phase 5 · 07 (POS Tagging) 会构建一个。现在先默认所有词都是 `NOUN`，同时明确承认这个限制。

## Use It / 应用它

NLTK 和 spaCy 都提供了生产可用版本，各自只需要几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 会处理 contraction、Unicode 和你的 regex 漏掉的边界情况。`PorterStemmer` 会跑完五个阶段。`WordNetLemmatizer` 需要把 NLTK 的 Penn Treebank 词性标注转换成 WordNet 的缩写集合。上面那段转换胶水代码，正是大多数教程会跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整条 pipeline 藏在 `nlp(text)` 后面。Tokenization、POS tagging 和 lemmatization 都会运行。大规模处理时比 NLTK 更快，开箱准确率也更高。代价是你不太容易替换单个组件。

### When to pick which / 如何选择

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| 教学、研究、替换单个组件 | NLTK |
| 生产、多语言、速度重要 | spaCy |
| Transformer pipeline（反正会用模型自带 tokenizer） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### The two failure modes nobody warns you about / 几乎没人提醒你的两个失败模式

多数教程讲完算法就停了。真实预处理 pipeline 有两个坑，几乎从不出现在入门资料里。

**Reproducibility drift / 可复现性漂移。** NLTK 和 spaCy 会在版本之间改变 tokenization 和 lemmatizer 行为。spaCy 2.x 中产生 `['do', "n't"]` 的输入，在 3.x 中可能变成 `["don't"]`。你的模型是在一种分布上训练的，推理却跑在另一种分布上。准确率悄悄下降，而且没人知道原因。把库版本固定在 `requirements.txt` 里。写一个预处理回归测试，冻结 20 个样例句子的期望分词结果。每次升级都跑。

**Training / inference mismatch / 训练与推理不一致。** 训练时做了激进预处理（小写化、移除 stopword、stemming），部署时却直接喂原始用户输入，性能会断崖式下跌。这是生产 NLP 最常见的失败。如果训练时做了预处理，推理时必须运行完全相同的函数。把预处理作为模型包里的函数发布，而不是让 serving 团队照着 notebook cell 重写。

## Ship It / 交付它

一个可复用 prompt，帮助工程师不用读三本教材也能选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## Exercises / 练习

1. **Easy / 简单。** 扩展 `tokenize`，让 URL 保持为单个 token。测试：`tokenize("Visit https://example.com today.")` 应该产生一个 URL token。
2. **Medium / 中等。** 实现 Porter step 1b。如果一个词包含元音，并且以 `ed` 或 `ing` 结尾，就移除该后缀。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **Hard / 困难。** 构建一个 lemmatizer：优先用 WordNet 作为查表来源，当 WordNet 没有条目时 fallback 到你的 Porter stemmer。在一个带词性标注的 corpus 上，分别与纯 WordNet 和纯 Porter 比较准确率。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Token | 一个词 | 模型消费的任意单位。可以是 word、subword、character 或 byte。 |
| Stem | 词根 | 基于规则剥离后缀得到的结果。不一定是真实单词。 |
| Lemma | 词典形式 | 你会在词典里查到的形式。正确计算需要语法上下文。 |
| POS tag | 词性 | NOUN、VERB、ADJ 这类类别。准确 lemmatization 需要它。 |
| Morphology | 词形变化规则 | 词如何根据时态、数、格改变形式。Lemmatization 依赖它。 |

## Further Reading / 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，至今仍是最清楚的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实 pipeline 如何串起来。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的 tokenization 边界情况。
