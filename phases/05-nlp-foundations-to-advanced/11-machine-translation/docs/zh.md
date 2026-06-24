# Machine Translation / 机器翻译

> Translation 是为 NLP 研究买单三十年的任务，现在仍然在继续买单。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 使用 NLLB/mBART 这类 pretrained multilingual encoder-decoder 完成机器翻译
- 解释 tokenizer、model size、beam search、length penalty 对 MT 质量的影响
- 使用 BLEU、chrF、COMET 与 LLM-as-judge 评估翻译质量
- 识别 hallucination、off-target generation、terminology drift、formality mismatch 等生产失败模式

## The Problem / 问题

模型读取一种语言的句子，输出另一种语言的句子。长度会变，词序会变。有些 source words 会映射到多个 target words，反过来也一样。习语拒绝一对一映射。"I miss you" 的法语是 "tu me manques"，字面意思是“你对我来说缺失了”。没有 word-level alignment 能撑住这种情况。

Machine translation 是逼着 NLP 发明 encoder-decoders、attention、transformers，并最终走向整个 LLM 范式的任务。每一次进步都来自翻译质量可测，而且人机差距顽固存在。

这一课不讲历史，而是讲 2026 年的工作 pipeline：pretrained multilingual encoder-decoder（NLLB-200 或 mBART）、subword tokenization、beam search、BLEU 和 chrF evaluation，以及仍然会漏进生产的一小撮失败模式。

## The Concept / 概念

![MT pipeline：tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

现代 MT 是在 parallel text 上训练的 transformer encoder-decoder。Encoder 以对应语言的 tokenization 读取 source。Decoder 通过 cross-attention（lesson 10）使用 encoder output，一次生成一个 target subword。Decoding 使用 beam search，避免 greedy-decoding 陷阱。输出会 detokenized、detruecased，并与 reference 比分。

三个操作性选择决定真实 MT 质量。

- **Tokenizer.** 在混合语言 corpus 上训练的 SentencePiece BPE。跨语言共享 vocabulary，是 NLLB 能做 zero-shot pairs 的基础。
- **Model size.** NLLB-200 distilled 600M 可以在笔记本上跑。NLLB-200 3.3B 是论文发布的生产默认。54.5B 是研究上限。
- **Decoding.** 通用内容使用 beam width 4-5。Length penalty 用来避免输出过短。需要术语一致性时使用 constrained decoding。

```figure
seq2seq-alignment
```

## Build It / 动手构建

### Step 1: a pretrained MT call / 第 1 步：调用 pretrained MT

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三件事重要。`src_lang` 告诉 tokenizer 使用哪种 script 和 segmentation。`forced_bos_token_id` 告诉 decoder 要生成哪种语言。两者都是 NLLB-specific tricks；mBART 和 M2M-100 有自己的约定，不能互换。

### Step 2: BLEU and chrF / 第 2 步：BLEU 和 chrF

BLEU 衡量 output 和 reference 的 n-gram overlap。使用 1-4 四种 reference n-gram size，计算 precision 的几何平均，再加上对过短输出的 brevity penalty。分数范围是 [0, 100]。它很常用，也很难解释：30 BLEU 是“可用”；40 是“好”；50 是“非常强”；低于 1 BLEU 的差异常常只是噪声。

chrF 衡量 character-level F-score。对形态丰富语言更敏感，因为 BLEU 会低估匹配。通常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它会规范化 tokenization，让不同论文的分数可比。自己手写 BLEU 计算，是制造误导性 benchmark 的常见方式。

### The three-tier evaluation hierarchy (2026) / 三层评估体系（2026）

现代 MT evaluation 使用三类互补指标。上线时至少带两个。

- **Heuristic**（BLEU、chrF）。快、基于 reference、可解释、不敏感于 paraphrase。用于 legacy comparison 和 regression detection。
- **Learned**（COMET、BLEURT、BERTScore）。在人工判断上训练的神经模型；比较 translation 与 source/reference 的语义相似度。自 2023 年以来，COMET 与 MT research 的相关性最高，是 2026 年质量重要场景的生产默认。
- **LLM-as-judge**（reference-free）。Prompt 大模型按 fluency、adequacy、tone、cultural appropriateness 给翻译打分。Rubric 设计好时，GPT-4-as-judge 与人类一致率约 80%。适合没有 reference 的开放内容。

实用 2026 stack：用 `sacrebleu` 算 BLEU 和 chrF，用 `unbabel-comet` 算 COMET，再用 prompted LLM 给最终面向人的信号。信任任何生产数据上的指标前，先用 50-100 个人工标注样本校准。

Reference-free metrics（COMET-QE、BLEURT-QE、LLM-as-judge）可以在没有 reference 的情况下评估翻译，这对缺少参考译文的长尾语言对很重要。

### Step 3: what breaks in production / 第 3 步：生产中会坏在哪里

上面的工作 pipeline 会在 80% 时间里流畅翻译，并在剩下 20% 里安静失败。需要点名的失败模式：

- **Hallucination / 幻觉。** 模型发明 source 中没有的内容。常见于陌生领域词汇。症状：输出很流畅，但声称了 source 没说过的事实。缓解：对领域术语做 constrained decoding，受监管内容做人审，监控输出明显长于输入的情况。
- **Off-target generation / 目标语言错误。** 模型翻译到错误语言。NLLB 在稀有语言对上很容易这样。缓解：验证 `forced_bos_token_id`，并始终用 language-ID model 检查输出。
- **Terminology drift / 术语漂移。** "Sign up" 在文档 1 中变成 "s'inscrire"，在文档 2 中变成 "créer un compte"。对 UI 文本和用户可见字符串来说，一致性比原始质量更重要。缓解：glossary-constrained decoding 或 post-edit dictionary。
- **Formality mismatch / 正式程度不匹配。** 法语 "tu" vs "vous"，日语敬语等级。模型会选训练中更常见的形式。对 customer-facing content 这通常是错的。缓解：如果模型支持，用 formality token 做 prompt prefix，或在 formal-only corpora 上 fine-tune 小模型。
- **Length explosion on short input / 短输入长度爆炸。** 很短的输入句子经常产生过长翻译，因为 source tokens 少于约 5 个时 length penalty 会失稳。缓解：用与 source length 成比例的 hard max-length cap。

### Step 4: fine-tuning for a domain / 第 4 步：面向领域微调

Pretrained models 是通才。法律、医疗或游戏对话翻译，会从领域 parallel data 微调中得到明显收益。Recipe 并不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量 parallel examples 胜过几十万条噪声网页抓取样本。训练数据质量是生产里最大的杠杆。

## Use It / 应用它

2026 年 MT 生产 stack：

| Use case / 用例 | Recommended starting point / 推荐起点 |
|---------|---------------------------|
| 任意语言互译，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 英语中心，高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短任务、低成本推理、英法/德/西 | Helsinki-NLP / Marian models |
| 延迟关键、浏览器侧 | ONNX-quantized Marian（约 50 MB） |
| 最高质量，愿意付费 | GPT-4 / Claude / Gemini with translation prompts |

到 2026 年，LLM 在若干语言对上已经超过专用 MT 模型，尤其是习语内容和长上下文。代价是 per-token 成本和延迟。当 context length、style consistency 或通过 prompting 做 domain adaptation 比吞吐更重要时，选择 LLM。

## Ship It / 交付它

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## Exercises / 练习

1. **Easy / 简单。** 使用 `nllb-200-distilled-600M` 把一段 5 句英文段落翻译成法语，再翻回英语。测量 round-trip 与原文有多接近。你会看到语义保留，但词汇选择会漂移。
2. **Medium / 中等。** 使用 `fasttext lid.176` 或 `langdetect` 实现 translation output 的 language-ID check。集成到 MT 调用里，让 off-target generation 在返回前被拦截。
3. **Hard / 困难。** 在你选择的 5,000-pair 领域 corpus 上 fine-tune `nllb-200-distilled-600M`。在 held-out set 上测量 fine-tuning 前后的 BLEU。报告哪些句子类型提升，哪些退化。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译分数 | 带 brevity penalty 的 n-gram precision。[0, 100]。 |
| chrF | Character F-score | 字符级 F-score。对形态丰富语言更敏感。 |
| NMT | Neural MT | 在 parallel text 上训练的 transformer encoder-decoder。2017+ 默认方案。 |
| NLLB | No Language Left Behind | Meta 的 200-language MT model family。 |
| Constrained decoding | 受控输出 | 强制特定 tokens 或 n-grams 在输出中出现 / 不出现。 |
| Hallucination | 发明内容 | 模型输出 source 不支持的内容。 |

## Further Reading / 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用 fine-tuning walkthrough。
