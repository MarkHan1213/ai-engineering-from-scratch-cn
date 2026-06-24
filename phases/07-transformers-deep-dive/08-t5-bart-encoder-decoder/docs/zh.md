# T5, BART — Encoder-Decoder Models / T5、BART：Encoder-Decoder Models

> Encoder 理解。Decoder 生成。把它们重新放在一起，你就得到适合 input → output 任务的模型：translate、summarize、rewrite、transcribe。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 解释 encoder-decoder 如何通过 cross-attention 把 source representation 提供给 decoder
- 实现 T5-style span corruption，并理解 sentinel token target format
- 比较 T5 span corruption 与 BART multi-noise denoising 的训练信号
- 根据 input-output 结构、latency 和 quality target 判断是否选择 encoder-decoder

## The Problem / 问题

Decoder-only GPT 和 encoder-only BERT 分别为了不同目标简化了 2017 年架构。但许多任务天然是 input-output：

- Translation：English → French。
- Summarization：5,000-token article → 200-token summary。
- Speech recognition：audio tokens → text tokens。
- Structured extraction：prose → JSON。

对这些任务，encoder-decoder 是最干净的匹配。Encoder 生成 source 的 dense representation。Decoder 生成 output，并在每一步 cross-attend 到这个 representation。训练在 output side 做 shift-by-one。Loss 与 GPT 相同，只是额外 conditioned on encoder output。

两篇论文定义了现代 playbook：

1. **T5**（Raffel et al. 2019）。"Text-to-Text Transfer Transformer." 把每个 NLP task 都改写成 text-in、text-out。单一 architecture、单一 vocabulary、单一 loss。Pretraining 使用 masked span prediction（corrupt input 中的 spans，再在 output 中 decode 它们）。
2. **BART**（Lewis et al. 2019）。"Bidirectional and Auto-Regressive Transformer." Denoising autoencoder：用多种方式 corrupt input（shuffle、mask、delete、rotate），让 decoder reconstruct 原文。

到 2026 年，encoder-decoder 格式仍留在 input structure 很重要的地方：

- Whisper（speech → text）。
- Google 的 translation stack。
- 一些具有 distinct context-and-edit structures 的 code-completion / repair models。
- Flan-T5 及其用于 structured reasoning tasks 的变体。

Decoder-only 赢得了聚光灯，但 encoder-decoder 从未消失。

## The Concept / 概念

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### The forward loop / Forward loop

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键在于 encoder 每个 input 只运行一次。Decoder autoregressively 运行，但每一步都 cross-attend 到*同一个* encoder output。对 long inputs 来说，缓存 encoder output 是免费的 speedup。

### T5 pretraining — span corruption / T5 预训练：span corruption

随机选择 input 中的一些 spans（平均长度 3 tokens，总计 15%）。每个 span 替换成一个唯一 sentinel：`<extra_id_0>`、`<extra_id_1>` 等。Decoder 只输出被 corrupt 的 spans，并以对应 sentinel 开头：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

这比预测整个 sequence 更便宜。在 T5 论文 ablation 中，它与 MLM（BERT）和 prefix-LM（UniLM）有竞争力。

### BART pretraining — multi-noise denoising / BART 预训练：多噪声 denoising

BART 尝试五种 noising functions：

1. Token masking。
2. Token deletion。
3. Text infilling（mask 一个 span，让 decoder 插入正确长度）。
4. Sentence permutation。
5. Document rotation。

组合 text infilling + sentence permutation 会得到最佳 downstream numbers。Decoder 总是 reconstruct 原始文本。BART 的 output 是完整 sequence，而不只是 corrupted spans，所以 pretraining compute 比 T5 更高。

### Inference / 推理

与 GPT 相同，都是 autoregressive generation。Greedy / beam / top-p sampling 都适用。Beam search（width 4–5）是 translation 和 summarization 的标准选择，因为 output distribution 比 chat 更窄。

### When to pick each variant in 2026 / 2026 年何时选择各变体

| Task | Encoder-decoder? | Why |
|------|------------------|-----|
| Translation | Yes, usually | Clear source sequence; fixed output distribution; beam search works |
| Speech-to-text | Yes (Whisper) | Input modality differs from output; encoder shapes audio features |
| Chat / reasoning | No, decoder-only | No persistent "input" — the conversation is the sequence |
| Code completion | Usually no | Decoder-only with long context wins; code models like Qwen 2.5 Coder are decoder-only |
| Summarization | Either works | BART, PEGASUS beat earlier decoder-only baselines; modern decoder-only LLMs match them |
| Structured extraction | Either | T5 is clean because "text → text" absorbs any output format |

约 2022 年之后的趋势是：decoder-only 接管了过去 encoder-decoder 拥有的很多任务，因为 (a) instruction-tuned decoder-only LLMs 能通过 prompting 泛化到任何任务，(b) 单一 architecture 更容易 scale，(c) RLHF 假设的是 decoder。Encoder-decoder 留在 input modality 不同（speech、images）或 beam search quality 很重要的地方。

## Build It / 动手构建

见 `code/main.py`。我们会为 toy corpus 实现 T5-style span corruption。这是本课最有用的单个组件，因为它出现在此后几乎所有 encoder-decoder pretraining recipe 里。

### Step 1: span corruption / 第 1 步：span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

Target format 是 T5 约定：`<sent0> span0 <sent1> span1 ...`。Corrupted input 会把 unchanged tokens 与 span locations 上的 sentinel tokens 交错起来。

### Step 2: verify round-trip / 第 2 步：验证 round-trip

给定 corrupted input 和 target，reconstruct 原始句子。如果 corruption 可逆，forward pass 就定义清楚了。这是 sanity check；真实 training 不会这么做，但这个测试很便宜，也能抓住 span bookkeeping 中的 off-by-one bugs。

### Step 3: BART noising / 第 3 步：BART noising

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## Use It / 应用它

HuggingFace reference：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 trick：task name 放进 input text。同一个模型能处理几十种任务，因为每个任务都是 text-in、text-out。到 2026 年，这个 pattern 已被 instruction-tuned decoder-only models 泛化，但 T5 最先把它 codify。

## Ship It / 交付它

见 `outputs/skill-seq2seq-picker.md`。这个 skill 会根据 input-output structure、latency 和 quality targets，在 encoder-decoder 与 decoder-only 之间做选择。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`，对一个 30-token sentence 应用 span corruption，验证把 non-sentinel source tokens 与 decoded target spans 拼回去能复原原文。
2. **Medium / 中等。** 实现 BART 的 `text_infill` noise：用一个 `<mask>` token 替换 random spans，decoder 必须推断正确 span length 与内容。展示一个例子。
3. **Hard / 困难。** 在 tiny English → pig-Latin corpus（200 pairs）上 fine-tune `flan-t5-small`。在 held-out 50-pair set 上测 BLEU。与相同 compute 下 fine-tune `Llama-3.2-1B` 对比。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Encoder-decoder | “Seq2seq transformer” | 两个 stacks：用于输入的 bidirectional encoder，以及带 cross-attention 的 causal decoder。 |
| Cross-attention | “Where source talks to target” | Decoder 的 Q × encoder 的 K/V。Encoder 信息进入 decoder 的唯一位置。 |
| Span corruption | “T5's pretraining trick” | 用 sentinel tokens 替换 random spans；decoder 输出这些 spans。 |
| Denoising objective | “BART's game” | 对 input 应用 noise function，训练 decoder reconstruct clean sequence。 |
| Sentinel token | “The `<extra_id_N>` placeholder” | 标记 source 中 corrupted spans，并在 target 中重新标记它们的 special tokens。 |
| Flan | “Instruction-tuned T5” | 在 >1,800 个 tasks 上 fine-tuned 的 T5；让 encoder-decoder 在 instruction-following 上仍有竞争力。 |
| Beam search | “Decoding strategy” | 每一步保留 top-k partial sequences；translation/summarization 的标准做法。 |
| Teacher forcing | “Training-time input” | 训练时喂给 decoder 真实 previous output token，而不是 sampled token。 |

## Further Reading / 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年 canonical encoder-decoder。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — reference implementation。
