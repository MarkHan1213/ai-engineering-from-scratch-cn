# GPT — Causal Language Modeling / GPT：Causal Language Modeling

> BERT 能看两边。GPT 只能看过去。Triangle mask 是现代 AI 中影响最深的一行代码。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 实现 causal mask，并解释它如何防止模型偷看未来 token
- 区分 GPT 的 parallel training 与 serial autoregressive inference
- 推导 shift-by-one next-token loss 的输入和 target 对齐方式
- 比较 greedy、temperature、top-k、top-p、min-p 和 speculative decoding 的使用场景

## The Problem / 问题

Language model 回答一个问题：给定前 `t-1` 个 tokens，token `t` 的 probability distribution 是什么？用这个信号，也就是 next-token prediction 训练后，模型就能一次生成一个 token，从而生成任意文本。

要在 whole sequence 上并行端到端训练，你需要让每个位置的预测只依赖更早的位置。否则模型会通过看答案轻松作弊。

Causal mask 做的就是这件事。它是一个上三角矩阵，其中未来位置是 `-inf`，在 softmax 之前加到 attention scores 上。Softmax 之后，这些位置权重变成 0。每个位置只能 attend 到自己和之前的位置。因为你一次把它应用到整个 sequence，所以一次 forward pass 就能得到 N 个并行 next-token predictions。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi，本质上都是 decoder-only causal transformers，核心 loop 相同。只是更大、更好的数据，以及更好的 RLHF。

## The Concept / 概念

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### The mask / Mask

给定 length 为 `N` 的 sequence，构建一个 `N × N` matrix：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前把 `M` 加到 raw attention scores 上。`exp(-inf) = 0`，所以 masked positions 贡献零权重。Attention matrix 的每一行都只是在 previous positions 上的 probability distribution。

实现成本：一次 `torch.tril()` 调用。计算时间：nanoseconds。对整个领域的影响：everything。

### Parallel training, serial inference / 并行训练，串行推理

Training：对整个 `(N, d_model)` sequence 做一次 forward pass，计算 N 个 cross-entropy losses（每个 position 一个），求和，backprop。沿 sequence 并行。这就是 GPT training 可以 scale 的原因：一个 GPU pass 里能处理 batch 中的 1M tokens。

Inference：逐 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV cache（Lesson 12）会保存 `t1…tn` 的 hidden states，避免每一步重算。但 inference 的 serial depth = output length。这就是 autoregressive tax，也是每个 LLM 的 decoding latency bottleneck。

### The loss — shift-by-one / Loss：shift-by-one

给定 tokens `[t1, t2, t3, t4]`：

- Input：`[t1, t2, t3]`
- Targets：`[t2, t3, t4]`

对每个 position `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是 whole sequence 的 cross-entropy。

你听说过的每个 transformer LM 都训练在这个 loss 上。Pre-training、fine-tuning、SFT，loss 相同，数据不同。

### Decoding strategies / 解码策略

训练完成后，sampling choices 比很多人想象的更重要。

| Method | What it does | When to use |
|--------|--------------|-------------|
| Greedy | Argmax every step | Deterministic tasks, code completion |
| Temperature | Divide logits by T, sample | Creative tasks, higher T = more diversity |
| Top-k | Sample from top-k tokens only | Kills low-probability tails |
| Top-p (nucleus) | Sample from smallest set with cumulative prob ≥ p | 2020+ default; adapts to distribution shape |
| Min-p | Keep tokens with `p > min_p * max_p` | 2024+; better at rejecting long tails than top-p |
| Speculative decoding | Draft model proposes N tokens, big model verifies | 2–3× latency reduction at same quality |

2026 年，对 open-weights models 来说，min-p + temperature 0.7 是合理默认值。Speculative decoding 是任何 production inference stack 的基本配置。

### What made the "GPT recipe" work / “GPT recipe” 为什么有效

1. **Decoder-only.** 没有 encoder overhead。每层一次 attention + FFN。
2. **Scaling.** 124M → 1.5B → 175B → trillions。Chinchilla scaling laws（Lesson 13）告诉你如何花 compute。
3. **In-context learning.** 在约 6B–13B 出现。模型无需 fine-tuning 就能跟随 few-shot examples。
4. **RLHF.** 在 human preferences 上 post-training，把 raw pretrained text 转成 chat assistants。
5. **Pre-norm + RoPE + SwiGLU.** 让大规模训练稳定。

核心 architecture 从 GPT-2 以来变化不大。真正有趣的变化发生在 data、scale 和 post-training。

```figure
causal-mask
```

## Build It / 动手构建

### Step 1: the causal mask / 第 1 步：causal mask

见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 前把它加到 attention scores 上。这就是完整机制。

### Step 2: a 2-layer GPT-ish model / 第 2 步：一个 2-layer GPT-ish model

堆两个 decoder blocks（masked self-attention + FFN，没有 cross-attention）。添加 token embedding、positional encoding 和 unembedding（与 token embedding matrix tied，这是 GPT-2 以来的标准技巧）。

### Step 3: next-token prediction, end-to-end / 第 3 步：端到端 next-token prediction

在 20-token toy vocab 上，对每个 position 生成 logits。对 shift-by-one target 计算 cross-entropy loss。不做 gradient，这里是 forward-pass sanity check。

### Step 4: sampling / 第 4 步：sampling

实现 greedy、temperature、top-k、top-p、min-p。在固定 prompt 上运行每种方法并比较 outputs。一个 sampling function 大约 10 行。

## Use It / 应用它

PyTorch，2026 年惯用写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底层，`generate()` 会运行 forward pass，取 final-position logits，sample 下一个 token，append，再重复。每个 production LLM inference stack（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现同一个 loop，只是做了 heavy optimization：batched prefill、continuous batching、KV cache paging、speculative decoding。

**GPT vs BERT, one line each / GPT 与 BERT 各一句：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。Loss 决定模型能否生成。

## Ship It / 交付它

见 `outputs/skill-sampling-tuner.md`。这个 skill 会为新的 generation task 选择 sampling parameters，并标记何时必须使用 deterministic decoding。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`，验证 softmax 之后的 causal attention matrix 是 lower-triangular。抽查：row 3 应该只在 columns 0–3 上有 weights。
2. **Medium / 中等。** 实现 width 4 的 beam search。比较 10 个短 prompts 上 beam-4 与 greedy 的 perplexity。Beam 是否总是赢？（提示：通常在 translation 上是，但 open-ended chat 上不是。）
3. **Hard / 困难。** 实现 speculative decoding：用 tiny 2-layer model 作为 draft，用 6-layer model 作为 verifier。在 100 个 length 64 completions 上测量 wall-clock speedup。确认 outputs 与 verifier 的 greedy 一致。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Causal mask | “The triangle” | 添加到 attention scores 的上三角 `-inf` matrix，让 position `i` 只能看到 positions `≤ i`。 |
| Next-token prediction | “The loss” | 模型 distribution 与真实 next token 在每个 position 上的 cross-entropy。 |
| Autoregressive | “Generate one at a time” | 把输出再喂回输入；parallelism 只存在于训练中，不存在于 generation 中。 |
| Logits | “Pre-softmax scores” | LM head 在 softmax 之前的 raw output；sampling 在这些值上发生。 |
| Temperature | “Creativity knob” | 用 T 除 logits；T→0 = greedy，T→∞ = uniform。 |
| Top-p | “Nucleus sampling” | 截断到 cumulative sum ≥p 的最小集合，并从剩余 token 中 sample。 |
| Min-p | “Better than top-p” | 保留 `p ≥ min_p × max_p` 的 tokens；cutoff 会适应 distribution sharpness。 |
| Speculative decoding | “Draft + verify” | Cheap model 提议 N 个 tokens；big model 并行验证。 |
| Teacher forcing | “Training trick” | 训练时喂入真实 previous token，而不是模型预测。每个 seq2seq LM 的标准做法。 |

## Further Reading / 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 与 in-context learning。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — speculative decoding 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — canonical causal-LM reference code。
