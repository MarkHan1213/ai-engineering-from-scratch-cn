# Speech Recognition (ASR) — CTC, RNN-T, Attention / 语音识别（ASR）：CTC、RNN-T 与注意力

> 语音识别是在每个 timestep 上做 audio classification，再用一个懂英语和静音的 sequence model 把结果粘起来。CTC、RNN-T 和 attention 是三种做法。选一个，并理解为什么。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 区分 CTC、RNN-T 和 attention encoder-decoder 的建模假设、延迟特性和部署取舍
- 手写 greedy CTC decode、概念版 beam-search CTC 和 word-level WER
- 解释 blank token、alignment marginalization、LM fusion 和 VAD 在 ASR 中的作用
- 为离线、多语言、流式、边缘和领域专用场景选择 2026 年常用 ASR stack

## The Problem / 问题

你有一段 10 秒、16 kHz 的音频，想得到字符串：“turn on the kitchen lights”。挑战是结构性的：音频 frames 和字符并不是一一对齐的。单词 “okay” 可能持续 200 ms，也可能持续 1200 ms。静音会切分话语。有些音素比其他音素更长。输出 token 数也无法提前知道。

有三种 formulation 可以解决：

1. **CTC (Connectionist Temporal Classification)。** 在每个 frame 上输出包含特殊 *blank* 的 token probabilities。decode 时折叠重复项并移除 blanks。非自回归、速度快。wav2vec 2.0、MMS 都使用它。
2. **RNN-T (Recurrent Neural Network Transducer)。** Joint network 基于 encoder frame 和 previous tokens 预测下一个 token。可流式。Google on-device ASR、NVIDIA Parakeet 使用它。
3. **Attention encoder-decoder。** Encoder 把音频压缩成 hidden states，decoder 通过 cross-attention 自回归生成 tokens。Whisper、SeamlessM4T 使用它。

2026 年，LibriSpeech test-clean 上的 SOTA WER 是 1.4%（Parakeet-TDT-1.1B, NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差距很小；部署差异很大。

## The Concept / 概念

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC intuition / CTC 直觉。** 让 encoder 输出 `T` 个 frame-level distributions，词表为 `V+1` 个 tokens（V 个字符 + blank）。对于长度为 `U < T` 的目标字符串 `y`，任何能 collapse 到 `y` 的 frame alignment 都算数。CTC loss 会对所有这类 alignments 求和。推理时：逐 frame argmax，折叠重复 token，移除 blanks。

优势：非自回归、可流式、zero lookahead。缺点：*conditional independence assumption*，也就是每个 frame prediction 彼此独立，因此没有内部 language model。常用外部 LM 通过 beam search 或 shallow fusion 修补。

**RNN-T intuition / RNN-T 直觉。** 增加一个 *predictor* network，用来 embedding token history；再加一个 *joiner*，把 predictor state 和 encoder frame 合成对 `V+1` 的 joint distribution（这里 `+1` 是 null / no-emit）。它显式建模了 CTC 忽略的条件依赖。因为每一步只依赖过去 frames 和过去 tokens，所以可以流式。

优势：可流式 + 内部 LM。缺点：训练更复杂、更耗内存（3D loss lattice）；RNN-T loss kernels 本身就是一整个库类别。

**Attention encoder-decoder / 注意力 encoder-decoder。** Encoder（6–32 层 transformer）处理 log-mel frames。Decoder（6–32 层 transformer）cross-attend 到 encoder outputs，自回归生成 tokens。没有 alignment 约束，attention 可以看音频中的任意位置。除非限制 attention（chunked Whisper-Streaming，2024），否则不可流式。

优势：离线 ASR 质量最高，用标准 seq2seq 工具链训练也容易。缺点：自回归延迟与输出长度成正比；没有工程改造就不能 streaming。

### WER: the one number / WER：那个核心数字

**Word Error Rate** = `(S + D + I) / N`，其中 S=substitutions，D=deletions，I=insertions，N=reference word count。它等价于 word level 的 Levenshtein edit distance。越低越好。WER 超过 20% 通常不可用；低于 5% 在朗读语音上接近 human-parity。标准 benchmark 上的 2026 年数字：

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

这些模型都基于 encoder-decoder 或 RNN-T。纯 CTC 系统（wav2vec 2.0）在 test-clean 上大约是 1.8–2.1%。

## Build It / 动手构建

### Step 1: greedy CTC decode / 第 1 步：greedy CTC decode

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续重复项，丢弃 blanks。例如：`a a _ _ a b b _ c` → `a a b c`。

### Step 2: beam-search CTC / 第 2 步：beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产系统会使用 prefix tree beam search，并融合 LM；这里保留概念骨架。

### Step 3: WER / 第 3 步：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Step 4: inference against Whisper / 第 4 步：用 Whisper 做推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

这是 2026 年最强通用 ASR 的一行调用。在 24 GB GPU 上速度约为 20× realtime。

### Step 5: streaming with Parakeet or wav2vec 2.0 / 第 5 步：用 Parakeet 或 wav2vec 2.0 做流式识别

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Streaming ASR 需要 chunked encoder attention 和 carryover state；应使用支持这些机制的库（Parakeet 用 NeMo，`transformers` pipeline 可用 `chunk_length_s`）。

## Use It / 应用它

2026 年的 stack：

| Situation | Pick |
|-----------|------|
| English, offline, max quality | Whisper-large-v3-turbo |
| Multilingual, robust | SeamlessM4T v2 |
| Streaming, low latency | Parakeet-TDT-1.1B or Riva |
| Edge, mobile, <500 ms latency | Whisper-Tiny quantized or Moonshine (2024) |
| Long-form | Whisper with VAD-based chunking (WhisperX) |
| Domain-specific (medical, legal) | Fine-tune wav2vec 2.0 + domain LM fusion |

## Pitfalls that still ship in 2026 / 2026 年仍会上线的坑

- **No VAD / 没有 VAD。** 让 Whisper 跑静音会产生 hallucinations（“Thanks for watching!”）。一定要先用 VAD gate。
- **Character vs word vs subword WER / 字符、词、subword 级 WER 混用。** 报告 word-level WER，并且在 normalization（小写、去标点）之后计算。
- **Language ID drift / 语言识别漂移。** Whisper 的 auto LID 会把噪声片段错误路由到 Japanese 或 Welsh；如果你知道语言，强制 `language="en"`。
- **Long clips without chunking / 长音频不切块。** Whisper 有 30 秒窗口。超过这个长度要用 `chunk_length_s=30, stride=5`。

## Ship It / 交付它

保存为 `outputs/skill-asr-picker.md`。为给定部署目标选择 model、decoding strategy、chunking 和 LM fusion。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会 greedy decode 一个手写 CTC 输出，并对 reference 计算 WER。
2. **Medium / 中等。** 正确实现第 2 步中的 prefix-tree beam search（考虑 blank merge rule）。在 10 个 synthetic examples 上和 greedy 比较。
3. **Hard / 困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。计算前 100 条 utterances 的 WER。与公开数字比较。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| CTC | blank-token loss | 对所有 frame-to-token alignments 做 marginalization；non-AR。 |
| RNN-T | streaming loss | CTC + next-token predictor；能处理 word-order。 |
| Attention enc-dec | Whisper-style | Encoder + cross-attending decoder；离线质量最佳。 |
| WER | 你要报告的数字 | word level 的 `(S+D+I)/N`。 |
| Blank | “空”的 token | CTC 中表示“此 frame 不发射”的特殊 token。 |
| LM fusion | 外部 language model | beam search 时加入加权 LM log-probs。 |
| VAD | 静音门控 | Voice activity detector；裁掉非语音。 |

## Further Reading / 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年 canonical paper；v3-turbo 扩展出现在 2024 年。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 Open ASR Leaderboard 领先模型。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 覆盖 25+ 模型的 live benchmark。
