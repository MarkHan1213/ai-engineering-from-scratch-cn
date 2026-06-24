# Audio Evaluation — WER, MOS, UTMOS, MMAU, FAD, and the Open Leaderboards / 音频评估：WER、MOS、UTMOS、MMAU、FAD 与开放榜单

> 无法度量，就无法交付。本课列出 2026 年每类音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、WER-on-ASR-round-trip）、audio-language（MMAU、LongAudioBench）、music（FAD、CLAP）和 speaker（EER）。也包括你该去哪里比较的 leaderboards。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 为 ASR、TTS、voice cloning、speaker verification、diarization、classification、music generation、LALM 和 streaming S2S 选择正确指标
- 实现 WER normalization、TTS round-trip WER、SECS、FAD 和 EER 的最小评估代码
- 理解 MOS、UTMOS、MMAU-Pro、LongAudioBench、FAD、CLAP、DER、RTFx 和 latency percentiles 的含义
- 建立固定 eval harness，报告分布和分组结果，而不是只看平均值

## The Problem / 问题

每个音频任务都有多个指标，每个指标衡量不同维度。用错指标，就会交付一个 dashboard 上很好、生产里很差的模型。2026 年 canonical list：

| Task | Primary | Secondary |
|------|---------|-----------|
| ASR | WER | CER · RTFx · first-token latency |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS · CER |
| Speaker verification | EER | minDCF · FAR / FRR at operating point |
| Diarization | DER | JER · speaker confusion |
| Audio classification | top-1 · mAP | macro F1 · per-class recall |
| Music generation | FAD | CLAP · listening panel MOS |
| Audio language model | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latency P50/P95 | WER · MOS |

## The Concept / 概念

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR metrics / ASR 指标

**WER (Word Error Rate)。** `(S + D + I) / N`。打分前做 lowercase、strip punctuation、normalize numbers。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。&lt; 5% = 朗读语音上的 human-parity。

**CER (Character Error Rate)。** 同一个公式，character-level。用于普通话、粤语等 word segmentation 模糊的 tone languages。

**RTFx (inverse real-time factor)。** 每 wall-clock second 可处理多少 audio seconds。越高越好。Parakeet-TDT 可达 3380×。Whisper-large-v3 约 30×。

**First-token latency。** 从 audio input 到第一个 transcript token 的 wall-clock。Streaming 场景关键。Deepgram Nova-3：约 150 ms。

### TTS metrics / TTS 指标

**MOS (Mean Opinion Score)。** 1–5 分人工评分。Gold standard，但慢。每个 sample 收集 20+ listeners，每个 model 100+ samples。

**UTMOS (2022-2026)。** Learned MOS predictor。在标准 benchmark 上与 human MOS 相关性约 0.9。F5-TTS：UTMOS 3.95；ground truth：4.08。

**SECS (Speaker Encoder Cosine Similarity)。** 用于 voice cloning。Reference 与 cloned output 的 ECAPA embedding cosine。&gt; 0.75 = recognizable clone。

**WER-on-ASR-round-trip。** 用 Whisper 跑 TTS output，再与 input text 计算 WER。它能捕捉 intelligibility regressions。2026 SOTA：&lt; 2% CER。

**TTFA (time-to-first-audio)。** Wall-clock latency。Kokoro-82M：约 100 ms；F5-TTS：约 1 s。

### Voice-cloning-specific / Voice cloning 专用

**SECS + MOS + CER** 三联指标。SECS 高但 MOS 低，说明 timbre-right-but-unnatural；反过来说明声音自然但说话人不对。

### Speaker verification / 说话人验证

**EER (Equal Error Rate)。** False Accept Rate 等于 False Reject Rate 的 threshold。ECAPA 在 VoxCeleb1-O 上为 0.87%。

**minDCF (min Detection Cost)。** 在选定 operating point（常见 FAR=0.01）上的加权 cost。比 EER 更接近生产。

### Diarization / 说话人分离标注

**DER (Diarization Error Rate)。** `(FA + Miss + Confusion) / total_speaker_time`。Missed speech + false-alarm speech + speaker-confusion，各自按比例计。AMI meetings：DER ~10–20% 是现实值。pyannote 3.1 + Precision-2 commercial 在录音良好的音频上可达到 &lt;10% DER。

**JER (Jaccard Error Rate)。** DER 的替代方案，对 short-segment bias 更鲁棒。

### Audio classification / 音频分类

Multi-label：所有类别上的 **mAP (mean Average Precision)**。AudioSet：BEATs-iter3 为 0.548 mAP。

Multi-class exclusive：**top-1, top-5 accuracy**。Speech Commands v2：99.0% top-1（Audio-MAE）。

Imbalanced：**macro F1** + **per-class recall**。报告 per-class，aggregate accuracy 会隐藏失败类别。

### Music generation / 音乐生成

**FAD (Fréchet Audio Distance)。** Real 与 generated audio 的 VGGish-embedding distributions 之间的距离。MusicGen-small 在 MusicCaps 上为 4.5。MusicLM 为 4.0。越低越好。

**CLAP Score。** 使用 CLAP embeddings 的 text-audio alignment score。&gt; 0.3 = reasonable alignment。

**Listening panel MOS。** Consumer-grade music 的最终裁决。Suno v5 在 TTS Arena 上基于 paired human preferences 达到 ELO 1293。

### Audio-language benchmarks / Audio-language benchmark

**MMAU (Massive Multi-Audio Understanding)。** 10k audio-QA pairs。

**MMAU-Pro。** 1800 个 hard items，四类：speech / sound / music / multi-audio。4-way 随机机会是 25%。Gemini 2.5 Pro overall 约 60%；所有模型 multi-audio 约 22%。

**LongAudioBench。** 带 semantic queries 的多分钟 clips。Audio Flamingo Next 超过 Gemini 2.5 Pro。

**AudioCaps / Clotho。** Captioning benchmarks。SPICE、CIDEr、FENSE metrics。

### Streaming speech-to-speech / 流式语音到语音

**Latency P50 / P95 / P99。** 从 end-of-user-speech 到第一个可听 response 的 wall-clock。Moshi：200 ms；GPT-4o Realtime：300 ms。

**WER / MOS** on the output。

**Barge-in responsiveness。** 从用户打断到 assistant mute 的时间。目标 &lt; 150 ms。

### The 2026 leaderboards / 2026 榜单

| Leaderboard | Tracks | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO from paired votes | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (within MMAU) |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |

## Build It / 动手构建

### Step 1: WER with normalization / 第 1 步：带 normalization 的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Step 2: TTS round-trip WER / 第 2 步：TTS round-trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Step 3: SECS for voice cloning / 第 3 步：voice cloning 的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Step 4: FAD for music generation / 第 4 步：music generation 的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Step 5: EER for speaker verification (same code as Lesson 6) / 第 5 步：speaker verification 的 EER（与第 6 课相同）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Use It / 应用它

每次部署都要配一个固定 eval harness，并在每次 model update 时运行。三条基本规则：

1. **Normalize before scoring / 打分前先 normalize。** Lowercase、punctuation-strip、number-expand。报告 normalization rule。
2. **Report distributions, not averages / 报告分布，不只报平均。** Latency 用 P50/P95/P99。Classification 报 per-class recall。MMAU 报 per-category。
3. **Run one canonical public benchmark / 跑一个标准公开 benchmark。** 即使你的 production data 不同，报告 Open ASR / TTS Arena / MMAU 可以让 reviewers apples-to-apples 地比较。

## Pitfalls / 常见坑

- **UTMOS extrapolation / UTMOS 外推。** 它在 VCTK-style clean speech 上训练；对 noisy / cloned / emotional audio 打分较差。
- **MOS panel bias / MOS 面板偏差。** 20 个 Amazon Mechanical Turk workers ≠ 20 个目标用户。高风险场景请付费找领域面板。
- **FAD depends on reference set / FAD 依赖 reference set。** 跨模型比较时使用同一个 reference distribution。
- **Aggregate WER / 聚合 WER。** Overall 5% WER 可能掩盖 accented speech 上 30% WER。按 demographic slice 报告。
- **Public benchmark saturation / 公开 benchmark 饱和。** 多数 frontier models 在标准 benchmark 上接近天花板。构建能反映你流量的 in-house held-out set。

## Ship It / 交付它

保存为 `outputs/skill-audio-evaluator.md`。为任意 audio model release 选择 metrics、benchmarks 和 reporting format。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。在 toy inputs 上计算 WER / CER / EER / SECS / FAD-ish / MMAU-ish。
2. **Medium / 中等。** 构建 TTS round-trip WER harness。让你的 Kokoro 或 F5-TTS output 经过 Whisper。对 50 个 prompts 计算 WER。标记 WER &gt; 10% 的 prompts。
3. **Hard / 困难。** 在 MMAU-Pro speech + multi-audio subsets（各 50 个 items）上给第 10 课选择的 LALM 打分。报告 per-category accuracy，并与公开数字比较。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| WER | ASR score | normalization 后 word level 的 `(S+D+I)/N`。 |
| CER | Character WER | 用于 tone languages 或 char-level systems。 |
| MOS | Human opinion | 1–5 rating；20+ listeners × 100 samples。 |
| UTMOS | ML MOS predictor | Learned model；与 human MOS 相关性约 0.9。 |
| SECS | Voice-clone similarity | reference 与 clone 的 ECAPA cosine。 |
| EER | Speaker verif score | FAR = FRR 时的 threshold。 |
| DER | Diarization score | (FA + Miss + Confusion) / total。 |
| FAD | Music-gen quality | VGGish embeddings 上的 Fréchet distance。 |
| RTFx | Throughput | 每 wall-clock second 的 audio seconds。 |

## Further Reading / 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 带 normalization utilities 的 WER/CER library。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — learned MOS predictor。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — music-gen standard。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 live rankings。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — human-vote TTS leaderboard。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM reasoning leaderboard。
- [HEAR benchmark](https://hearbenchmark.com/) — audio SSL benchmarks。
