# Voice Cloning & Voice Conversion / 声音克隆与声音转换

> Voice cloning 会用别人的声音朗读你的文本。Voice conversion 会把你的声音改写成别人的声音，同时保留你说的内容。两者都依赖同一个分解：把 speaker identity 和 content 分开。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 voice cloning 和 voice conversion，并理解 content、speaker、prosody 的分解与重组
- 用 F5-TTS、KNN-VC、watermark 和 consent gate 描述一条可上线的 cloning/conversion pipeline
- 解释 zero-shot、few-shot fine-tuning、recognition-synthesis、disentanglement 和 neural codec cloning 的差异
- 把 watermarking、consent record 和 deepfake detection 当成系统要求，而不是后补功能

## The Problem / 问题

到 2026 年，消费级 GPU 上只需 5 秒音频，就足以生成任何人声音的高质量 clone。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都已经提供 zero-shot 或 few-shot cloning。这项技术既是福音（accessibility TTS、dubbing、assistive voices），也是武器（诈骗电话、政治 deepfake、IP theft）。

两个紧密相关的任务：

- **Voice cloning (TTS-side)：** text + 5-second reference voice → 用该声音说出的 audio。
- **Voice conversion (speech-side)：** source audio（A 说了 X）+ B 的 reference voice → B 说 X 的 audio。

二者都会把 waveform 分解为 (content, speaker, prosody)，再把一个来源的 content 和另一个来源的 speaker 重组。

2026 年交付时必须遵守的关键约束：**watermarking 和 consent gates 在欧盟（AI Act，2026 年 8 月起可执行）以及加州（AB 2905，2025 年生效）已是法律要求**。你的 pipeline 必须写入不可闻 watermark，并拒绝无同意的 clone。

## The Concept / 概念

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**Zero-shot cloning / 零样本克隆。** 把一段 5 秒 clip 传给一个在数千个说话人上训练过的模型。Speaker encoder 把 clip 映射成 speaker embedding；TTS decoder 以该 embedding 和文本为条件生成音频。

使用者：F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024)。

**Few-shot fine-tuning / 小样本微调。** 录制目标声音 5–30 分钟。用一小时对 base model 做 LoRA-fine-tune。质量会从“还可以”跃迁到“难以区分”。Coqui 和 ElevenLabs 都支持这个模式；社区也常和 F5-TTS 搭配使用。

**Voice conversion (VC) / 声音转换。** 两大家族：

- **Recognition-synthesis。** 运行类似 ASR 的模型提取 content representation（例如 soft phoneme posteriors、PPGs），再用 target speaker embedding 重合成。对语言和口音鲁棒。KNN-VC (2023)、Diff-HierVC (2023) 使用这一类。
- **Disentanglement。** 训练一个 autoencoder，在 bottleneck 的 latent space 中分离 content、speaker 和 prosody。推理时替换 speaker embedding。质量较低但更快。AutoVC (2019)、VITS-VC variants 使用这一类。

**Neural codec-based cloning (2024+) / 基于 neural codec 的克隆。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox 会把音频看作 SoundStream / EnCodec 产生的 discrete tokens，并在 codec tokens 上训练大型 autoregressive 或 flow-matching model。短 prompts 上的质量可与 ElevenLabs 相当。

### The ethics bit, not a bolt-on / 伦理不是外挂

**Watermarking / 水印。** PerTh (Perth) 和 SilentCipher (2024) 会在音频中不可感知地嵌入约 16–32 bit ID。它能经受 re-encoding、streaming 和常见编辑。开源且可生产使用。

**Consent gates / 同意门控。** 每个 cloned output 都必须绑定可验证的 consent record。“I, Rohit, on 2026-04-22, authorize this voice for X purpose.” 存入 tamper-evident log。

**Detection / 检测。** AASIST、RawNet2、Wav2Vec2-AASIST 都可作为 detectors。ASVspoof 2025 challenge 发布了 state-of-the-art detectors 对 ElevenLabs、VALL-E 2 和 Bark 输出的 EER：0.8–2.3%。

### Numbers (2026) / 2026 年数字

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70 对大多数听众来说通常已难以和目标声音区分。

## Build It / 动手构建

### Step 1: decompose with recognition-synthesis (code-only demo in main.py) / 第 1 步：用 recognition-synthesis 分解（`main.py` 中的纯代码 demo）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上很简单；实现重量都在 `tts_model` 和 speaker encoder 里。

### Step 2: zero-shot clone with F5-TTS / 第 2 步：用 F5-TTS zero-shot clone

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Reference transcript 必须与 audio 完全匹配；不匹配会破坏 alignment。

### Step 3: voice conversion with KNN-VC / 第 3 步：用 KNN-VC 做 voice conversion

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 用 WavLM 提取 source 和 target pool 的 per-frame embeddings，然后用 pool 中最近邻替换每个 source frame。它是 non-parametric 的，用 1 分钟目标语音就能工作。

### Step 4: embed a watermark / 第 4 步：嵌入 watermark

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

约 32 bits payload，在 MP3 re-encode 和轻微噪声后仍可检测。

### Step 5: consent gate / 第 5 步：consent gate

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Use It / 应用它

2026 年的 stack：

| Situation | Pick |
|-----------|------|
| 5-sec zero-shot clone, open-source | F5-TTS or OpenVoice v2 |
| Commercial production cloning | ElevenLabs Instant Voice Clone v2.5 |
| Voice conversion (rewriting) | KNN-VC or Diff-HierVC |
| Many-speaker fine-tune | StyleTTS 2 + speaker adapter |
| Cross-lingual cloning | XTTS v2 or VALL-E X |
| Deepfake detection | Wav2Vec2-AASIST |

## Pitfalls / 常见坑

- **Misaligned reference transcript / reference transcript 不对齐。** F5-TTS 和类似模型要求 reference text 与 reference audio 完全匹配，包括标点。
- **Reverberant reference / 有混响的 reference。** 回声会毁掉 clone。使用干声、近讲麦克风录制。
- **Emotional mismatch / 情绪不匹配。** “cheerful” 的训练 reference 会让所有 clone 都显得 cheerfully。让 reference emotion 匹配目标用途。
- **Language leakage / 语言泄漏。** 克隆英语说话人后让模型说法语，常常仍会带英语口音；使用 cross-lingual models（XTTS、VALL-E X）。
- **No watermark / 没有 watermark。** 2026 年 8 月起在欧盟不可合法上线。

## Ship It / 交付它

保存为 `outputs/skill-voice-cloner.md`。设计一条 cloning 或 conversion pipeline，包含 consent gate + watermark + quality target。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会通过计算两个 “speakers” 在 swap 前后的 cosine，演示 speaker-embedding swap。
2. **Medium / 中等。** 使用 OpenVoice v2 克隆你自己的声音。测量 reference 和 clone 之间的 SECS。通过 Whisper 测量 CER。
3. **Hard / 困难。** 对 20 个 clones 应用 SilentCipher watermark，经过 128 kbps MP3 encode+decode 后检测 payload。报告 bit-accuracy。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 秒就够 | Pretrained model + speaker embedding；无需训练。 |
| PPG | Phonetic posteriorgram | 作为 language-agnostic content rep 的 per-frame ASR posteriors。 |
| KNN-VC | nearest-neighbor conversion | 用最近的 target-pool frame 替换每个 source frame。 |
| Neural codec TTS | VALL-E style | EnCodec/SoundStream tokens 上的 AR model。 |
| Watermark | 不可闻签名 | 嵌入音频、可经受 re-encode 的 bits。 |
| SECS | cloning fidelity | target 和 clone speaker embeddings 之间的 cosine。 |
| AASIST | deepfake detector | Anti-spoof model；检测合成语音。 |

## Further Reading / 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — open-source SOTA zero-shot cloning。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) and [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — neural-codec TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于 disentanglement 的 voice conversion。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — retrieval-based VC。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — production-ready 32-bit audio watermark。
- [ASVspoof 2025 results](https://www.asvspoof.org/) — detector vs synthesizer arms race，2026 年仍在更新。
