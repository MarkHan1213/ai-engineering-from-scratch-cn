# Music Generation — MusicGen, Stable Audio, Suno, and the Licensing Earthquake / 音乐生成：MusicGen、Stable Audio、Suno 与授权地震

> 2026 年的 music generation：Suno v5 和 Udio v4 主导商业市场；MusicGen、Stable Audio Open 和 ACE-Step 领先开源。技术问题大多已解决。法律问题（Warner Music 5 亿美元和解、UMG 和解）在 2025–2026 年重塑了这个领域。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 instrumental generation、song generation 和 conditional / controllable music generation
- 理解 neural-codec token LM、latent diffusion 和 closed hybrid production systems 的技术路线
- 用 MusicGen 生成音乐、做 melody conditioning，并用 FAD 做回归评估
- 为音乐生成部署选择模型、授权策略、长度/结构方案和 disclosure metadata

## The Problem / 问题

Text → 30 秒到 4 分钟的 music clip，包含 lyrics、vocals 和 structure。这里有三个子问题：

1. **Instrumental generation / 器乐生成。** 像 “lo-fi hip-hop drums with warm keys” 这样的文本 → audio。MusicGen、Stable Audio、AudioLDM。
2. **Song generation (with vocals + lyrics) / 歌曲生成（带人声和歌词）。** “Country song about rainy Texas nights” → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **Conditional / controllable / 条件式或可控生成。** 扩展已有 clip、重新生成 bridge、替换 genre、stem-separate 或 inpaint。Udio 的 inpainting + stem separation 是 2026 年需要对齐的 feature。

## The Concept / 概念

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### Token LM over neural-codec tokens / neural-codec tokens 上的 Token LM

Meta 的 **MusicGen**（2023, MIT）和很多衍生模型：以 text/melody embeddings 为条件，自回归预测 EnCodec tokens（32 kHz，4 codebooks），再用 EnCodec decode。300M - 3.3B params。强 baseline；超过 30 秒后开始吃力。

**ACE-Step**（开源，4B XL 于 2026 年 4 月发布）把这条路线扩展到带歌词条件的 full-song generation。它是开源社区最接近 Suno 的模型。

### Diffusion over mels or latents / mels 或 latents 上的 diffusion

**Stable Audio (2023)** 和 **Stable Audio Open (2024)**：在压缩 audio latent 上做 latent diffusion。擅长 loops、sound design、ambient textures。不擅长结构化完整歌曲。

**AudioLDM / AudioLDM2**：通过 T2I-style latent diffusion 做 text-to-audio，并泛化到音乐、音效、语音。

### Hybrid (production) — Suno, Udio, Lyria / 混合式生产系统：Suno、Udio、Lyria

Closed weights。很可能是 AR codec LM + diffusion-based vocoder，并带专门的 voice / drum / melody heads。Suno v5（2026）是 ELO 1293 的质量领先者。Udio v4 加入 inpainting + stem separation（bass、drums、vocals 可分轨下载）。

### Evaluation / 评估

- **FAD (Fréchet Audio Distance)。** 用 VGGish 或 PANNs features 计算 generated 与 real audio distribution 的 embedding-level distance。越低越好。MusicGen small 在 MusicCaps 上 FAD 4.5；SOTA 约 3.0。
- **Musicality (subjective) / 音乐性（主观）。** Human preference。Suno v5 以 ELO 1293 领先。
- **Text-audio alignment / 文本-音频对齐。** Prompt 与 output 的 CLAP score。
- **Musicality artifacts / 音乐性瑕疵。** Off-beat transitions、vocal-phrase drift、30 秒之后结构丢失。

## 2026 model map / 2026 模型地图

| Model | Params | Length | Vocals | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | yes | Apache-2.0 |
| YuE | 7B | &gt; 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes, ELO 1293 | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |
| Google Lyria 3 (closed) | ? | real-time | yes | commercial |
| MiniMax Music 2.5 | ? | 4 min | yes | commercial API |

## The legal landscape (2025-2026) / 法律格局（2025–2026）

- **Warner Music vs Suno settlement / Warner Music 与 Suno 和解。** 5 亿美元。WMG 现在对 Suno 上的 AI-likeness、music rights 和 user-generated tracks 拥有监督权。Udio 上也有类似的 UMG 和解。
- **EU AI Act** + **California SB 942**：AI-generated music 必须披露。
- MIT 许可下的 **Riffusion / MusicGen** 没有 compliance baggage，但也没有商业人声能力。

安全上线模式：

1. 只生成 instrumental（MusicGen、Stable Audio Open、MIT/CC0 outputs）。
2. 使用带 per-generation license 的商业 APIs（Suno、Udio、ElevenLabs Music）。
3. 在自有或授权 catalog 上训练（多数企业最后会走这里）。
4. 给 generations 加 watermarks + metadata。

## Build It / 动手构建

### Step 1: generate with MusicGen / 第 1 步：用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三个尺寸：`small`（300M，快）、`medium`（1.5B）、`large`（3.3B）。Small 足够验证“这个想法是否成立”。

### Step 2: melody conditioning / 第 2 步：melody conditioning

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接收 chromagram，在替换 timbre 的同时保留旋律。适合“把这段旋律变成 string quartet”。

### Step 3: FAD evaluation / 第 3 步：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish-embedding distance。适合 genre-level regression tests；不能替代 human listeners。

### Step 4: adding to the LLM-music workflow / 第 4 步：接入 LLM-music workflow

结合第 7–8 课的思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Use It / 应用它

| Goal | Stack |
|------|-------|
| Instrumental sound design | Stable Audio Open |
| Game / adaptive music | Google Lyria RealTime (closed) |
| Full songs with vocals (commercial) | Suno v5 or Udio v4 with explicit license |
| Full songs with vocals (open) | ACE-Step XL or YuE |
| Short ad jingle | MusicGen melody-conditioned on a hummed reference |
| Music-video background | MusicGen + Stable Video Diffusion |

## Pitfalls that still ship in 2026 / 2026 年仍会上线的坑

- **Copyright-laundering prompts / 版权洗白 prompt。** “Song in the style of Taylor Swift” —— 商业 Suno/Udio 现在会过滤，开源模型不会。加入你自己的 filter list。
- **Repetition / drift past 30 s / 30 秒后的重复与漂移。** AR models 会循环。可以 crossfade 多个 generations，或用 ACE-Step 提高结构一致性。
- **Tempo drift / 速度漂移。** 模型会偏离 BPM。Prompt 中使用 BPM tags，并用 librosa 的 `beat_track` 做后处理过滤。
- **Vocal intelligibility / 人声可懂度。** Suno 很强；开源模型在词上常常糊。如果歌词重要，使用商业 API 或 fine-tune。
- **Mono output / 单声道输出。** 开源模型生成 mono 或 fake-stereo。用合适的 stereo reconstruction 升级（ezst、Cartesia's stereo diffusion）。

## Ship It / 交付它

保存为 `outputs/skill-music-designer.md`。为 music-gen deployment 选择 model、license strategy、length / structure plan 和 disclosure metadata。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。它会用 ASCII 符号生成一个“generative” chord progression + drum pattern，也就是 music-gen cartoon。如果愿意，可以用任意 MIDI renderer 播放。
2. **Medium / 中等。** 安装 `audiocraft`，用 MusicGen-small 对 4 个 genre prompts 分别生成 10 秒 clips，并对 reference genre set 测量 FAD。
3. **Hard / 困难。** 使用 ACE-Step（或 MusicGen-melody），用不同 timbre prompts 为同一旋律生成三个变体。计算与 prompt 的 CLAP similarity，验证 alignment。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| FAD | Audio FID | real 与 generated 的 embedding distributions 之间的 Fréchet distance。 |
| Chromagram | melody as pitches | 每 frame 12 维向量；用于 melody conditioning。 |
| Stems | 乐器分轨 | 分离后的 bass / drums / vocals / melody WAV。 |
| Inpainting | 重新生成某段 | Mask 一个时间窗口；模型只重新生成那段。 |
| CLAP | Text-audio CLIP | 对比式 audio-text embedding；评估 text-audio alignment。 |
| EnCodec | music codec | Meta 的 neural codec，MusicGen 使用；32 kHz，4 codebooks。 |

## Further Reading / 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开源 autoregressive benchmark。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — sound-design 默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 2026 年 4 月发布的开源 4B full-song generator。
- [Suno v5 platform docs](https://suno.com) — 商业质量领先者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 用于 music + sound effects 的 latent diffusion。
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月先例。
