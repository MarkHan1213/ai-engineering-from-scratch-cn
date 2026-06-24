# Qwen-VL Family and Dynamic-FPS Video / Qwen-VL 家族与 Dynamic-FPS 视频

> Qwen-VL family，包括 Qwen-VL（2023）、Qwen2-VL（2024）、Qwen2.5-VL（2025）、Qwen3-VL（2025），是 2026 年最有影响力的 open vision-language model lineage。每一代都做出一个决定性的架构下注，并在 12 个月内被 open ecosystem 复制：M-RoPE 的 native dynamic resolution、带 absolute time alignment 的 dynamic-FPS sampling、ViT 中的 window attention，以及 structured agent output formats。到 Qwen3-VL，recipe 已经稳定：2D-RoPE-ViT encoder 支持 native-aspect-ratio inputs，MLP projector 接入大 Qwen3 language base，并在训练阶段把 OCR、grounding 和 agent behavior 当作一等目标。本课按时间线读这个家族，解释每个 knob 为什么在那里。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**Prerequisites / 前置知识：** Phase 12 · 06 (patch-n'-pack)
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 计算 M-RoPE 的三轴 rotations（temporal、height、width），并解释三轴为何都需要。
- 为一个视频选择 dynamic-FPS sampling strategy，并分析 tokens-per-second 与 event-detection accuracy 的关系。
- 按顺序说出 Qwen-VL 四代升级，以及每次升级解锁了什么。
- 连接 Qwen2.5-VL-style JSON agent output format，并从 VLM response 解析 structured tool calls。

## The Problem / 问题

Qwen-VL 在 2023 年 8 月发布，直接回应 LLaVA-1.5 与 BLIP-2。Qwen 团队瞄准的差距有三类：resolution、video 和 structured output。

Resolution：LLaVA-1.5 运行在 336x336。对照片还行，对中文发票或密集电子表格截图几乎无用。Qwen-VL 的第一个创新是 448x448 和 grounded bounding-box output，让模型能指物体。

Video：Video-LLaMA 叠加 per-frame encoders 再喂给 LLM。短视频可用，但多分钟视频中 temporal axis 才是信号。Qwen 团队希望有一个理解时间的单一 encoder。

Structured output：LLaVA 输出 free-form text。Agent 需要 JSON。Qwen-VL 在明确 JSON output formats 上训练，包括用文本表示 bounding-box coordinates。

每一代 Qwen-VL 都在这三个轴上推进。

## The Concept / 概念

### Qwen-VL (August 2023) / Qwen-VL（2023 年 8 月）

第一代：OpenCLIP ViT-bigG/14 作为 encoder（2.5B 参数）、Llama-compatible Q-Former（1-step with 256 queries）、Qwen-7B base。贡献：

- 448x448 resolution（当时 open VLM 的 SOTA）。
- Grounding：在带显式 coordinate-token output 的 image-text pairs 上训练。“The cat is at <box>(112, 204), (280, 344)</box>”。
- 从一开始就中英 multilingual training。

当时 benchmark 上：英文接近 GPT-4V，中文占优。真正的 headline 是 grounding supervision。

### Qwen2-VL (September 2024) — M-RoPE and native resolution / Qwen2-VL（2024 年 9 月）：M-RoPE 与 native resolution

Qwen2-VL 把 fixed-resolution + Q-Former stack 换成 natively dynamic-resolution ViT encoder。关键变化：

- Native dynamic resolution。ViT 接受任意 HxW，只要能被 28 整除（patch 14 + 2x spatial merge）。1120x672 图像（40x24 merged patches）产生 960 visual tokens。没有 resize、tiling 或 thumbnail。
- M-RoPE（Multimodal RoPE）。每个 token 携带 3D position（t, h, w），而不是 1D。图像 t=0，视频 t=frame_index。RoPE 用每个轴的频率旋转 query/key vectors。没有 positional embedding table。
- MLP projector。移除 Q-Former；对 merged patch tokens 使用 2-layer MLP。
- Dynamic FPS 视频。默认 1-2 FPS 采样，但模型接受任意 frame count。

结果：Qwen2-VL-7B 在多个 multimodal benchmark 上追平 GPT-4o，并在 DocVQA 上击败它（94.5 vs 88.4）。架构变化是决定性一步。

### Qwen2.5-VL (February 2025) — dynamic FPS + absolute time / Qwen2.5-VL（2025 年 2 月）：dynamic FPS + absolute time

Qwen2.5-VL 最大变化是视频。Dynamic FPS 不只是“需要时采更多帧”。论文形式化了：

- Absolute time tokens。不是 position indices（frame 0、1、2），而是实际 timestamp。“At 0:04, the cat jumps.” 模型看到 interleaved `<time>0.04</time>` tokens。
- Dynamic FPS。慢镜头用 1 FPS，动作场景用 4+ FPS。用户或训练器选择；M-RoPE 适配。
- ViT 中的 window attention。空间 attention 被 windowed（局部块内）以提升吞吐，每几层加入 global attention。
- 明确 JSON output format。使用 tool-call data 训练："{\"tool\": \"click\", \"coords\": [380, 220]}"。开箱即 agent-ready。
- MRoPE-v2 scaling。Position 随 max input size scale，避免 10 分钟视频跑出频率范围。

Benchmark：Qwen2.5-VL-72B 在大多数视频 benchmark 上击败 GPT-4o，在 documents 上追平 Gemini 2.0，并成为 GUI grounding 的 open-model SOTA（ScreenSpot：84% accuracy vs GPT-4o 的 38%）。

### Qwen3-VL (November 2025) / Qwen3-VL（2025 年 11 月）

Qwen3-VL 是整合式升级，而不是重新发明：更大的 LLM backbone（Qwen3-72B）、更多训练数据、改进 OCR，以及 Qwen3 “thinking mode” 带来的更强 reasoning。ViT 与 M-RoPE 保持不变。论文重点是数据和训练，而非架构。

这个 lineage 的 takeaway：到 2025 年，Qwen-VL 架构已经稳定。后续代际主要 scale compute 和 data，而不是 primitives。

### M-RoPE mathematically / M-RoPE 数学形式

经典 RoPE 按 position `m` 对维度 `d` 的 query `q` 做成对旋转：

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 把 hidden dim 拆成三个 band。假设 `d = 96`，分配 32 dims 给 temporal、32 给 height、32 给 width。每个 band 按自己的轴位置旋转。一个 patch 位于 `(t=5, h=10, w=20)` 时，会对三段分别应用 `R_t(5)`、`R_h(10)`、`R_w(20)`。

Text tokens 使用 `t = text_index, h = 0, w = 0`（或规范化选择）以保持兼容。Video frames 使用 `t = frame_time, h = row, w = col`。Single images 使用 `t = 0`。

收益是：同一种 position encoding 处理 text、image 和 video，不需要分支代码或不同 position tables。

### Dynamic-FPS sampling logic / Dynamic-FPS 采样逻辑

给定时长为 `T` 秒、目标 token budget 为 `B` 的视频：

1. 计算可承受的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选择满足 `fps <= fps_max` 的目标 FPS。
3. 如果 motion high（optical-flow heuristic 或用户明确请求），选更高 FPS；motion low 则选更低。
4. 按选定 FPS 均匀采样；在 frame 之间插入 `<time>t</time>` tokens。

Qwen2.5-VL 在训练中隐式学习这套逻辑；推理时用户通过 `fps` 参数控制。60 秒动作序列，4 FPS、81 tokens/frame = 19440 tokens，在 32k context 内可管理。

### Structured agent output / 结构化 Agent 输出

Qwen2.5-VL 的 agent training 明确面向 structured tool calls：

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出做 JSON.parse。对比 free-form “click at (1024, 512)”，后者需要 regex 且有歧义。这一转变解释了为什么 Qwen2.5-VL 的 ScreenSpot 分数从 Qwen2-VL 的 55% 跳到 84%。

## Build It / 动手构建

本课动手构建三个小组件：M-RoPE 的 position calculator、dynamic-FPS frame sampler，以及 Qwen2.5-VL 风格 JSON tool-call parser。它们对应 Qwen-VL 家族最重要的三个工程能力：native resolution、video time alignment 和 agent-ready output。

## Use It / 应用它

`code/main.py` 实现：

- 对混合 text、image patches 和 video frames 的 packed sequence 计算 M-RoPE positions。
- Dynamic-FPS sampler：给定 duration、budget、motion_level，选择 FPS 并输出 frame timestamps。
- 一个 toy Qwen2.5-VL JSON-output parser，处理带 coordinate fields 的 tool-call responses。

运行它，然后把固定 FPS 换成 dynamic-FPS，感受 5 分钟视频上的差异。

## Ship It / 交付它

本课产出 `outputs/skill-qwen-vl-pipeline-designer.md`。给定 video task（monitoring、agent、action recognition、accessibility），它会输出 Qwen2.5-VL configuration（frame budget、FPS strategy、window-attention flag、agent-output mode）和 latency estimate。部署 Qwen-VL-family model 做视频产品时使用它。

## Exercises / 练习

1. 对 hidden 48（每 band 16，base theta 10000）中位于 `(t=3, h=5, w=7)` 的 patch 计算 M-RoPE rotations。展示每个 band 前三对维度的 rotation angles。

2. 10 分钟 security-camera recording 在 1 FPS 下有多少 frames？384 resolution + 3x pool 下总 tokens 是多少？Qwen2.5-VL 的默认 32k context 能处理吗？

3. 分别为 30 秒 tennis rally、30 秒 recipe demo、30 秒 UI-agent recording 选择 FPS。用 dynamic-FPS logic 解释。

4. Qwen2.5-VL 完全移除了 Q-Former。为什么简单 MLP 在 2025 年可行，而 2023 年不行？提示：data scale 和 encoder quality。

5. 把三个 Qwen2.5-VL JSON tool-call outputs 解析成 Python dict。Malformed JSON 会失败在哪里？Qwen cookbook 推荐什么 recovery strategy？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| M-RoPE | “Multimodal RoPE” | hidden dim 中带 temporal、height、width bands 的 3D rotary position embedding |
| Dynamic FPS | “Smart sampling” | 根据 motion、duration 和 token budget 为每个视频选择 frame sampling rate |
| Absolute time token | “Timestamp token” | 在序列中 interleaved 的 `<time>t</time>`，让模型看到实际秒数而非 frame index |
| Window attention | “Local attention” | 空间 self-attention 限制在小窗口以提速，并周期性加入 global attention |
| Structured agent output | “JSON mode” | 训练数据监督 VLM 输出包含 coords 和 tool names 的 parseable JSON |
| min_pixels / max_pixels | “Resolution bounds” | Qwen2.5-VL per-request controls，用于限制总像素数，从而限制 token count |
| Grounding | “Point-at-it” | 以 text tokens 输出 bounding-box coordinates；Qwen-VL v1 就开始使用 |

## Further Reading / 延伸阅读

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
