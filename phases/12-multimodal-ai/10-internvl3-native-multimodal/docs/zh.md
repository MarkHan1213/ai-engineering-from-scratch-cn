# InternVL3: Native Multimodal Pretraining / InternVL3：原生多模态预训练

> InternVL3 之前，几乎每个 open VLM 都遵循同样三步：拿一个在万亿文本 token 上训练好的 text LLM，接上 vision encoder，再微调接缝。这个方案能工作，但会积累 alignment debt：text LLM 的全部预训练预算都花在 pure text 上，并不原生理解 visual tokens。后接 vision 时，LLM 必须重新学习 visual input 与 text reasoning 的关系，同时还不能忘掉文本能力。InternVL3（Zhu et al., 2025 年 4 月）拒绝 post-hoc 路线：一次 pretraining，从 step one 就把 text 和 multimodal interleaved 起来。结果是 78B open model 在 MMMU-Pro 上追平 Gemini 2.5 Pro。本课解释为什么 native pretraining 值得做，以及一旦这么做系统会发生什么变化。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, training-corpus mixer)
**Prerequisites / 前置知识：** Phase 12 · 05, Phase 12 · 07 (recipes)
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 解释 post-hoc VLM training 为什么会积累 alignment debt，并引用三个可测症状（catastrophic forgetting、answer drift、visual-text inconsistency）。
- 描述 InternVL3 的 native pretraining corpus mix，以及 text : interleaved : caption ratio 为什么重要。
- 比较 V2PE（variable visual position encoding）与 Qwen2-VL 的 M-RoPE。
- 说出 Visual Resolution Router（ViR）和 Decoupled Vision-Language（DvD）两种 deployment optimizations。

## The Problem / 问题

Post-hoc VLM training 是默认做法。LLaVA、BLIP-2、Qwen-VL、Idefics 都拿一个 already-pretrained LLM（Llama、Vicuna、Qwen、Mistral）再加 vision。训练阶段通常是：

1. Frozen LLM + frozen vision encoder + trainable projector，在 caption pairs 上训练以对齐 embeddings。
2. Unfreeze LLM，在 instruction data（LLaVA-Instruct、ShareGPT4V）上训练。
3. 可选 task-specific fine-tune。

Alignment debt 有三个症状：

- Catastrophic forgetting。Post-hoc VLM 忘掉 text-only skills。GSM8K 分数下降 5-10 点。Hellaswag 下降。纯文本 agents 回退。
- Answer drift。同一个视觉问题稍微改写，答案就变。Vision encoder 与 LLM 的连接比 LLM 自己的 tokens 更弱。
- Visual-text inconsistency。VLM 能正确描述图像，却在后续问题中给出与自己描述矛盾的答案。Visual tokens 没有像 text tokens 一样参与 LLM 内部 consistency checks。

这些症状有文献记录。MM1.5 Section 4 对其量化。LLaVA-OneVision 的 ablations 也有暗示。Native pretraining 是对应答案。

## The Concept / 概念

### Native multimodal pretraining / 原生多模态预训练

InternVL3 从零开始在一个 native multimodal corpus 上训练。数据混合为：

- 40% text-only data（FineWeb、Proof-Pile-2 等）
- 35% interleaved image-text data（OBELICS、MMC4-style）
- 20% paired image-caption data
- 5% video-text data

Vision tokens、text tokens 和 cross-modal interactions 从第一步梯度开始就参与同一个 loss。没有 alignment pretraining，没有 projector freezing stage，也没有要恢复的 catastrophic forgetting。

Base model 的训练是单阶段。后面仍有 instruction tuning，但 base model 已经把 visual tokens 当作一等公民。

### V2PE (variable visual position encoding) / V2PE（可变视觉位置编码）

Qwen2-VL 使用 M-RoPE 和固定 axis allocation。InternVL3 引入 V2PE：position encoding 会随 modality type（text、image、video）变化，并带可学习 scaling。实践中：

- Text tokens 获取 1D position（text index）。
- Image patches 获取 2D position（row, col）。
- Video frames 获取 3D position（time, row, col）。

三者共享同一个 RoPE frequency base，但每个 band 在 hidden dim 中的 allocation 是可学习参数，而不是固定拆分。预训练期间模型可以自由权衡 temporal 与 spatial frequency resolution。

V2PE 的 ablation claim：同 compute 下，视频 benchmark 比 M-RoPE 高 1-2 点。不是革命，但更干净。

### Visual Resolution Router (ViR) / Visual Resolution Router（ViR）

这是部署优化。不是所有图像都需要 full-resolution encoding。一张只有单个物体、低细节的照片，如果按 1280px native 编码就是浪费 token。ViR 是一个小 classifier，在编码前预测回答问题所需的最低 resolution。

Routing 有三档：low-res（256 tokens）、medium（576）、high（2048+）。生产流量中约 60% query 低或中档就足够。净效果：等质量下吞吐提升 2-3x。

### Decoupled Vision-Language deployment (DvD) / Decoupled Vision-Language 部署（DvD）

Serving 大 VLM 时，vision encoder 每张图运行一次，而 LLM 要为每个 output token 自回归运行。两者瓶颈不同（vision = conv + attention 的 GPU memory bandwidth；LLM = KV cache）。DvD 把它们拆到不同 GPU，并在中间 streaming handoff。

对 8B + 400M encoder 模型，DvD 相比 co-located 通常让每节点吞吐翻倍。

### Single-stage vs multi-stage quality / 单阶段与多阶段质量

InternVL3 的主要 benchmark claim：78B 参数追平 Gemini 2.5 Pro 的 MMMU-Pro；38B 追平 GPT-4o；8B 领先 open-8B leaderboard。全部基于 single-stage pretrain + instruction-tune recipe。

Alignment-debt hypothesis 可测：单位 vision-benchmark gain 下，InternVL3-8B 在 text benchmarks（MMLU、GSM8K）上丢分少于 Qwen2.5-VL-7B。模型更像 generalist，因为训练是一体的，不是两段拼接。

### InternVL3.5 and InternVL-U / InternVL3.5 与 InternVL-U

InternVL3.5（2025 年 8 月）扩展同一 recipe。还是 native-pretrain approach，更多数据、更多参数。MMMU 提升是 incremental。

InternVL-U（2026）加入 unified generation：在同一 backbone 上增加 MMDiT heads 做 image output。“U” 表示 “Understanding + generation”，追随 Transfusion-style unified models（Lesson 12.13）。同一个 native-pretrain backbone 支持理解和生成 heads。

### Trade-offs of native pretraining / Native pretraining 的代价

Native pretraining 不免费：

- Compute。从零训练新 VLM 的成本接近训练 text LLM，数百万 GPU-hours。Post-hoc adaptation 复用现有 LLM weights，省掉大部分成本。
- Data。大规模 interleaved image-text corpora 稀缺。OBELICS 是 141M documents；MMC4 是 571M。纯文本则有 15T tokens。多模态预训练数据稀缺是硬约束。
- Base-LLM reuse。Native pretraining 放弃了以后 drop in 新 LLM 的选项。Post-hoc 可以把 Llama-3.1 换成 Llama-4，只重训 adapter。

InternVL3 的赌注是：alignment debt 比 reuse loss 更糟。Benchmark 支持这个判断。但生产成本会阻止许多实验室低成本复刻。Post-hoc VLM 仍会长期存在，因为多数项目更便宜。

## Build It / 动手构建

本课构建两个模拟器：training-corpus mixer 用来推演 text/interleaved/caption/video 的步数比例，ViR router simulator 用来估算按分辨率路由后的平均 token count。你会把 native vs post-hoc 的收益和成本放到同一张表里。

## Use It / 应用它

`code/main.py` 是 training-corpus mixer 和 ViR router simulator。它会：

- 接收目标 corpus mix（%text、%interleaved、%caption、%video），计算每种 modality 的 expected steps。
- 在一批 queries 上模拟 ViR routing（分布：50% low-detail、30% medium、20% high-detail），报告 average token count。
- 给定 encoder vs LLM FLOPs，报告 DvD throughput estimates。
- 打印 post-hoc vs native pretraining 在参数、compute、data 和 expected alignment-debt symptoms 上的 side-by-side。

## Ship It / 交付它

本课产出 `outputs/skill-native-vs-posthoc-auditor.md`。给定一个 proposed VLM training plan，它会审计应该走 native 还是 post-hoc，标记 alignment-debt risk，并推荐 corpus mix。设计新 open-VLM 项目并选择训练策略时使用它。

## Exercises / 练习

1. 估算 InternVL3-8B（native pretrain）与 LLaVA-OneVision-7B（post-hoc）之间的 compute delta。GPU-hours ratio 大约是多少？差距由什么解释？

2. InternVL3 报告 40% text / 35% interleaved / 20% caption / 5% video。如果目标任务是 video-heavy，提出新的 ratio，并说明 base model 为什么仍然需要大量 text 和 caption data。

3. 阅读 MM1.5 Section 4 关于 forgetting 的内容。说出 post-hoc training 中 regression 最大的具体 benchmark，以及损失了多少。

4. ViR 把 60% traffic 路由到 low-resolution encoding。它会在哪类 query 上误路由（需要 high-res 却送到 low-res）？提出三种 router-failure modes。

5. DvD 把 vision 与 LLM 拆到不同 GPU。在哪种 traffic pattern 下，DvD 会伤害 throughput 而不是帮助？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Native multimodal pretraining | “From scratch together” | Text + image + video tokens 从 step 1 起共同参与 loss，而不是事后接上 |
| Alignment debt | “Post-hoc penalty” | 把 vision 接到 frozen LLM 后，在文本技能和答案一致性上产生的可测退化 |
| V2PE | “Variable visual pos encoding” | Per-modality 可学习 position encoding allocation；InternVL3 的 M-RoPE 后继 |
| ViR | “Resolution router” | 编码前按 query 选择最低所需 resolution 的小分类器，节省 inference tokens |
| DvD | “Decoupled deployment” | Vision encoder 在一张 GPU、LLM 在另一张 GPU，中间 stream handoff；大 VLM 吞吐可翻倍 |
| InternVL-U | “Unified understanding + generation” | 2026 follow-up，在 native-pretrain backbone 上加入 image-generation heads |
| Interleaved corpus | “OBELICS / MMC4” | 按自然阅读顺序包含 text 与 images 的 documents；native pretraining 的原材料 |

## Further Reading / 延伸阅读

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
