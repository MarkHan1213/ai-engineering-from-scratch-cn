# Chameleon and Early-Fusion Token-Only Multimodal Models / Chameleon 与 Early-Fusion Token-Only 多模态模型

> 到目前为止我们看到的 VLM 都把图像和文本分开。Visual tokens 来自 vision encoder，进入 projector，然后在 LLM 内部与文本相遇。Vision 和 text vocabularies 从不重叠。Chameleon（Meta, 2024 年 5 月）问：如果它们重叠会怎样？训练一个 VQ-VAE，把图像变成 shared vocabulary 中的离散 token 序列。每个多模态文档现在都是一个序列：text tokens 与 image tokens interleaved，一个 autoregressive loss。副作用是模型可以生成 mixed-modality outputs：一次 inference call 中交替输出文本和图像 token。本课读取 early-fusion thesis，并从头构建一个 toy 版本。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites / 前置知识：** Phase 12 · 05, Phase 8 (Generative AI)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 解释 shared vocabulary + single loss 为什么会改变模型能做的事。
- 描述 VQ-VAE 如何把图像 tokenize 成与 transformer next-token objective 兼容的 discrete sequence。
- 说出 Chameleon 的 training-stability tricks：QK-Norm、dropout placement、LayerNorm ordering。
- 对比 Chameleon 与 BLIP-2 的 Q-Former 路线，并说明各自适合什么场景。

## The Problem / 问题

Adapter-based VLM（LLaVA、BLIP-2、Qwen-VL）把文本和图像当成两种东西。Text token 走 `embed(text_token)`；图像走 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条 input paths，在中途合流。

这带来三个后果：

1. LLM 只能消费图像，不能输出图像。输出只可能是文本。
2. Mixed-modality documents（例如文章中交替出现段落与图片）很别扭：要么在模型外解析多模态输入，要么链式生成。
3. Distributional mismatch。Visual tokens 与 text tokens 位于 hidden space 的不同区域，会产生细微 alignment 问题。

Chameleon 拒绝这个前提：图像只是 shared vocabulary 中的一串 discrete tokens。在 interleaved documents 上训练模型，一个 loss、一个 autoregressive decoder，就能自然得到 mixed-modality generation。

## The Concept / 概念

### VQ-VAE as image tokenizer / 作为图像 tokenizer 的 VQ-VAE

Tokenizer 是 vector-quantized variational autoencoder。架构：

- Encoder：CNN + ViT，把图像映射到 spatial feature map，例如 32x32 个 dim 256 features。
- Codebook：K 个可学习向量构成的 vocabulary（Chameleon 使用 8192），同样 dim 256。
- Quantization：对每个 spatial feature，按 L2 distance 查找最近 codebook entry，用整数 index 替换连续 feature。
- Decoder：CNN，把 quantized features 还原成 pixels。

训练：VAE reconstruction loss + commitment loss + codebook loss。Codebook indices 构成图像的离散 alphabet。

对 Chameleon：一张图变成 32*32 = 1024 tokens，来自 8192 大小的 vocabulary。再与文本 tokens（来自 LLM BPE vocabulary，例如 32000）concatenate。最终 vocabulary：40192。Transformer 看到的是一个序列，一个 loss。

### The shared vocabulary / 共享词表

Chameleon 的 vocabulary 组合 text tokens、image tokens 和 modality separators。每个 token 都有单一 ID。Input embedding layer 把每个 ID 映射到 D-dim hidden vector。Output projection 把 hidden 映射回 vocab logits。Softmax 选择下一个 token，不管它属于什么 modality。

Separators 很关键：`<image>` 和 `</image>` 标签包围 image-token sequence。生成时，如果模型输出 `<image>`，下游软件就知道接下来的 1024 个 token 是要送给 decoder 渲染成像素的 VQ indices。

### Mixed-modality generation / 混合模态生成

Inference 就是在 shared vocabulary 上做 next-token prediction。示例 prompt：“Draw a cat and describe it.” Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主决定顺序：可能先图后文，先文后图，也可能 interleave。同一个 decoder，同一个 loss。

对比 adapter VLM，后者只能 text-only generation。Chameleon 重新打开了模型输出模态的问题。

### Training stability — QK-Norm, dropout, LayerNorm ordering / 训练稳定性：QK-Norm、dropout 与 LayerNorm 顺序

Early-fusion training 在大规模下不稳定。Chameleon 论文记录了三个技巧：

- QK-Norm。在 attention 内部，对 query 和 key projections 做 LayerNorm，再 dot product。防止深层 logit magnitude 爆炸。许多 2024 年后的大模型都采用。
- Dropout placement。在每个 residual-add 后做 dropout，而不只是 attention 和 MLP 后。图像 token 的梯度可能支配训练，需要更强正则。
- LayerNorm ordering。Residual branch 使用 Pre-LN（标准），并在最后一个 block 的 skip connection 上额外加 LN。稳定 final-layer gradient flow。

没有这些技巧，34B-param Chameleon 在多个 checkpoint 发散。有了它们才收敛。训练 recipe 和架构同等重要。

### The tokenizer's reconstruction ceiling / Tokenizer 的重建上限

VQ-VAE 是有损的。8192 个 codebook entries、512x512 图像 1024 tokens 时，reconstruction PSNR 大约封顶在 26-28 dB。足以生成可识别图像，但明显差于 continuous-space diffusion（Stable Diffusion 3 达到 32+ dB）。

Tokenizer 是瓶颈。更好的 tokenizer（MAGVIT-v2、IBQ、SBER-MoVQGAN）会抬高上限。Emu3（Lesson 12.12）仅靠更好的 tokenizer 就达到 SDXL-quality generation。

### Chameleon vs BLIP-2 / LLaVA / 与 BLIP-2 / LLaVA 对比

Chameleon（early fusion, shared vocab）：

- 一个 loss，一个 decoder。
- 生成 mixed-modality output。
- Tokenizer 是质量上限。
- 昂贵：inference path 上生成图像时需要 VQ-VAE decoder。

BLIP-2 / LLaVA（late fusion, separate towers）：

- Vision in, text out only。
- 复用 pretrained LLM。
- 理解任务没有 tokenizer bottleneck。
- 便宜：single forward pass。

按任务选择。如果需要 image generation，选 Chameleon family。如果只需要 understanding，adapter-VLM 更简单，也复用更多 pretrained compute。

### Fuyu and AnyGPT / Fuyu 与 AnyGPT

Fuyu（Adept, 2023）是相关方案：完全跳过单独 vision encoder，把 raw image patches 通过 LLM input projection 当作 tokens 喂入，没有 tokenizer。比 Chameleon 简单，但失去 shared-vocab output generation。

AnyGPT（Zhan et al., 2024）把 Chameleon 扩展到四个 modality：text、image、speech、music。每个 modality 都用 VQ-VAE 技巧，共享 transformer。Any-to-any generation。Lesson 12.16 会展开。

## Build It / 动手构建

本课构建一个 toy early-fusion stack：先把小图像 patch quantize 成 codebook indices，再把 text ids、image ids 和 separators 放进同一个 vocabulary，最后用一个极小 autoregressive decoder 生成 interleaved sequence。重点是信号流，而不是模型规模。

## Use It / 应用它

`code/main.py` 构建一个 toy end-to-end early-fusion model：

- 一个 tiny VQ-VAE-style quantizer，把 8x8 patches 映射到 codebook indices（K=16）。
- 一个 shared vocabulary：(text ids 0..31) + (image ids 32..47) + (separators 48, 49)。
- 一个 toy autoregressive decoder（bigram table），在 synthetic captions + image-token sequences 上训练。
- Sampling loop，给定 prompt 后输出交替的 text + image tokens。

代码刻意把 transformer 缩小到 bigrams，这样你能从头到尾追踪信号流。

## Ship It / 交付它

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定 product spec（只理解 vs 理解 + 生成、所需 image quality、cost budget），它会在 Chameleon-family（early fusion）与 LLaVA-family（late fusion）之间选择，并给出 quantitative rules of thumb。

## Exercises / 练习

1. Chameleon 使用 K=8192 个 codebook entries，512x512 图像 1024 tokens。估算相对 24-bit RGB 图像的 compression ratio。它是有损的吗？损失多大？

2. 在同样 VQ-VAE 密度下，一张 4K 图像（3840x2160）会产生多少 image tokens？Chameleon-style model 能一次 inference 生成 4K 图像吗？先坏掉的是 context、tokenizer quality 还是 KV cache？

3. 用 pure Python 实现 QK-Norm。给定 64-dim query 和 key，展示 LayerNorm 前后的 dot product。为什么深层模型里 magnitude control 很重要？

4. 阅读 Chameleon Section 2.3 关于 training stability。描述论文观察到的 34B 无 QK-Norm 失败模式。所谓 “norm explosion” signature 是什么？

5. 扩展 toy decoder，让它在 text-only prompt 下输出 mixed-modality response。训练数据分布为 60% text-first / 40% image-first 时，测量模型选择 image-first vs text-first 的频率。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Early fusion | “Unified tokens” | 图像从 step one 起被转换为与 transformer vocabulary 共享的 discrete tokens |
| VQ-VAE | “Image tokenizer” | CNN + ViT + codebook，把图像映射到 transformer 可预测的 integer indices |
| Shared vocabulary | “One dictionary” | 单个 token ID space，覆盖 text + image + modality separators |
| QK-Norm | “Attention stabilizer” | 在 query 与 key dot product 前做 LayerNorm，防止 norm blowup |
| Mixed-modality generation | “Text + image output” | 一次 inference 自主生成 interleaved text 和 image tokens |
| Codebook size | “K entries” | VQ-VAE 可量化到的离散向量数；在 compression 与 fidelity 之间取舍 |
| Tokenizer ceiling | “Reconstruction limit” | 解码 VQ tokens 能达到的最佳 PSNR；决定模型图像质量上限 |

## Further Reading / 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
