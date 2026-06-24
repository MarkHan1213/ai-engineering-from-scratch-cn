# Vision-Language Models — The ViT-MLP-LLM Pattern / 视觉语言模型：ViT-MLP-LLM 模式

> Vision encoder 把 image 转成 tokens。MLP projector 把这些 tokens 映射进 LLM 的 embedding space。Language model 负责剩下的事情。这个模式，ViT-MLP-LLM，就是 2026 年每个 production VLM。

**Type / 类型：** Learn + Use / 学习 + 使用
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 4 Lesson 14 (ViT), Phase 4 Lesson 18 (CLIP), Phase 7 Lesson 02 (Self-Attention)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 说清 ViT-MLP-LLM architecture，并解释三个组件各自贡献什么
- 比较 Qwen3-VL、InternVL3.5、LLaVA-Next 和 GLM-4.6V 的 parameter count、context length 和 benchmark performance
- 解释 DeepStack：为什么 multi-level ViT features 比单一 last-layer feature 更能收紧 vision-language alignment
- 在 production 中用 Cross-Modal Error Rate（CMER）测量 VLM hallucination，并根据这个信号采取行动

## The Problem / 问题

CLIP（Phase 4 Lesson 18）提供 images 和 text 的 shared embedding space，足够做 zero-shot classification 和 retrieval。它不能回答 “how many red cars are in this image?”，因为 CLIP 不生成文本，只打 similarity scores。

Vision-Language Models（VLMs），例如 Qwen3-VL、InternVL3.5、LLaVA-Next、GLM-4.6V，会把 CLIP-family image encoder 接到完整 language model 上。模型看到一张 image 加一个 question，然后生成 answer。到 2026 年，open-source VLMs 在 multimodal benchmarks（MMMU、MMBench、DocVQA、ChartQA、MathVista、OSWorld）上已经接近或超过 GPT-5 和 Gemini-2.5-Pro。

三件套（ViT、projector、LLM）已经成为标准。不同模型之间的差异在于使用哪个 ViT、哪个 projector、哪个 LLM、哪些 training data，以及怎样做 alignment recipe。一旦理解这个模式，替换任意组件就是机械操作。

## The Concept / 概念

### The ViT-MLP-LLM architecture / ViT-MLP-LLM 架构

```mermaid
flowchart LR
    IMG["Image<br/>(H x W x 3)"] --> ViT["Vision encoder<br/>(ViT, CLIP-L,<br/>SigLIP, DINOv3)"]
    ViT --> FEATS["Image tokens<br/>(N, d_vit)"]
    FEATS --> PROJ["Projector<br/>(2-4 layer MLP<br/>or Q-former)"]
    PROJ --> VTOK["Image tokens<br/>in LLM space<br/>(N, d_llm)"]
    TXT["Text prompt"] --> TOK["LLM tokenizer"]
    TOK --> TTOK["Text tokens<br/>(M, d_llm)"]
    VTOK --> CONCAT["Interleave<br/>or concat"]
    TTOK --> CONCAT
    CONCAT --> LLM["Decoder LLM<br/>(Qwen3, LLaMA, etc.)"]
    LLM --> OUT["Text answer"]

    style ViT fill:#dbeafe,stroke:#2563eb
    style PROJ fill:#fef3c7,stroke:#d97706
    style LLM fill:#dcfce7,stroke:#16a34a
```

1. **Vision encoder**：pretrained ViT（CLIP-L/14、SigLIP、DINOv3，或 fine-tuned variant）。产出 patch tokens。
2. **Projector**：一个小模块（2-4 layer MLP，或 Q-former），把 vision tokens 映射到 LLM 的 embedding dimension。大多数 fine-tuning 都发生在这里。
3. **LLM**：decoder-only language model（Qwen3、Llama、Mistral、GLM、InternLM）。按序读取 vision + text tokens，生成文本。

原则上三部分都可训练。实践中，vision encoder 和 LLM 大多保持 frozen，只训练 projector，用很低成本接入几十亿参数的信号。

### DeepStack / DeepStack

Vanilla projection 只使用最后一个 ViT layer。DeepStack（Qwen3-VL）从多个 ViT depths 采样 features 并 stack 起来。更深层携带 high-level semantics；浅层携带 fine-grained spatial 和 textural information。把两者都输入 LLM，可以缩小 “what does the image contain”（语义）和 “where exactly”（空间 grounding）之间的差距。

### Three training stages / 三个训练阶段

现代 VLMs 分阶段训练：

1. **Alignment**：freeze ViT 和 LLM。只在 image-caption pairs 上训练 projector。教 projector 把 vision space 映射到 language space。
2. **Pre-training**：unfreeze everything。在大规模 interleaved image-text data（500M+ pairs）上训练。建立模型的 visual knowledge。
3. **Instruction tuning**：在精选的 (image, question, answer) triples 上 fine-tune。教 conversational behaviour 和 task formats。这一步把 “vision-aware LM” 变成可用 assistant。

大多数 LoRA fine-tunes 会用小规模 labelled dataset 处理 stage 3。

### Model family comparison (early 2026) / 模型家族对比（2026 年初）

| Model | Params | Vision encoder | LLM | Context | Strengths |
|-------|--------|----------------|-----|---------|-----------|
| Qwen3-VL-235B-A22B (MoE) | 235B (22B active) | custom ViT + DeepStack | Qwen3 | 256K | General SOTA, GUI agent |
| Qwen3-VL-30B-A3B (MoE) | 30B (3B active) | custom ViT + DeepStack | Qwen3 | 256K | Smaller MoE alternative |
| Qwen3-VL-8B (dense) | 8B | custom ViT | Qwen3 | 128K | Production dense default |
| InternVL3.5-38B | 38B | InternViT-6B | Qwen3 + GPT-OSS | 128K | Strong MMBench / MMVet |
| InternVL3.5-241B-A28B | 241B (28B active) | InternViT-6B | Qwen3 | 128K | Competitive with GPT-4o |
| LLaVA-Next 72B | 72B | SigLIP | Llama-3 | 32K | Open, easy to fine-tune |
| GLM-4.6V | ~70B | custom | GLM | 64K | Open-source, strong OCR |
| MiniCPM-V-2.6 | 8B | SigLIP | MiniCPM | 32K | Edge-friendly |

### Visual agents / Visual agents

Qwen3-VL-235B 在 OSWorld 上达到全球顶级水平。OSWorld 是衡量 **visual agents** 操作 GUI（desktop、mobile、web）的 benchmark。模型看到 screenshot，理解 UI，然后输出 actions（click、type、scroll）。与 tools 结合后，它可以闭环处理常见 desktop tasks。这就是大多数 2026 年 “AI PC” demos 背后的运行方式。

### Agentic capabilities + RoPE variants / Agentic 能力与 RoPE 变体

VLMs 需要知道 video 中某一 frame 发生在 **什么时候**。Qwen3-VL 从 T-RoPE（temporal rotary position embeddings）演进到 **text-based time alignment**，也就是把显式 timestamp text tokens 与 video frames 交错输入。模型看到 “`<timestamp 00:32>` frame, prompt”，就能推理 temporal relationships。

### The alignment problem / Alignment 问题

爬取数据集里 12% 的 image-text pairs 包含并未完全 grounded in image 的描述。用这种数据训练的 VLM 会悄悄学会 hallucinate：捏造 objects、读错 numbers、发明 relationships。在 production 中，这就是主导 failure mode。

Skywork.ai 提出了 **Cross-Modal Error Rate（CMER）** 来跟踪它：

```
CMER = fraction of outputs where the text confidence is high but the image-text similarity (via a CLIP-family checker) is low
```

高 CMER 意味着模型正在自信地说出没有 image grounding 的内容。监控 CMER 并把它当作 production KPI 后，他们的部署将 hallucination rate 降低了约 35%。关键不是 “fix the model”，而是 “route high-CMER outputs to human review”。

### Fine-tuning with LoRA / QLoRA / 使用 LoRA / QLoRA 微调

对大多数团队来说，完整 fine-tune 一个 70B VLM 不现实。对 attention + projector layers 做 LoRA（rank 16-64），或者用 4-bit base weights 做 QLoRA，可以放进单张 A100 / H100。成本：5,000-50,000 examples，$100-$5,000 compute，2-10 小时训练。

### Spatial reasoning is still weak / Spatial reasoning 仍然较弱

当前 VLMs 在 spatial reasoning benchmarks（above-below、left-right、counting、distance）上得分约 50-60%，高于随机但低于人类。如果你的 use case 依赖 “which object is on top of which”，必须大量验证。对纯 spatial tasks，比通用 VLM 更好的替代方案包括 specialised keypoint / pose estimator、depth model，或 detection model 加 box geometry post-processing。

## Build It / 动手构建

### Step 1: The projector / 步骤 1：Projector

这是你最常训练的部分。2-4 layer MLP，使用 GELU。

```python
import torch
import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, vit_dim=768, llm_dim=4096, hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vit_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_dim),
        )

    def forward(self, x):
        return self.net(x)
```

输入是 `(N_patches, d_vit)` token tensor。输出是 `(N_patches, d_llm)`。LLM 会把每一行 output 当作另一个 token。

### Step 2: Assemble ViT-MLP-LLM end-to-end / 步骤 2：端到端组装 ViT-MLP-LLM

这是 minimal VLM forward pass 的骨架。真实代码使用 `transformers`；这里展示 conceptual layout。

```python
class MinimalVLM(nn.Module):
    def __init__(self, vit, projector, llm, image_token_id):
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.llm = llm
        self.image_token_id = image_token_id  # placeholder token in text prompt

    def forward(self, image, input_ids, attention_mask):
        # 1. vision features
        vision_tokens = self.vit(image)                     # (B, N_patches, d_vit)
        vision_embeds = self.projector(vision_tokens)       # (B, N_patches, d_llm)

        # 2. text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, M, d_llm)

        # 3. replace image placeholder tokens with vision embeds
        merged = self._merge(text_embeds, vision_embeds, input_ids)

        # 4. run LLM
        return self.llm(inputs_embeds=merged, attention_mask=attention_mask)

    def _merge(self, text_embeds, vision_embeds, input_ids):
        out = text_embeds.clone()
        expected = vision_embeds.size(1)
        for b in range(input_ids.size(0)):
            positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(positions) != expected:
                raise ValueError(
                    f"batch item {b} has {len(positions)} image tokens but vision_embeds has {expected} patches."
                    " Every sample in the batch must be pre-padded to the same number of image placeholder tokens.")
            out[b, positions] = vision_embeds[b]
        return out
```

Text 中的 `<image>` placeholder token 会被真实 image embeddings 替换。这也是 LLaVA、Qwen-VL 和 InternVL 使用的同一模式。

### Step 3: CMER computation / 步骤 3：CMER 计算

一个轻量 runtime check。

```python
import torch.nn.functional as F


def cross_modal_error_rate(image_emb, text_emb, text_confidence, sim_threshold=0.25, conf_threshold=0.8):
    """
    image_emb, text_emb: embeddings of image and generated text (normalised internally)
    text_confidence:     mean per-token probability in [0, 1]
    Returns:             fraction of high-confidence outputs with low image-text alignment
    """
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    sim = (image_emb * text_emb).sum(dim=-1)        # cosine similarity
    high_conf_low_sim = (text_confidence > conf_threshold) & (sim < sim_threshold)
    return high_conf_low_sim.float().mean().item()
```

把 CMER 当作 production KPI。按 endpoint、prompt type、customer 监控它。CMER 上升说明模型开始在某个 input distribution 上 hallucinate。

### Step 4: Toy VLM classifier (runnable) / 步骤 4：Toy VLM classifier（可运行）

演示 projector 可以训练。输入假的 “ViT features”；一个 tiny LLM-style token 预测 class。

```python
class ToyVLM(nn.Module):
    def __init__(self, vit_dim=32, llm_dim=64, num_classes=5):
        super().__init__()
        self.projector = Projector(vit_dim, llm_dim, hidden=64)
        self.head = nn.Linear(llm_dim, num_classes)

    def forward(self, vision_tokens):
        projected = self.projector(vision_tokens)
        pooled = projected.mean(dim=1)
        return self.head(pooled)
```

它可以在 synthetic (feature, class) pairs 上 200 steps 内拟合，足以说明 projector pattern 是有效的。

## Use It / 使用它

2026 年 production teams 使用 VLMs 的三种方式：

- **Hosted API**：OpenAI Vision、Anthropic Claude Vision、Google Gemini Vision。无需 infra，但有 vendor risk。
- **Open-source self-host**：通过 `transformers` 和 `vllm` 部署 Qwen3-VL 或 InternVL3.5。控制力完整，但 upfront effort 更高。
- **Fine-tune on domain**：加载 Qwen2.5-VL-7B 或 LLaVA-1.6-7B，在 5k-50k custom examples 上做 LoRA，用 `vllm` 或 `TGI` serving。

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image

model_id = "Qwen/Qwen3-VL-8B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": Image.open("plot.png")},
        {"type": "text", "text": "What does this chart show?"},
    ],
}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
generated = model.generate(**inputs, max_new_tokens=256)
answer = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

`apply_chat_template` 隐藏了 `<image>` placeholder tokenisation；模型会在内部处理 merge。

## Ship It / 交付内容

本课会产出：

- `outputs/prompt-vlm-selector.md`：根据 accuracy、latency、context length 和 budget，在 Qwen3-VL / InternVL3.5 / LLaVA-Next / API 之间做选择。
- `outputs/skill-cmer-monitor.md`：生成代码，用 cross-modal error rate 给 production VLM endpoint 加 instrumentation、per-endpoint dashboards 和 alerting thresholds。

## Exercises / 练习

1. **（Easy）** 用任意 open VLM，在 5 张图片上运行三个 prompts（“what is this?”、“count the objects”、“describe the scene”）。手动把每个 answer 标成 correct / partially correct / hallucinated。计算一个 first-pass CMER-like rate。
2. **（Medium）** 用 LoRA（rank 16）在目标领域的 500 张 images + captions 上 fine-tune Qwen2.5-VL-3B 或 LLaVA-1.6-7B。比较 zero-shot 与 fine-tuned 的 MMBench-style accuracy。
3. **（Hard）** 把 VLM 的 image encoder 从默认 SigLIP/CLIP 替换成 DINOv3。只重新训练 projector（frozen LLM + frozen DINOv3）。测量 dense-prediction tasks（counting、spatial reasoning）是否提升。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| ViT-MLP-LLM | “The VLM pattern” | Vision encoder + projector + language model；每个 2026 年 VLM 的模式 |
| Projector | “The bridge” | 2-4 layer MLP（或 Q-former），把 vision tokens 映射到 LLM embedding space |
| DeepStack | “Qwen3-VL feature trick” | Stack 多层 ViT features，而不是只用 last-layer |
| Image token | “<image> placeholder” | Text stream 中的特殊 token，会被 projected vision embeddings 替换 |
| CMER | “Hallucination KPI” | Cross-Modal Error Rate；当 text confidence 高但 image-text similarity 低时偏高 |
| Visual agent | “VLM that clicks” | 带 tool calls 操作 GUIs（OSWorld、mobile、web）的 VLM |
| Q-former | “Fixed-count token bridge” | BLIP-2 风格 projector，产生固定数量的 visual query tokens |
| Alignment / pre-training / instruction tuning | “Three stages” | 标准 VLM training pipeline |

## Further Reading / 延伸阅读

- [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631)
- [InternVL3.5 Advancing Open-Source Multimodal Models (arXiv 2508.18265)](https://arxiv.org/html/2508.18265v1)
- [LLaVA-Next series](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)
- [BentoML: Best Open-Source VLMs 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [MMMU: Multi-discipline Multimodal Understanding benchmark](https://mmmu-benchmark.github.io/)
- [VLMs in manufacturing (Robotics Tomorrow, March 2026)](https://www.roboticstomorrow.com/story/2026/03/when-machines-learn-to-see-like-experts-the-rise-of-vision-language-models-in-manufacturing/26335/)
