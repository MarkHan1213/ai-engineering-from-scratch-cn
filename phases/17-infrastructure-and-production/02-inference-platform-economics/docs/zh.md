# Inference Platform Economics — Fireworks, Together, Baseten, Modal, Replicate, Anyscale / 推理平台经济学：Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场已经不再是 GPU time rental。它分成三类：custom silicon（Groq、Cerebras、SambaNova）、GPU platforms（Baseten、Together、Fireworks、Modal）和 API-first marketplaces（Replicate、DeepInfra）。Fireworks 在 2026 年 5 月 1 日把 GPU 价格提高 $1/hr；$4B valuation 和每天 10T+ tokens 说明 volume-driven model 能跑通。Baseten 2026 年 1 月完成 $300M Series E，估值 $5B。竞争定位规则很简单：Fireworks 优化 latency，Together 优化 catalog breadth，Baseten 优化 enterprise polish，Modal 优化 Python-native DX，Replicate 优化 multimodal reach，Anyscale 优化 distributed Python。本课给你一张可以直接递给 founder 的矩阵。

**类型：** 学习
**语言：** Python（stdlib, toy per-call economics comparator）
**前置知识：** 第 17 阶段 · 01（Managed LLM Platforms）, 第 17 阶段 · 04（vLLM Serving Internals）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出三类市场分段（custom silicon、GPU platforms、API-first），并把每个 vendor 映射到分段。
- 解释为什么 “per-token” API pricing model 会向 serving engine 的 cost curve 收敛，而不是向硬件成本收敛。
- 计算至少三个 vendor 的 effective cost per request，并解释什么时候 per-minute（Baseten、Modal）胜过 per-token。
- 为给定工作负载识别正确默认平台：serverless bursty、steady high-throughput、fine-tuned variants、multimodal。

## The Problem / 问题

你评估了 managed hyperscaler platforms。你决定需要一个更窄、更快的 provider：Fireworks 追 latency，Together 追 breadth，Baseten 服务 fine-tuned custom model。现在你有六个真实选择，而 pricing pages 对不上。Fireworks 显示 $/M tokens；Baseten 显示 $/minute；Modal 显示 $/second；Replicate 显示 $/prediction。没有工作负载模型，你无法正面比较它们。

更糟糕的是，每张价格页背后的商业模式都不同。Fireworks 在 shared GPUs 上跑自研 engine（FireAttention）；per-token rate 反映的是它们的 utilization curve。Baseten 给你 Truss + dedicated GPUs；per-minute 反映 exclusivity。Modal 是真正的 Python serverless：按秒计费，sub-second cold starts。同样是一个 LLM response，却对应三种不同 cost function。

本课建模这六类选择，并告诉你什么时候谁胜出。

## The Concept / 概念

### The three segments / 三个分段

**Custom silicon** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在同一模型上，decode 通常比 GPU-based cluster 快 5-10x。per-token 价格更高（Groq 在 2025 年末 Llama-70B 约 $0.99/M），但对 latency-sensitive use cases 无可替代。Groq 是 voice agents 和 real-time translation 的生产选择。

**GPU platforms** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（2026 年 H100、H200、B200）或偶尔 AMD 上。它们位于 “raw GPU rental”（RunPod、Lambda）和 “hyperscaler managed service”（Bedrock）之间。

**API-first marketplaces** — Replicate、DeepInfra、OpenRouter、Fal。目录广，pay-per-prediction 或 pay-per-second，强调 time-to-first-call。

### Fireworks — latency-optimized GPU platform / Fireworks：延迟优化 GPU 平台

- FireAttention engine（custom）；宣传在等价配置下 latency 比 vLLM 低 4x。
- Batch tier 约为 serverless rate 的 50%，适合非交互工作负载。
- Fine-tuned model 以 base model 同价服务：这是相对于会对你的 LoRA 收 premium 的 provider 的真实差异点。
- 2026 年中：on-demand GPU rental 自 2026 年 5 月 1 日起提高 $1/hour。规模量价可谈。
- 财务信号：$4B valuation，每天处理 10T+ tokens。

### Together — breadth-optimized / Together：目录广度优化

- 200+ models，包括上游发布后数天内跟进的 open-source releases。
- 等价 LLM 模型上比 Replicate 便宜 50-70%；“AI Native Cloud” 的定位是 volume 和 catalog。
- Inference + fine-tuning + training 在同一个 API 里。

### Baseten — enterprise-polish-optimized / Baseten：企业质感优化

- Truss framework：把 dependencies、secrets、serving config 放进一个 manifest 进行模型打包。
- GPU 范围从 T4 到 B200。per-minute billing，cold-start mitigation 合理。
- SOC 2 Type II，HIPAA-ready。常见 fintech 和 healthcare 选择。
- $5B valuation，2026 年 1 月 Series E（CapitalG、IVP、NVIDIA 投资 $300M）。

### Modal — Python-native-optimized / Modal：Python-native 优化

- 纯 Python 的 infrastructure-as-code。给函数加 `@modal.function(gpu="A100")`，一条命令部署。
- Per-second billing。预热后 cold starts 2-4s；小模型 <1s。
- 2025 年 $87M Series B，估值 $1.1B。独立调查中 developer experience 评分最高。

### Replicate — multimodal breadth / Replicate：多模态广度

- Pay-per-prediction。图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS plugins）。
- LLM per-token rates 竞争力较弱，但在 multimodal variety 上胜出。

### Anyscale — Ray-native / Anyscale：Ray-native

- 构建在 Ray 上；RayTurbo 是 Anyscale 的专有 inference engine（与 vLLM 竞争）。
- 适合 distributed Python workloads：推理只是更大 graph 中的一个节点。
- Managed Ray clusters；与 Ray AIR 和 Ray Serve 深度集成。

### Per-token versus per-minute — when each wins / per-token 与 per-minute：什么时候谁赢

Per-token 适合 latency-insensitive 且 bursty 的工作负载：只为实际使用付费。Per-minute 适合利用率高且可预测的工作负载：一旦 GPU 被你持续打满，就胜过 per-token。

粗略规则：当工作负载超过 dedicated GPU 约 30% sustained utilization 时，per-minute（Baseten、Modal）开始胜过 per-token（Fireworks、Together）。低于这个值，per-token 胜出，因为你不用为空闲付费。

### Custom engine is the real moat / Custom engine 才是真护城河

vLLM 和 SGLang 之上的每个平台都声称有 custom engine：FireAttention、RayTurbo、Baseten 的 inference stack。Custom-engine 说法会带 marketing 色彩；诚实表达是：vLLM + SGLang 约代表生产 open-source inference 的 80%，平台层差异主要是 DX、attribution 和 SLAs。

### Numbers you should remember / 你应该记住的数字

- Fireworks GPU rental：2026 年 5 月 1 日起提高 $1/hr。
- Fireworks claim：等价配置下 latency 比 vLLM 低 4x。
- Together：LLMs 上比 Replicate 便宜 50-70%。
- Baseten valuation：$5B（Series E，2026 年 1 月，$300M round）。
- Modal valuation：$1.1B（Series B，2025）。
- Sustained utilization 超过约 30% 后，per-minute 胜过 per-token。

```figure
cost-per-token
```

## Build It / 动手构建

用 `code/main.py` 把同一个请求分布投到不同定价模型上，显式计算 per-token、per-minute、per-prediction 的有效成本，而不是只读 pricing page。

## Use It / 应用它

`code/main.py` 会在一个合成工作负载上，跨 pricing models 比较六个 vendors。它报告 $/day 和 effective $/M tokens。运行它，找到 per-token 与 per-minute 之间的 break-even。

## Ship It / 交付它

本课产出 `outputs/skill-inference-platform-picker.md`。给定 workload profile、SLA 和 budget，它会选择 primary inference platform，并指出 runner-up。

## Exercises / 练习

1. 运行 `code/main.py`。对于一块 H100 上的 70B model，Baseten（per-minute）在什么 sustained utilization 下胜过 Fireworks（per-token）？自己推导 crossover，并与经验规则比较。
2. 你的产品同时提供 image generation、chat 和 speech-to-text。为每个 modality 选择平台，并指出统一它们的 gateway pattern。
3. Fireworks 把你的 primary model 价格提高 $1/hr。如果 40% 流量迁到 batch tier（50% off），建模 blended cost impact。
4. 一个受监管客户要求 SOC 2 Type II + HIPAA + dedicated GPUs。哪三个平台可行？哪一个在 FinOps 上胜出？
5. 比较 Llama 3.1 70B 在 Fireworks serverless、Together on-demand、Baseten dedicated 和 Replicate API 上每 1,000 predictions 的成本。10 predictions/day 谁最便宜？10,000 呢？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Custom silicon | “non-GPU chips” | Groq LPU、Cerebras WSE、SambaNova RDU：为 decode 优化 |
| FireAttention | “Fireworks engine” | Custom attention kernel；宣传 latency 比 vLLM 低 4x |
| Truss | “Baseten's format” | 模型打包 manifest：dependencies + secrets + serving config |
| Per-token | “API pricing” | 按消耗 token 收费；不为空闲付费 |
| Per-minute | “dedicated pricing” | 按 GPU wall-clock time 收费；高利用率时胜出 |
| Per-prediction | “Replicate pricing” | 按模型调用收费；常见于 image/video |
| RayTurbo | “Anyscale engine” | Ray 上的专有 inference；在 Ray clusters 上与 vLLM 竞争 |
| Batch tier | “50% off” | 非交互队列，降价运行；Fireworks、OpenAI 常见 |
| Fine-tuned at base rate | “Fireworks LoRA” | LoRA-served requests 按 base model rate 收费（差异点） |

## Further Reading / 延伸阅读

- [Fireworks Pricing](https://fireworks.ai/pricing) — per-token rates、batch tier、GPU rental。
- [Baseten Pricing](https://www.baseten.co/pricing/) — per-minute rates、committed capacity、enterprise tiers。
- [Modal Pricing](https://modal.com/pricing) — per-second GPU rates 和 free tier。
- [Together AI Pricing](https://www.together.ai/pricing) — model catalog 和 per-token rates。
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 和 managed Ray pricing。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — comparative assessment。
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — vendor landscape。
