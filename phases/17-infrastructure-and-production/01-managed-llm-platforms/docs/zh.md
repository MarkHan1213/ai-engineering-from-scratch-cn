# Managed LLM Platforms — Bedrock, Vertex AI, Azure OpenAI / 托管 LLM 平台：Bedrock、Vertex AI、Azure OpenAI

> 三家超大规模云厂商，三套不同策略。AWS Bedrock 是模型市场：Claude、Llama、Titan、Stability、Cohere 都放在同一个 API 后面。Azure OpenAI 是独家 OpenAI 合作关系，再加上用于专属容量的 Provisioned Throughput Units（PTUs）。Vertex AI 以 Gemini 为中心，长上下文和多模态叙事最完整。2026 年，Artificial Analysis 在 Llama 3.1 405B 等价负载上测得 Azure OpenAI 中位延迟约 50 ms，Bedrock 约 75 ms；PTUs 解释了差距，因为专属容量通常胜过共享 on-demand。决策规则不是“谁最快”，而是“哪套模型目录和 FinOps 界面匹配我的产品”。本课教你把取舍写清楚后再选，而不是凭感觉。

**类型：** 学习
**语言：** Python（stdlib, toy cost-and-latency comparator）
**前置知识：** 第 11 阶段（LLM Engineering）, 第 13 阶段（Tools & Protocols）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出三种平台策略（marketplace vs exclusive vs Gemini-first），并把它们匹配到产品用例。
- 解释 Azure OpenAI 的 Provisioned Throughput Units（PTUs）买到的是什么，以及为什么在 405B 规模上，on-demand Bedrock 通常看起来慢约 25 ms。
- 画出每个平台的 FinOps 归因界面：Bedrock Application Inference Profiles、Vertex project-per-team、Azure scopes + PTU reservations。
- 写下一条“two-provider minimum”策略，并解释为什么 2026 年单一厂商锁定会变成昂贵错误。

## The Problem / 问题

你为产品选了 Claude 3.7 Sonnet。现在你要把它服务出来。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者放到 gateway 后面。直接 API 最简单；Bedrock 增加 BAAs、VPC endpoints、IAM 和 CloudWatch 归因；gateway 增加跨 provider 的 failover、统一账单和 rate limits。

更深层的问题是目录。如果同一个产品里同时需要 Claude、Llama 和 Gemini，你无法从一个地方买齐，除非那个“地方”同时包含 Bedrock、Vertex 和 Azure OpenAI。超大规模云厂商不是可互换的；它们分别押注了模型层归属的不同答案。

本课会把这三种押注、延迟差距、FinOps 差距和 lock-in 风险铺开。

## The Concept / 概念

### Three strategies / 三种策略

**AWS Bedrock** — marketplace。Claude（Anthropic）、Llama（Meta）、Titan（AWS first-party）、Stability（image）、Cohere（embeddings）、Mistral，再加上图像和 embedding 子目录。一个 API，一个 IAM 界面，一个 CloudWatch export。Bedrock 的押注是：客户更想要可选项，而不是单一模型。

**Azure OpenAI** — exclusive partnership。你在 Azure 数据中心获得 GPT-4 / 4o / 5 / o-series、DALL·E、Whisper，以及 OpenAI 模型的 fine-tuning。“Azure OpenAI Service”目录里没有非 OpenAI 模型；这些会进入 Azure AI Foundry（另一个产品）。Azure 的押注是 OpenAI 会继续站在 frontier，并且客户想要围绕这段特定关系的企业级控制。

**Vertex AI** — Gemini 优先，其他模型其次。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，加上 Model Garden（第三方）。Vertex 的押注是多模态长上下文：1M-token Gemini context 是差异点。

### Latency gap at scale / 规模化延迟差距

Artificial Analysis 持续跑基准。在等价 Llama 3.1 405B 部署（shared on-demand）上，Azure OpenAI 的 median first-token latency 约 50 ms；Bedrock 约 75 ms。这个差距不是 AWS 失败，而是容量模型不同。Azure 销售 PTUs（Provisioned Throughput Units），为你的 tenant 预留 GPU 容量。Bedrock 的等价能力（Provisioned Throughput）也存在，但大约从每 unit 每小时 $21 起，多数客户仍留在 shared on-demand。

On-demand shared capacity 会和其他所有客户的流量竞争。Dedicated capacity 不会。如果你的产品 SLA 是 TTFT < 100 ms at P99，要么买 Azure PTUs，要么买 Bedrock Provisioned Throughput，要么接受默认方差。

### Provisioned Throughput economics / Provisioned Throughput 经济账

Azure PTUs：一块预留推理计算资源。对可预测工作负载而言，相比 on-demand 最高可节省约 70%。成本按小时固定，不管有没有流量；空闲时也要为预留付费。break-even 通常在 40-60% sustained utilization 附近。

Bedrock Provisioned Throughput：根据模型和 region，每小时约 $21-$50。数学相似：break-even 大约在峰值利用率的一半。需要月度承诺。

Vertex provisioned capacity 按 Gemini SKU 销售；价格随模型和 region 变化，公开程度更低。

### FinOps surface — the real differentiator / FinOps 界面才是真差异

**Bedrock Application Inference Profiles** 是 marketplace 里最干净的归因方式。给 profile 打上 `team`、`product`、`feature`；所有模型调用都通过它路由；CloudWatch 可以按 profile 拆出成本，不需要后处理。这个能力 2025 年加入，至今仍是 hyperscaler native 里粒度最高的。

**Vertex** 的归因方式是 project-per-team 加 labels-everywhere。把每个团队建模为一个 GCP project，给所有资源打 label，再用 BigQuery Billing Export + DataStudio 汇总。工作量更大，但 BigQuery 允许你对成本数据写任意 SQL。

**Azure** 依赖 subscription/resource-group scopes 加 tags，并把 PTU reservations 作为一等成本对象。Tags 从 resource groups 继承，不来自 request，所以 per-request attribution 需要 Application Insights custom metrics，或一个会打 header 的 gateway。

模式是：Bedrock 原生最干净，Vertex 借 BigQuery 最灵活，Azure 不自行埋点就最不透明。

### Lock-in is the 2026 risk / 2026 年的风险是 lock-in

当一个模型统治市场时，绑定单一 hyperscaler 还可以接受。2026 年 frontier 每月移动：一个季度是 Claude 3.7，下个季度是 Gemini 2.5，再下个季度是 GPT-5。锁定一个平台，就会把你挡在三分之二的 frontier 之外。

有效团队采用的模式是：任何产品关键 LLM 调用至少两个 provider。Bedrock 加 Azure OpenAI 是常见组合：Claude 来自一个，GPT 来自另一个，用同一个 gateway 做 failover。成本上浮可以忽略，因为 gateway 会做最优路由；可用性上浮在 outage 时是决定性的，比如 Azure OpenAI 2025 年 1 月事件、AWS us-east-1 outage。

### Data residency, BAAs, and regulated industries / 数据驻留、BAAs 与受监管行业

Bedrock：多数 region 提供 BAAs；支持 VPC endpoints；支持 guardrails。常见 fintech 默认选项。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；EU data residency；企业受监管场景默认选项。
Vertex：HIPAA、GDPR、按 region 的 data residency；Google Cloud compliance stack。

三者都满足基础勾选项。差异在数据保留策略、日志处理方式，以及 abuse-monitoring 是否读取你的流量（多数默认 opt-in；enterprise 可 opt-out）。

### Numbers you should remember / 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等价负载上的 median TTFT：约 50 ms（with PTUs）。
- Bedrock on-demand median TTFT：约 75 ms。
- Bedrock Provisioned Throughput：每 unit 每小时 $21-$50。
- Azure PTU break-even：约 40-60% sustained utilization。
- 高利用率下 PTU 相比 on-demand 的节省：最高 70%。

## Build It / 动手构建

本课的动手入口是 `code/main.py`：先把 workload 的模型需求、TTFT SLA、日流量和合规要求编码成输入，再比较三家平台在延迟、成本和归因能力上的差异。

## Use It / 应用它

`code/main.py` 会在一个合成工作负载上比较三个平台：它建模 on-demand vs PTU economics、TTFT variance 和 cost attribution fidelity。运行它，观察 PTUs 在哪里回本，以及 marketplace 的模型广度在什么时候足以抵消 TTFT 差距。

## Ship It / 交付它

本课产出 `outputs/skill-managed-platform-picker.md`。给定一个 workload profile（需要的模型、TTFT SLA、daily volume、compliance requirements），它会推荐 primary platform、fallback，以及 FinOps instrumentation plan。

## Exercises / 练习

1. 运行 `code/main.py`。对一个 70B class model，Azure PTU 在什么 sustained utilization 下优于 on-demand？计算 break-even，并与宣传的 40-60% 区间比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个 two-provider deployment：哪个模型放在哪个 hyperscaler，前面放什么 gateway，failover policy 是什么？
3. 一个受监管的 healthcare 客户要求 BAAs、US-East data residency，以及 sub-100ms P99 TTFT。选择平台，并用三个具体功能说明理由。
4. 你发现本月 Bedrock 账单在流量不变时涨了 4x。没有 Application Inference Profiles 时，你会如何找元凶？有 profiles 时，需要多久？
5. 阅读 Azure OpenAI 和 Bedrock pricing pages。对于一个 100M-token/month 的 Claude workload，哪个更便宜：direct Anthropic API、Bedrock on-demand，还是 Bedrock Provisioned Throughput？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Bedrock | “AWS LLM service” | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | “Azure's ChatGPT” | Azure 数据中心里的独家 OpenAI 模型，带企业控制 |
| Vertex AI | “Google's LLM” | Gemini-first 平台，Model Garden 提供第三方模型 |
| PTU | “dedicated capacity” | Provisioned Throughput Unit：预留推理 GPU，按小时计价 |
| Application Inference Profile | “Bedrock tagging” | 带 tags 的 per-product cost/usage profile，CloudWatch-native |
| Model Garden | “Vertex catalog” | Vertex AI 的第三方模型区，与 Gemini 分离 |
| Two-provider minimum | “LLM redundancy” | 每条关键 LLM 路径都跨 ≥2 个 hyperscalers 运行的策略 |
| BAA | “HIPAA paperwork” | Business Associate Agreement；PHI 必需；三家都提供 |
| Abuse monitoring | “the log watcher” | Provider-side 对 prompts/outputs 的安全扫描；enterprise 可 opt-out |

## Further Reading / 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — 权威 rate card 和 Provisioned Throughput pricing。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU economics 和 rate cards。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini tiers 和 Model Garden surcharges。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — 跨 providers 的持续 latency 和 throughput benchmarks。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — attribution mechanics 横向对比。
