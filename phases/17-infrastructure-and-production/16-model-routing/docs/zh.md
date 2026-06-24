# Model Routing as a Cost-Reduction Primitive / 把 Model Routing 当成降本原语

> Dynamic broker 会评估每个请求（task type、token length、embedding similarity、confidence），把简单查询发给便宜模型，把复杂查询升级到 frontier model。也叫 model cascading。生产案例显示，在 US/UK/EU deployments 中，iso-quality 下可降本 20-60%；高流量 SaaS 上 30% routing efficiency improvement 会转化为六位数年节省。2026 年背景是 LLM inference 价格每年约下降 10x：GPT-4-class token 从 2022 年末的 $20/M 降到 2026 年约 $0.40/M。大部分下降来自更好的 serving stacks（Phase 17 · 04-09），不是硬件。Routing 是你在不造成产品退化的前提下，把价格下降转成 margin 的方式。Failure mode 是 cheap-model drift：route 把 40% 发给弱模型，reasoning tasks 质量下降 3-5%，一个季度没人注意。用 online quality metrics gate routes，不要只靠 offline eval sets。

**类型：** 学习
**语言：** Python（stdlib, toy cascading router simulator）
**前置知识：** 第 17 阶段 · 01（Managed LLM Platforms）, 第 17 阶段 · 19（AI Gateways）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 model cascading：cheap-first with confidence check，低 confidence 时 escalate。
- 枚举四个 routing signals（task classification、prompt length、embedding similarity to known-hard set、self-confidence from first-pass）。
- 在目标 routing split 和 quality loss tolerance 下计算 expected blended cost。
- 说出 drift-monitoring metric（online quality gate），用于捕捉 cheap-model creep。

## The Problem / 问题

你的服务在 GPT-5 上每月花 $80k。Analytics 显示 70% 查询很简单：“what time is it in Paris?” “rephrase this sentence.” Haiku-class model 可以用 3% 的成本完美处理。30% 需要 GPT-5 的 reasoning：coding、math、multi-step planning。

如果把 70% 路由到 cheap，30% 路由到 expensive，账单会在产品质量不变时下降约 65%。这就是 routing。难点是构建 broker，同时不让质量回退。

## The Concept / 概念

### Four routing signals / 四个 routing signals

1. **Task classification**：simple/complex/codegen/math/chat。可以用 rules-based classifier、小 LLM（Haiku-class at $0.25/M），或 embedding similarity to labeled buckets。输出：route = cheap / balanced / frontier。

2. **Prompt length**：>4K tokens 的 prompts 通常需要 frontier 保持 coherence。<500 tokens 的 prompts 通常不需要。

3. **Embedding similarity to known-hard set**：如果 query 与 known-hard bucket 接近（cosine > 0.88），直接 escalate to frontier。

4. **Self-confidence from first-pass**：先发 cheap；如果模型 log-probs 显示 low confidence，或拒绝，或输出 hedging language，就在 frontier 上 retry。会给约 10% 流量增加 P95 latency，但在其他 90% 上节省 50%+。

### Three patterns / 三种模式

**Pre-route**（前置 classifier）：增加约 5-10ms latency；整体最快。

**Cascade**（cheap-first，低 confidence escalate）：median latency 约 1.2x（cheap run 加 verify），升级时约 2x。质量底线最好。

**Ensemble route**（并行跑 cheap 和 frontier，对 sample 用 reward-model 选择）：质量最高、成本最高；仅用于关键 A/B。

### Implementation / 实现

AI gateways（Phase 17 · 19）暴露 routing。LiteLLM 有带 fallback 和 cost-routing 的 `router` config。Portkey 有 guards + routing。Kong AI Gateway 使用 plugin-based routing。OpenRouter 的 model marketplace 暴露 recommendation API。

Open-source：RouteLLM（LMSYS）、Not Diamond（commercial）、Prompt Mule。

### The 2026 price curve / 2026 价格曲线

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | new tier |

大部分改善来自 serving efficiency：Phase 17 · 04-09 的核心内容变成了 provider-side cost drops。Routing 让你在 app layer 捕捉这些收益，而不是等待所有用户迁到 cheap tier。

### Drift is the real risk / Drift 才是真风险

你的 route 把 40% 发给 cheap model。六个月后，task distribution 变化（用户更高级，问题更长）。Router 没注意到，因为 classifier 用 Q1 data 训练。质量静默下降。没人足够大声抱怨。你在竞争对手 benchmark 中才发现落后。

用 online quality metrics gate routes：

- 每条 route 的 user thumbs-up / thumbs-down。
- 每条 route 5% held-out sample 的 automated LLM-judge。
- Escalation rate：如果 cascade up-route >30%，说明 cheap model 被过度路由。
- 每条 route 的 refusal rate。

### Numbers you should remember / 你应该记住的数字

- 2026 routing savings at iso-quality：案例显示 20-60%。
- LLM price drop 2022-2026：总体每年约 10x。
- GPT-4-level 2022 vs 2026：约 $20/M → 约 $0.40/M。
- Cascade latency impact：median 约 1.2x，escalated 约 2x（约 10% 流量）。

## Build It / 动手构建

在 `code/main.py` 中生成 mixed workload，给每个请求赋予难度、成本和质量标签，再比较 pre-route、cascade、ensemble 三种 broker 策略。

## Use It / 应用它

`code/main.py` 在 mixed workload 上模拟 pre-route、cascade 和 ensemble。它报告 blended cost、quality loss 和 escalation rate。

## Ship It / 交付它

本课产出 `outputs/skill-router-plan.md`。给定 workload 和 quality budget，它会选择 routing pattern 和 signals。

## Exercises / 练习

1. 运行 `code/main.py`。在什么 accuracy floor 下，cascade 胜过 pre-route？
2. 你的用户群 30% enterprise（complex queries）、70% free tier（simple）。设计 routing split。用哪个 online metric 做 gate？
3. 一条 route 让质量下降 2%，但节省 40%。要不要 ship？取决于产品，正反两边都论证。
4. 使用 OpenAI / Anthropic APIs 的 logprobs 实现 confidence check。初始 threshold 设多少？
5. 六个月内 escalation rate 从 8% 升到 22%。诊断三个原因及对应修复。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Model routing | “cost broker” | 每个请求动态选择模型 |
| Model cascade | “cheap-first escalate” | 先跑便宜模型，低 confidence 退到 frontier |
| Pre-route | “classify first” | 前置 classifier；不重跑 |
| Ensemble route | “parallel pick” | 并行运行多个模型，由 reward-model 选择最佳 |
| Escalation rate | “uprouted %” | cascade requests 中升级的比例 |
| RouteLLM | “LMSYS router” | OSS router library |
| Not Diamond | “commercial router” | SaaS model-routing product |
| Drift | “cheap creep” | distribution shift 发生而 router 未察觉 |
| Online quality gate | “live check” | 对 live traffic 做 automated LLM-judge sampling |

## Further Reading / 延伸阅读

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — multi-model gateway with routing primitives。
