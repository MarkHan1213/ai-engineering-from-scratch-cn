# Differential Privacy for LLMs / 面向 LLM 的差分隐私

> DP-SGD 仍是标准做法：向 gradient updates 注入噪声，提供形式化的 (epsilon, delta) guarantees。Compute、memory 和 utility overhead 都很大；parameter-efficient DP fine-tuning（LoRA + DP-SGD）是 2025 年常见配置（ACM 2025）。两类证据存在张力：canary-based membership inference（Duan et al., 2024）报告对 language models 成功有限；training-data extraction（Carlini et al., 2021; Nasr et al., 2025）恢复了大量 verbatim memorization。Resolution（arXiv:2503.06808，2025 年 3 月）：差异在测量对象——inserted canaries vs “most extractable” data。新的 canary designs 支持无需 shadow models 的 loss-based MIA，并首次对真实数据上、现实 DP guarantees 下训练的 LLM 给出非平凡 DP audit。替代方案：PMixED（arXiv:2403.15638）——在 inference time 通过 next-token distributions 的 mixture of experts 做 private prediction；DP synthetic data generation（Google Research 2024）。Emerging attack：Differential Privacy Reversal via LLM Feedback——confidence-score leakage。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib, DP-SGD noise-injection and ε-δ accountant demonstration)
**Prerequisites / 前置知识：** Phase 01 · 09 (information theory), Phase 10 · 01 (large-model training)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 定义 (epsilon, delta)-differential privacy，并说出 DP-SGD recipe。
- 解释 2024-2025 的张力：canary MIA 与 training-data extraction 给出不同图景。
- 描述 PMixED，以及为什么 inference-time private prediction 是 DP training 的替代方案。
- 描述 Differential Privacy Reversal via LLM Feedback attack。

## The Problem / 问题

LLMs 会记忆。Carlini et al. 2021 展示 production language models 能按需复现 verbatim training text。DP 是形式化防御：训练过程让 output 对任意单个 training example 都 provably insensitive。2024-2025 的证据显示 DP-SGD 必要，但部署中的 ε values 可能不匹配 threat model。

## The Concept / 概念

### (ε, δ)-differential privacy / (ε, δ)-差分隐私

随机算法 M 是 (ε, δ)-DP，如果对任意只差一个 example 的两个 datasets，以及任意 event S：
P(M(D) in S) <= e^ε * P(M(D') in S) + δ。

解释：output distribution 足够接近（由 ε 参数化），使得任何单个个体的贡献都不能被可靠推断，例外概率为 δ。

### DP-SGD / DP-SGD

Abadi et al. 2016。标准 recipe：
1. 采样 mini-batch。
2. 计算 per-example gradients。
3. 把每个 per-example gradient clip 到 threshold C。
4. 对 clipped gradients 求和，并加入 std 为 σ * C 的 Gaussian noise。
5. 用 noisy sum 更新参数。

Privacy cost 由 accountant 跟踪（Moments Accountant、Rényi DP accountant）。LLM 文献中的 reported ε values 会随 threat model、data sensitivity 和 utility target 大幅变化；没有 universally “safe” default ε。公开例子在某些 LLM training settings 中大致覆盖 ε ≈ 1–10，但这些只是 illustrative，不是 recommended defaults。更低 ε 通常需要更多噪声，并可能增加 utility loss。

### LoRA + DP-SGD / LoRA + DP-SGD

对 frontier model 做 full DP-SGD 成本过高。LoRA（Hu et al. 2022）把 gradient updates 限制在小 adapter 上，减少 per-example gradient storage。LoRA + DP-SGD 是 2025 年常见配置。DP guarantees 作用在 adapter 上；base model 固定。

### The 2024-2025 tension / 2024-2025 的证据张力

两条证据线：

- **Canary MIA（Duan et al. 2024）。** 向 training data 插入 unique canaries，测量 membership-inference attacker 是否能识别它们。报告对 language models 成功有限。看起来 MIA 很难。
- **Training-data extraction（Carlini 2021, Nasr et al. 2025）。** 用 prefix prompt 模型，测量是否恢复 training 中的 verbatim text。报告 substantial memorization。看起来 relevant sense 下 MIA 很容易。

2025 年 3 月 resolution（arXiv:2503.06808）：二者测量不同东西。MIA 问的是 “example e 是否在 D 中？” 并使用 inserted canaries。Extraction 问的是 “我能恢复 D 的什么？” 对隐私来说，“most extractable” example 才关键；canaries 低估了这一点，因为它们没有被优化成可提取。

新的 canary designs。无需 shadow models 的 loss-based MIA。对真实数据和现实 DP guarantees 下训练的 LLM 的首次非平凡 DP audit。

### Alternatives to DP training / DP training 的替代方案

- **PMixED（arXiv:2403.15638）。** Inference-time private prediction。Next-token distributions 上的 mixture of experts；每个 expert 只看 training data shard；aggregation 加噪声以实现 DP。完全避免 DP training。
- **DP synthetic data generation（Google Research 2024）。** 用 DP-SGD 做 LoRA-fine-tune，采样 synthetic data，再用 synthetic data 训练 downstream classifier。

二者都绕开 full DP training 的 utility cost，但 threat model 不同。

### Differential Privacy Reversal via LLM Feedback / 通过 LLM Feedback 反转差分隐私

2025 emerging attack。利用 DP-trained model 的 confidence scores 作为 oracle 重新识别个体。即使 outputs 不泄漏，confidence distributions 也可能泄漏。

防御：不暴露 confidences，或在暴露前 truncate/quantize。除了 (ε, δ)-DP training，这还是额外要求。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lessons 20-21 是 bias/fairness。Lesson 22 是 privacy。Lesson 23 是 watermarking 形式的 provenance。Lesson 27 覆盖 regulatory data-provenance layer。

## Build It / 动手构建

本课实现 DP-SGD noise-injection 与 ε-δ accountant demonstration。你会 sweep noise multiplier σ 和 clipping norm C，看到 privacy budget 与 accuracy 的 trade-off，并加入 canary attack 观察 DP 前后差异。

## Use It / 应用它

`code/main.py` 在 toy binary-classification dataset 上模拟 DP-SGD。你可以 sweep noise multiplier σ 和 clipping norm C，跟踪 (ε, δ) budget 与 accuracy cost。一个 “canary attack” 会插入 unique training example，并测量 log-loss test 在 DP 前后是否能检测它。

## Ship It / 交付它

本课产出 `outputs/skill-dp-audit.md`。给定 language model deployment 的 DP claim，它会审计：(ε, δ) values、accountant、MIA evaluation protocol，以及 confidence-exposure vectors 是否被评估。

## Exercises / 练习

1. 运行 `code/main.py`。Sweep σ in {0.5, 1.0, 2.0}，报告 (ε, δ)-accuracy trade-off。识别 utility collapse 的点。

2. 实现 canary insertion 和 log-loss test。在 σ = 1.0 下测量 DP-SGD 前后的 detection rate。

3. 阅读 Nasr et al. 2025 关于 training-data extraction 的工作。为什么 extraction success 在 moderate ε 下没有 collapse？这对 MIA-as-evaluation 有什么含义？

4. 设计一个完全在 inference time 运行的 PMixED（arXiv:2403.15638）deployment。PMixED 处理了 DP-SGD 不处理的什么 threat model？

5. Sketch DP Reversal via LLM Feedback attack。设计一个限制 confidence-score leakage 的 countermeasure，并估计 deployment cost。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| DP | “(ε, δ)-differential privacy” | 形式化隐私：neighbouring-dataset change 下 output distribution 接近 |
| DP-SGD | “noise-injected SGD” | Gradient clipping + Gaussian noise addition；标准 DP training |
| LoRA + DP-SGD | “efficient private fine-tune” | 在 low-rank adapters 上做 DP-SGD；2025 标准配置 |
| MIA | “membership inference” | 判断某个 example 是否在 training data 中的 attack |
| Canary | “inserted watermark example” | 用于测量 DP leakage 的 unique training example |
| PMixED | “private inference mixture” | 通过 next-token distributions 上的 mixture-of-experts 做 inference-time DP |
| DP Reversal | “confidence leakage attack” | 把 model confidence 当作 re-identification oracle 的 attack |

## Further Reading / 延伸阅读

- [Abadi et al. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — standard DP training algorithm。
- [Carlini et al. — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — canonical extraction paper。
- [Duan et al. — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) — limited-success MIA。
- [Kowalczyk et al. — Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808) — tension resolution。
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — inference-time private prediction。
