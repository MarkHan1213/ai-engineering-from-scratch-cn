# Differential Attention (V2) / 差分 Attention（V2）

> Softmax attention 会把少量概率分散到每个不匹配 token 上。超过 100k tokens 后，这些噪声会累积起来并淹没信号。Differential Transformer（Ye et al., ICLR 2025）通过把 attention 写成两个 softmax 的差来修复它，从而减去共享 noise floor。DIFF V2（Microsoft，2026 年 1 月）是面向 production stack 的重写：decode latency 匹配 baseline Transformer，无需 custom kernels，并且兼容 FlashAttention。本课从 V1 讲到 V2，并用 stdlib Python 提供一个可运行的 toy implementation 来实现差分操作。

**类型：** Build
**语言：** Python (stdlib)
**前置要求：** Phase 7 · 02（self-attention），Phase 7 · 15（attention variants），Phase 10 · 14（architecture walkthrough）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 精确说明 softmax attention 为什么存在 noise floor，以及它为什么会随 context length 增长
- 推导 differential attention 公式，并解释为什么减法会抵消共享噪声成分，同时保留信号
- 走读 V1 到 V2 的 diff：什么变快了、什么变简单了、什么变稳定了，以及每个变化为什么是 production pre-training 所必需的
- 用纯 Python 从零实现 differential attention，并在 synthetic signal-plus-noise query 上经验验证 noise-cancellation property

## The Problem / 问题

标准 softmax attention 有一个数学性质，在 scale 上会变成操作层面的麻烦。对 query `q`，attention weights 是 `softmax(qK^T / sqrt(d))`。Softmax 永远不能产生精确零，每个不匹配 token 都会得到一些正概率质量。这部分 residual mass 就是噪声，而且它会随 context length 变大。128k tokens 时，即使每个不匹配 token 只拿到 0.001% 概率，127,999 个 token 合起来也会贡献约 12% 的总量。模型必须学会绕过一个随 context 增长的 noise floor。

经验上，这表现为 attention-head interference：long-context RAG 中 hallucinated citations、100k-token retrieval tasks 上 lost-in-the-middle failure，以及 needle-in-haystack benchmarks 超过 32k 后的微妙准确率退化。Differential Transformer 论文（arXiv:2410.05258, ICLR 2025）测量了这个差距：同等尺寸下，DIFF Transformers 的 perplexity 更低、long-context accuracy 更高，hallucinations 更少。

DIFF V1 有三个问题，让它无法进入 frontier pre-training pipelines。它的 value cache 每个 decode step 必须加载两次，需要破坏 FlashAttention 兼容性的 custom CUDA kernels，而且 per-head RMSNorm 在 70B+ scale 的长跑训练中不稳定。DIFF V2（Microsoft unilm blog，2026 年 1 月 20 日）修复了这三点。本课会走读两个版本，构建 difference operator，并在 toy query 上 benchmark noise cancellation。

## The Concept / 概念

### The noise floor of softmax / Softmax 的 noise floor

对 query `q` 和 keys `K = [k_1, ..., k_N]`，attention weights 为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 会是零。如果 `k_i` 与 `q` 完全无关，score `q . k_i` 也不是 0，而是围绕 0 波动，方差为 `||q||^2 / d`。经过 softmax normalization 后，每个无关 token 仍然会对 weighted sum 贡献 `O(1/N)`。无关 token 的总贡献是 `O((N-1)/N) = O(1)`，这不是小量。

模型真正想要的是类似 hard top-k 的东西：匹配 token 上高权重，其他位置接近零。Softmax 太平滑，无法直接做到。

### The differential idea / 差分思想

把每个 head 的 Q 和 K projections 分成两份：Q = (Q_1, Q_2)，K = (K_1, K_2)。计算两个 attention maps：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法会抵消两张 maps 共享的 noise distribution。如果两张 maps 都对 127k 个无关 token 给出近似 uniform weight（随机初始化时会如此），这些部分会相互抵消。信号，也就是少数真正相关 token 上的 peaked weight，只有在两张 maps 上幅度完全一致时才会被抵消；训练后它们不会如此。

`lambda` 是每个 head 的 learnable scalar，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负数。`lambda_init` 默认是类似 0.8 的小正数。

### Why this matches headed noise-canceling / 为什么它像定向降噪

想象两个带噪声的麦克风录下同一个声音。二者都捕捉到说话人和相关背景噪声。把一个信号从另一个中减掉，共享噪声会下降。声音能保留下来，是因为两路信号在相位或幅度上有足够差异，不会完全抵消。per-head `lambda` 学到的正是这个平衡。

### V1 vs V2: the diff / V1 与 V2 的差异

V1 保持参数量与 baseline Transformer 相同。为了得到每个 head 的两个 queries，它把 head dimension 减半。这损失了 head expressiveness，更痛的是每个 head 的 value cache 也被减半。Decode 每步必须加载 value cache 两次（每个 softmax branch 一次）。结果是：尽管参数量匹配，decode 仍然比 baseline 慢。

V2 加倍 query heads 数量，并保持 KV heads 不变（从 up-projection 借参数）。head dimension 与 baseline 相同。减法之后，额外维度会被投影回去，以匹配 baseline Transformer 的 O_W projection。三件事同时发生：

1. Decode speed 匹配 baseline（KV cache 只加载一次）。
2. FlashAttention 原样运行（无需 custom kernel）。
3. Decode 的 arithmetic intensity 提升（每从 HBM 加载一个 byte，会做更多 compute）。

V2 还移除了 V1 用于稳定减法的 per-head RMSNorm。在 70B 级 pre-training scale 上，那个 RMSNorm 会在训练后期引入不稳定。V2 用更简单的 initialization scheme 替代它，在没有额外 module 的情况下保持训练稳定。

### When to reach for it / 什么时候使用它

| Workload | Benefit |
|----------|---------|
| Long-context RAG (64k+) | Cleaner attention maps, fewer hallucinated citations |
| Needle-in-haystack benchmarks | Substantial accuracy lift past 32k |
| Multi-document QA | Less cross-document interference |
| Code completion at 8k | Marginal, not worth the architecture change |
| Short chat (< 4k) | Essentially indistinguishable from baseline |

收益随 context length 增长而增长。4k tokens 时，noise floor 足够小，standard attention 没问题。到 128k 时，它已经在伤害你。

### How it stacks with other 2026 knobs / 它与其他 2026 架构旋钮如何叠加

| Feature | Compatible with DIFF V2? |
|---------|------------------------|
| GQA | Yes (V2 increases Q heads, not KV heads) |
| MLA (DeepSeek) | Yes in principle, no published paper combining them |
| MoE | Yes (attention is independent of MLP block) |
| RoPE | Yes (unchanged) |
| YaRN / long-context scaling | Yes (exactly where DIFF helps most) |
| FlashAttention | Yes in V2 (was no in V1) |
| Speculative decoding | Yes (attention change is invisible to the spec-decode loop) |

```figure
differential-attention
```

## Build It / 动手构建

`code/main.py` 用纯 Python 实现 differential attention。一个具有已知 signal-plus-noise 结构的 toy query，让你可以直接测量 noise-cancellation ratio。

### Step 1: standard softmax attention / 步骤 1：标准 softmax attention

stdlib 矩阵操作：lists of lists、手写 matmul、带 max subtraction 数值稳定性的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Step 2: split Q, K into two halves / 步骤 2：把 Q、K 分成两半

V1 风格：把 head dimension 减半。V2 风格：保持 head dimension，并把 heads 数量翻倍。toy implementation 为了教学清晰使用 V1；数学相同，只有 bookkeeping 不同。

### Step 3: two softmax branches + subtraction / 步骤 3：两个 softmax branches 加减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出 weights 可以为负。这没问题，value cache 仍然处理 signed contributions。后续 V projection 会吸收符号。

### Step 4: noise cancellation measurement / 步骤 4：测量 noise cancellation

构建长度为 1024 的 synthetic sequence。把 signal token 放在已知位置，其余填充 noise。计算（a）standard softmax attention 在 signal position 上的 weight，以及（b）differential attention 的 weight。测量两者的 signal-to-noise ratio。DIFF attention 通常会产生高 3x-10x 的 signal-to-noise ratio，具体取决于两条 branches 被训练得有多不同。

### Step 5: V1 vs V2 parameter accounting / 步骤 5：V1 与 V2 参数核算

给定一个 config（hidden=4096, heads=32, d_head=128），打印：

- Baseline Transformer：Q、K、V 各自大小为 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 各自大小为 `hidden * hidden`，V 大小为 `hidden * hidden`（不变），内部 head dim 减半。增加 per-head `lambda` parameters（O(heads * d_head)）。
- DIFF V2：Q 大小为 `2 * hidden * hidden`，K 大小为 `hidden * hidden`，V 大小为 `hidden * hidden`。额外维度在 O_W 前投影回去。增加同样的 `lambda` parameters。

toy 会测量 V2 的额外参数成本（每个 attention block 大约额外 `hidden * hidden`）并打印出来。

## Use It / 使用它

截至 2026 年 4 月，DIFF V2 尚未在所有 production inference server 中上线，但 vLLM 和 SGLang 的集成正在进行。与此同时，这个模式已经出现在：

- Microsoft 内部 long-context production models。
- 几个以 256k+ context 为目标的 open model training runs 的研究复现。
- 将 DIFF attention 与 sliding-window attention 交替层结合的 hybrid architectures。

2026 年什么时候该使用它：

- 从零训练一个以 64k+ effective context 为目标的新模型。应该从一开始加入 differential attention；之后再 retrain 很贵。
- fine-tuning 一个 long-context model，且 lost-in-the-middle failures 主导你的 eval。对 Q projections 做 LoRA 可以近似 DIFF 结构。

什么时候不要用：

- 你正在 serving 一个 long-context 性能稳定的 pre-trained dense model。对已有权重来说，retraining cost 很少能回本。
- 你的 context 总是小于 16k。noise floor 可以忽略。

## Ship It / 交付

本课会产出 `outputs/skill-diff-attention-integrator.md`。给定 model architecture、target context length、hallucination profile 和 training budget，它会生成一份把 differential attention 加入新 pre-training run 或 LoRA fine-tune 的 integration plan。

## Exercises / 练习

1. 运行 `code/main.py`。验证 synthetic query 上 differential attention 报告的 signal-to-noise ratio 高于 standard softmax attention。改变 noise amplitude，并展示 standard attention 变得不可用的 crossover point。

2. 为一个 7B-class model（hidden=4096, heads=32, d_head=128, 32 layers）计算 baseline 到 DIFF V1、baseline 到 DIFF V2 的 parameter-count delta。说明哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 paper（arXiv:2410.05258）的 Section 3 和 DIFF V2 Hugging Face blog 的 Section 2。用两句话解释为什么 V1 per-head RMSNorm 是必要的，以及为什么 V2 可以移除它而不造成 training divergence。

4. 实现一个 ablation：分别用 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完整 subtraction）计算 differential attention。在 synthetic query 上，测量 signal-to-noise 随 sweep 如何变化。找出让 signal-to-noise 最大的 `lambda`。

5. 把 toy 扩展为 GQA + DIFF V2。选择 8 个 KV heads 和 32 个 Q heads。展示 KV cache size 与同样 (8, 32) 配置的 baseline GQA model 一致。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Differential attention | “Two softmaxes minus each other” | 把 Q、K 分成两半，计算两张 softmax maps，从第一张中减去第二张（乘以 lambda），再乘 V |
| Noise floor | “The non-zero tail of softmax” | softmax 给每个无关 token 的 O(1/N) 权重；在长上下文中累加为 O(1) |
| lambda | “The subtraction scale” | per-head learnable scalar，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | “The ICLR 2025 version” | 原始 Differential Transformer；为了保持参数量而减半 head dim，需要 custom kernel，decode 更慢 |
| DIFF V2 | “The January 2026 fix” | 加倍 Q heads 并保持 KV heads；decode speed 匹配 baseline，兼容 FlashAttention |
| Per-head RMSNorm | “The V1 stabilizer” | V1 在差分之后应用的额外 norm；V2 移除它以避免训练后期不稳定 |
| Signal-to-noise ratio | “How much attention is wasted” | true signal position 上的 weight 与无关位置 average weight 的比值 |
| Lost in the middle | “Long-context failure mode” | 长上下文中间位置文档的 retrieval accuracy 下滑现象；DIFF attention 可缓解它 |
| Arithmetic intensity | “FLOPs per byte loaded” | V2 在 decode 中通过每次 KV load 执行更多 query compute 提高的比值；对 memory-bound decode 很重要 |

## Further Reading / 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258)：原始论文，包含 noise-cancellation theory 和 long-context ablations
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2)：production-stack 重写，匹配 baseline decode，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333)：解释 subtraction 为什么能恢复 pretrained attention structure 的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900)：parameter-sharing 变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)：DIFF 所减去的 baseline Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172)：DIFF attention 瞄准的 long-context benchmark
