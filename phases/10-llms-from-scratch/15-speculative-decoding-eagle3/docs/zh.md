# Speculative Decoding and EAGLE-3 / 推测解码与 EAGLE-3

> Phase 7 · Lesson 16 已经证明了数学：Leviathan rejection rule 会精确保留 verifier 的分布。本课从 2026 年生产 speculative decoding 的 training-stack 视角出发。EAGLE-3 把 draft model 从廉价近似变成了一个专门设计的小网络，它在 verifier 自己的 hidden states 上训练，又加入 training-time test loop，让训练分布与推理分布对齐。结果是 3× 到 6.5× 的端到端加速，chat 场景中 per-token acceptance rate 超过 0.9，而且没有分布层面的质量折中。2026 年的每个生产 inference stack 都会默认启用它。

**类型：** Build
**语言：** Python (stdlib)
**前置要求：** Phase 7 · 16（speculative decoding math），Phase 10 · 12（inference optimization）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 用一句话陈述 Leviathan theorem，并证明 speculative loop 产出的 samples 与 verifier 分布完全一致
- 梳理两年演进：vanilla spec-decoding（Leviathan 2023）到 EAGLE、EAGLE-2、EAGLE-3，并说出每一步移除了什么具体限制
- 根据 acceptance rate `α` 和 draft-to-verifier cost ratio `c` 计算期望加速，并为不同 regime 选择最优 draft length `N`
- 从零实现完整 speculative loop：draft、verify、从 residual 中 reject-sample、rejection 时回滚 KV cache、full acceptance 时发出 bonus token

## The Problem / 问题

70B 模型的 autoregressive decoding 在 H100 上也许只有每秒 35 tokens。GPU 远未饱和。内存带宽才是天花板：每个 token 都要从 HBM 加载 70B 权重，做一步算术，然后产出一个 float。计算单元大部分时间都在空闲。

Speculative decoding 把它变成一个真正可解的吞吐问题。一个廉价 draft 通过 `N` 次小 forward pass 提议 `N` 个 token。verifier 在 prefix 加全部 `N` 个 drafts 上只跑一次。如果 verifier 在位置 `i` 的分布与 draft 在统计意义上同意（下面会精确定义），我们就接受；如果不同意，就拒绝并从 residual distribution 采样一个修正。一次大模型 forward 最多能产出 `N+1` 个 accepted tokens，而不是一个。

真正关键的定理来自 Leviathan, Kalman, Matias（ICML 2023）：输出分布与直接从 verifier 采样会得到的分布完全一致。不是近似，而是完全一致。这就是 speculative decoding 能进入生产的全部原因：它是纯 latency optimization，没有 quality tradeoff。

Phase 7 · Lesson 16 给了你数学。本课给你 training stack。一个好的 draft 能比一个便宜 draft 多带来 2× 加速。EAGLE、EAGLE-2 和 EAGLE-3（Li et al., 2024-2025）把 “draft = 同家族小模型” 变成了一门精确的工程纪律。2026 年生产 inference server 默认使用 EAGLE-3。

## The Concept / 概念

### The invariant: Leviathan rejection sampling / 不变量：Leviathan 拒绝采样

令 `p(t)` 为 draft 在给定 prefix 下对下一个 token 的分布，`q(t)` 为 verifier 的分布。采样 draft token `d ~ p`。以 `min(1, q(d) / p(d))` 的概率接受。若拒绝，则从 residual distribution `(q - p)_+ / ||(q - p)_+||_1` 采样。得到的 samples 按 `q` 分布。这一点与 `p` 有多差无关；`p` 越差，你拒绝越频繁，但输出仍然精确。

把这样的调用连续堆叠 `N` 次，用一次 verifier forward 处理 `prefix + d_1 + ... + d_N`。verifier 会同时返回 `q_1, q_2, ..., q_{N+1}`。从左到右检查。若在位置 `j` 第一次拒绝，就从 `residual(q_j, p_j)` 采样并停止。若全部接受，则从 `q_{N+1}` 采样一个 bonus token。

### What determines speedup / 什么决定加速

令 `α` 为每个 drafted token 的期望 acceptance rate。令 `c = cost(draft) / cost(verifier)` 为成本比。每次 verifier forward 的期望 accepted token 数是：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个 accepted token 的期望总 wall time 是 `(N * c + 1) / E[accepted]`。对 `N` 最小化就能得到 sweet spot。当 `α = 0.8, c = 0.05` 时：最优 `N` 大约是 5-7，加速 3.2×。当 `α = 0.95, c = 0.02` 时：最优 `N` 大约是 8-10，加速接近 5×。

最大的杠杆是 `α`。固定 `N = 5` 时，从 `α = 0.6`（vanilla draft）提升到 `α = 0.9`（EAGLE-3），每次 verifier forward 的期望 accepted tokens 会从 2.2 提升到 4.1。同一个 verifier 获得接近 2× 的吞吐提升。

### The two-year progression / 两年演进

**Vanilla speculative（Leviathan, 2023）。** Draft model 是同家族中独立训练的小 LLM。接入简单，`α ≈ 0.6`，最多约 2× 加速。

**EAGLE-1（Li et al., 2024）。** Draft 是一个 tiny transformer，通常一两层，接收 verifier 的 last-layer hidden state 并直接预测下一个 token。由于 draft 能看到 verifier 的 feature representation，它的分布更接近 verifier。`α` 提升到 0.7-0.8。

**EAGLE-2（Li et al., 2024）。** 加入 dynamic draft tree：不再提议一条 `N` 个 token 的序列，而是提出一棵小候选树，用 verifier 一次 forward（tree attention）给所有候选打分，并沿最高概率路径前进。draft length 变成每步自适应。accepted-path token 的 `α` 提升到 0.85 以上。

**EAGLE-3（Li et al., 2025, NeurIPS）。** 又做了两处改变。第一，完全去掉 feature-prediction loss；EAGLE-1/2 训练 draft 去匹配 verifier 的 hidden states，这会限制数据继续带来的收益。EAGLE-3 直接训练 token prediction。第二，training-time test（TTT）：draft 训练期间，把 draft 自己前几步的预测重新作为输入喂回去，和 inference 时的运行方式一致。这样对齐 train 和 test distributions，并阻止 error accumulation。实测加速：chat 上最高 6.5×；在 H100 上的 SGLang batch 64 中，吞吐比 vanilla decoding 提升 38%。

### KV cache rollback / KV cache 回滚

Verification 会在一次 pass 中把 verifier 的 KV cache 扩展 `N` 个 entries。如果 rejection 发生在位置 `j`，位置 `j-1` 之后的 cache 内容就错了。常见实现有两种：写入 scratch buffer，accept 后再 commit（vLLM、TensorRT-LLM）；或者保留物理 KV cache 和 logical length，并在 reject 时 truncate。无论哪种，rollback 成本只是每层每 head 的若干 bytes，相比 forward-pass 成本可以忽略。

对 EAGLE-2 tree search，verifier 会使用遵守 tree topology 的 non-causal mask 来运行 attention。工程上有些繁琐，但计算上就是带 custom mask 的标准 flash-attention 调用。

### Draft architectures in 2026 / 2026 年的 Draft 架构

| Strategy | Draft type | `α` | Speedup | Training cost |
|----------|-----------|-----|---------|---------------|
| Vanilla | Separate small LLM | 0.55-0.70 | 1.8-2.3× | None (reuse existing small model) |
| Medusa | Extra LM heads on verifier | 0.65-0.75 | 2-3× | ~1B SFT tokens |
| EAGLE-1 | 1-layer transformer on hidden states | 0.70-0.80 | 2.5-3× | ~60B tokens |
| EAGLE-2 | EAGLE-1 + dynamic draft tree | 0.80-0.88 | 3-4× | ~60B tokens |
| EAGLE-3 | Multi-layer feature fusion + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B tokens |
| Lookahead | No draft (Jacobi iteration) | N/A | 1.3-1.6× | None |

2026 年生产环境中：vLLM 和 SGLang 在可用时默认使用 EAGLE-3，否则使用 EAGLE-2。TensorRT-LLM 为 Meta 和 NVIDIA 公开模型提供最快的 Medusa path。llama.cpp 在 CPU 部署中提供 vanilla draft。

## Build It / 动手构建

见 `code/main.py`。这里实现的是完整 Leviathan speculative loop，包括 draft-of-N、verifier parallel pass、per-position rejection、residual sampling、bonus token、KV rollback，以及经验验证输出分布与直接从 `q` 采样一致。

### Step 1: the rejection rule / 步骤 1：拒绝规则

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Step 2: residual distribution / 步骤 2：Residual distribution

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Step 3: a full speculative step / 步骤 3：完整 speculative step

`spec_step` 函数会从 `p` 中 draft `N` 个 token，然后在一次并行 `q` 评估中全部 verify。对每个 drafted token，它应用 rejection rule；第一次 rejection 时，从 residual 中采样 correction。如果全部接受，就从 `q_{N+1}` 发出 bonus token。

### Step 4: KV rollback bookkeeping / 步骤 4：KV rollback bookkeeping

模拟器为每个 worker 跟踪 logical `kv_length`。接受 `k` 个 drafts 时，`kv_length += k`。在位置 `j` 拒绝时，cache 已经写过 `j`，但 logical length 会设为 `prefix_length + j + 1`，也就是 correction token 之后一个位置。后续读取会 truncate 到 logical length。

### Step 5: the Leviathan check / 步骤 5：Leviathan 检查

运行 50,000 次 speculative steps。统计 accepted tokens 的经验分布。再与从 `q` 直接采样 50,000 次得到的分布对比。chi-square statistic 应该明显低于 critical value。定理在实践中通过。

### Step 6: speedup vs. α / 步骤 6：speedup 与 α

通过不同幅度扰动 `p` 使其偏离 `q`，扫一遍 draft quality。测量 `α`，再画出不同 `α` 和 `N` 下每次 verifier call 的 expected tokens。代码会打印一张表，展示 EAGLE-3 级别 draft quality（`α ≈ 0.9`）如何解锁每次 verifier call 4-5 个 token。

## Use It / 使用它

使用 EAGLE-3 的 production-level `vllm serve`：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

根据 EAGLE-3 论文，在 H100 上 batch 64 的 SGLang 中，EAGLE-3 比 batch-64 vanilla decoding 大约多 1.38× 吞吐。

什么时候使用 speculative decoding：

- 任何 p50 latency 比 peak throughput 更重要的 interactive chat workload。
- Code generation 和 structured output（JSON、SQL）。因为 target distribution 高度可预测，`α` 会高于 0.9。
- Long-form generation（数千 tokens）。摊销后的加速会持续有效。

什么时候不要用：

- 很小的模型（< 3B）。draft 不会比 verifier 便宜太多。
- 很小的 batch-1 CPU 部署。draft model 的内存开销可能不值得。
- very-high-temperature 的创意采样，`α` 会崩掉。

## Ship It / 交付

本课会产出 `outputs/skill-eagle3-tuner.md`。给定 inference workload（model、batch size、target latency、task profile），它会推荐 speculative-decoding strategy 和 tuning parameters（draft family、`N`、tree depth、temperature-aware switching）。

## Exercises / 练习

1. 运行 `code/main.py`。确认 Leviathan distribution check 在 50,000 samples 上的 chi-square statistic 低于 95% critical value。

2. 固定 `α` 为 0.9、`c` 为 0.04，扫 `N` 从 1 到 10。画出每次 verifier call 的 expected tokens 和实际 wall time per token。找出使 wall time 最小的 `N`，并解释曲线形状。

3. 修改代码来模拟 EAGLE-2 tree search：每一步 draft 提出形状为 `[2, 2, 2]` 的树（8 条 candidate paths）。verifier 跑一次，概率最高的 accepted path 获胜。计算每个 leaf 的 `α` 和每次 verifier call 的总 token 数。与等价 compute 下的 linear-chain spec-decoding 对比。

4. 为两个并发序列实现 batched KV rollback simulator。Sequence A 所有 drafts 都接受；sequence B 在位置 2 拒绝。展示每个 sequence 的正确 `kv_length` 都被更新，并且没有浪费工作。

5. 阅读 EAGLE-3 paper 的 Section 4（Training-Time Test）。用两句话解释为什么没有 TTT 的 naive draft training 会遭遇 exposure bias，以及为什么训练时把 draft 自己的预测喂回去能修复它。把这与 seq2seq 中的 scheduled-sampling literature 联系起来。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Leviathan rule | “min(1, q over p)” | 以 `min(1, q(d)/p(d))` 做 Bernoulli accept/reject；rejection 时从 residual 采样，可精确保留 verifier distribution |
| Residual distribution | “(q minus p) plus, normalized” | `(q - p)_+` 截断到零并重新归一化；这是 rejection 时应采样的正确分布 |
| Acceptance rate α | “how often the draft is right” | rejection rule 下每个 token 的期望 Bernoulli-success probability；所有加速数学都由它支配 |
| EAGLE-1 | “hidden-state draft” | 条件化在 verifier last-layer hidden state 上的 tiny transformer draft（Li et al., 2024） |
| EAGLE-2 | “dynamic draft tree” | EAGLE-1 加 candidate continuation tree，在一次 verifier pass 中用 tree attention 打分 |
| EAGLE-3 | “training-time test” | 去掉 feature-prediction loss，直接训练 token prediction，并在训练中把 draft 自己的输出喂回去 |
| Training-time test (TTT) | “exposure bias fix” | 训练期间让 draft 自回归运行，使 train 与 test input distributions 匹配；是 scheduled sampling 的直接类比 |
| KV rollback | “undo rejected drafts” | rejection 后把 verifier 的 KV cache 重置到 accepted-prefix length 的 bookkeeping |
| Bonus token | “the free one” | 当所有 `N` 个 drafts 都接受时，从 `q_{N+1}` 额外采样一个，无需额外 verifier 成本 |
| Tree attention | “verify many candidates at once” | 使用尊重 draft tree topology 的 non-causal mask 的 attention；一次 forward 计算树中每个节点的 `q_i` |

## Further Reading / 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192)：奠基论文和等价性定理
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318)：几乎同期的独立提出，证明清晰
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077)：EAGLE-1，conditioned on hidden states 的 draft
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858)：dynamic tree search
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840)：2026 年生产默认方案
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774)：另一条无需独立 draft 的路线
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html)：所有策略接入完备的标准生产参考
