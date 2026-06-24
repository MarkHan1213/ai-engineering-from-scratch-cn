# Speculative Decoding — Draft, Verify, Repeat / Speculative Decoding：Draft、Verify、Repeat

> Autoregressive decoding 是串行的。每个 token 都要等前一个 token。Speculative decoding 打破这条链：cheap model 草拟 N 个 tokens，expensive model 一次 forward 验证全部 N 个。Draft 正确时，你用一次大模型 forward 换到了 N 次 generation。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 理解 speculative decoding 如何用 cheap draft model 降低 autoregressive decode latency
- 实现 acceptance/rejection step 与 residual distribution，保持 verifier distribution 不变
- 比较 vanilla speculative、Medusa、EAGLE、lookahead decoding 的 draft-verifier 设计
- 识别 acceptance rate、draft cost、KV rollback 对 production speedup 的影响

## The Problem / 问题

一个 70B LLM 在 H100 上 sample 一个 token 大约需要 30 ms。一个 3B draft model 大约需要 3 ms。如果让 3B draft 先看 5 个 tokens，再让 70B *一次*验证全部 5 个，总计是 `5×3 + 30 = 45 ms`，最多可接受 5 个 tokens；而 straight-line generation 是 `5×30 = 150 ms`。这就是 speculative decoding 的完整 pitch：用少量额外 GPU memory（draft model）换取 2–4× 更低 decode latency。

这个技巧必须保持 distribution。Leviathan et al.（2023）以及同时期 Chen et al. 引入的 speculative sampling，保证 output sequence 与 big model 单独生成时**同分布**。没有质量 tradeoff，只有更快。

2026 年 inference 中占主导的 draft-verifier pairs 有四类：

1. **Vanilla speculative (Leviathan 2023).** 单独的 draft model（例如 Llama 3 1B）+ verifier（例如 Llama 3 70B）。
2. **Medusa (Cai 2024).** 在 verifier 上加多个 decoding heads，并行预测 positions `t+1..t+k`。不需要单独 draft model。
3. **EAGLE family (Li 2024, 2025).** 轻量 draft 复用 verifier 的 hidden states；比 vanilla acceptance rate 更接近；typical 3–4×。
4. **Lookahead decoding (Fu 2024).** Jacobi iteration；完全不需要 draft model。Self-speculation。小众但 dependency-free。

2026 年每个 production inference stack 默认都带 speculative decoding。vLLM、TensorRT-LLM、SGLang、llama.cpp 至少支持 vanilla + EAGLE-2。

## The Concept / 概念

### The core algorithm / 核心算法

给定 verifier `M_q` 和更便宜的 draft `M_p`：

1. 设 `x_1..x_k` 为已经 decoded 的 prefix。
2. **Draft / 草拟**：使用 `M_p` autoregressively 提议 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，并记录 draft probabilities `p_1..p_N`。
3. **Verify in parallel / 并行验证**：在 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 上运行一次 `M_q`，得到 positions `k+1..k+N+1` 的 verifier probabilities `q_1..q_{N+1}`。
4. **Accept/reject each draft token left to right / 从左到右接受或拒绝 draft token**：对每个 `i`，以 probability `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 如果在 position `j` 第一次拒绝：从 normalized residual distribution `(q_j - p_j)_+` 中 sample `t_j`。`j` 之后所有 drafts 丢弃。
6. 如果全部 `N` 个都接受：从 `q_{N+1}` 中 sample 一个额外 token `t_{N+1}`（free bonus token）。

Residual distribution trick 是保持 output 与 `M_q` 从头 sample 完全同分布的数学关键。

### What determines speedup / 什么决定 speedup

设 `α` = 每个 draft token 的 expected acceptance rate。设 `c` = draft-to-verifier cost ratio。每一步：

- Naive generation 每 token 调用一次 big model。
- 当 `α` 较高时，speculative 每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` tokens 调用一次 big model。

经验规则：`α = 0.75`、`N = 5` 时，big-model calls 大约少 3×。Draft cost 是 5× cheap。总 wall-clock 下降约 2.5×。

**α depends on / α 取决于：**

- Draft 近似 verifier 的程度。Same family / same training data 会显著提升 α。
- Decoding strategy。Greedy draft 对 greedy verifier：α 高。Temperature sampling：更难匹配，acceptance 下降。
- Task type。Code 和 structured output 更容易接受（predictable）；free-form creative writing 接受率更低。

### Medusa — drafts without a draft model / Medusa：没有 draft model 的 drafts

Medusa 用 verifier 上的额外 output heads 替代 draft model。在 position `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个 head 输出自己的 logits。Inference 时从每个 head sample 得到 candidate sequence，再用一次 forward pass 通过 tree-attention scheme 同时验证所有 candidate continuations。

优点：没有第二个模型。缺点：增加 trainable parameters；需要 supervised fine-tuning stage（约 1B tokens）；acceptance rate 比有好 draft 的 vanilla speculative 略低。

### EAGLE — better draft by reusing hidden states / EAGLE：复用 hidden states 的更好 draft

EAGLE-1/2/3（Li et al., 2024–2025）把 draft model 做成一个 tiny transformer（通常 1 layer），输入 verifier 的 last-layer hidden states。因为 draft 看到了 verifier 的 feature representation，它的 predictions 与 verifier 的 output distribution 强相关。Acceptance rates 会从 vanilla 的 ~0.6 升到 0.85+。

EAGLE-3（2025）加入了 candidate continuations 上的 tree search。vLLM 和 SGLang 把 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认 spec pathway。

### The KV cache dance / KV cache 编排

Verification 会把 `N` 个 draft tokens 一次喂给 verifier。这会把 verifier 的 KV cache 扩展 `N` entries。如果部分 drafts 被拒绝，你必须把 cache rollback 到 accepted prefix length。

Production implementations（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）用 scratch KV buffers 处理它。先写入，接受后 commit。概念上不难，但实现细节很多。

## Build It / 动手构建

见 `code/main.py`。我们实现 core speculative-sampling algorithm（rejection step + residual distribution），包含：

- 一个 “big model”，它是手写 distribution 上的 deterministic-softmax（这样可以解析验证 acceptance math）。
- 一个 “draft model”，它是 big model 的扰动版本。
- 一个 acceptance / rejection loop，其 marginal distribution 与 direct sampling 相同。

### Step 1: the rejection step / 第 1 步：rejection step

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是 uniform random number。`q_prob` 是 verifier 对 drafted token 的 probability。`p_prob` 是 draft model 的 probability。Leviathan theorem 表明，这个 Bernoulli decision 加上拒绝时从 residual sample，能精确保持 verifier distribution。

### Step 2: residual distribution / 第 2 步：residual distribution

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素从 `q` 减 `p`，把负值 clamp 到 zero，再 renormalize。任何 rejection 都从这里 sample。

### Step 3: one speculative step / 第 3 步：一个 speculative step

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个 drafts 都接受 → 一个 bonus → 一次 verifier pass 产出六个 tokens。

### Step 4: measure acceptance rate / 第 4 步：测量 acceptance rate

在不同 draft-quality levels 下运行 10,000 speculative steps。画出 acceptance rate vs. draft 与 verifier distributions 之间的 KL divergence。你应该看到干净的 monotone relationship。

### Step 5: verify distribution equivalence / 第 5 步：验证 distribution equivalence

经验验证：speculative loop 生成 tokens 的 histogram 应匹配直接从 verifier sampling 的 histogram。这就是 Leviathan theorem 的实践版本。Chi-square test 会确认差异在 sampling error 内。

## Use It / 应用它

Production：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 有最快的 Medusa path。`faster-whisper` 为 Whisper-large 包装了 speculative decoding，使用 small draft。

**Picking a draft / 选择 draft：**

| Strategy | When to pick | Speedup |
|----------|--------------|---------|
| Vanilla draft (1B/3B Llama family) | Fast prototype, no training | 1.8–2.3× |
| Medusa heads | You can fine-tune the verifier | 2–3× |
| EAGLE-2 / 3 | Production, max speed | 3–4× |
| Lookahead | No draft, no training, no extra params | 1.3–1.6× |

**When NOT to spec-decode / 什么时候不做 spec-decode：**

- 只生成 1–5 tokens 的 single-sequence generation。Overhead 会主导。
- Wildly creative / high-temperature sampling（α 会下降）。
- Memory-constrained deployments（draft model 会增加 VRAM）。

## Ship It / 交付它

见 `outputs/skill-spec-decode-picker.md`。这个 skill 会为新的 inference workload 选择 speculative decoding strategy（vanilla / Medusa / EAGLE / lookahead）和 tuning parameters（N、draft temperature）。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。确认 50,000 tokens 上 speculative token distribution 与 verifier direct-sample distribution 匹配，chi-square p > 0.05。
2. **Medium / 中等。** 画出 `α = 0.5, 0.7, 0.85` 时 speedup（tokens per big-model forward）随 `N` 的变化。找出每个 α 的 optimal `N`。（提示：expected tokens per verify call = `(1 - α^{N+1}) / (1 - α)`。）
3. **Hard / 困难。** 实现 tiny Medusa：取 Lesson 14 的 capstone GPT，添加 3 个额外 LM heads，分别预测 positions t+2、t+3、t+4。在 tinyshakespeare 上用 joint multi-head loss 训练。与通过 truncating 同一个模型得到的 vanilla draft 比较 acceptance rates。
4. **Hard / 困难。** 实现 rollback：从 10-token prefix KV cache 开始，喂入 5 个 draft tokens，模拟在 position 3 rejection。验证下一轮 cache reads 正确匹配 “prefix + first 2 accepted drafts”。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Draft model | “The cheap one” | 提议 candidate tokens 的小模型；通常比 verifier 便宜 10–50×。 |
| Verifier | “The big one” | 我们要保持其 distribution 的 target model；每个 speculative step 运行一次。 |
| Acceptance rate (α) | “How often the draft is right” | Verifier 接受 draft 的 per-token probability。典型为 0.7–0.9。 |
| Residual distribution | “The rejection fallback” | Normalized `(q - p)_+`；rejection 时从这里 sample 可保持 verifier distribution。 |
| Bonus token | “The free one” | 当全部 N 个 drafts 被接受时，从 verifier 的 next-step distribution 再 sample 一个 token。 |
| Medusa | “Draft-less speculative” | Verifier 上多个 LM heads 并行预测 positions t+1..t+k。 |
| EAGLE | “Hidden-state draft” | Tiny transformer draft，conditioned on verifier last-layer hidden states。 |
| Lookahead decoding | “Jacobi iteration” | 使用 fixed-point iteration 的 self-speculation；不需要 draft model。 |
| Tree attention | “Verify many candidates at once” | 同时考虑多个 draft continuations 的 branching verification。 |
| KV rollback | “Undo rejected drafts” | Scratch KV buffer；acceptance 时 commit，reject 时 discard。 |

## Further Reading / 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — core algorithm 与 equivalence theorem。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同期提出；清晰的 Bernoulli-rejection proof。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 论文；tree-attention verification。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；hidden-state-conditioned draft。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；dynamic tree depth。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead，no-draft approach。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 四类策略全部接入的 canonical production reference。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 reference code。
