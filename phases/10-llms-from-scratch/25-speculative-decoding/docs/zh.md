# Speculative Decoding and EAGLE / 推测解码与 EAGLE

> frontier LLM 生成一个 token，需要对数十亿参数做一次完整 forward pass。这个 forward pass 其实严重过配：大多数时候，一个小得多的模型可以正确猜出接下来 3-5 个 token，而大模型只需要 *verify* 这个猜测。猜对时，你用一次大模型的价格拿到了 5 个 token。Speculative decoding（Leviathan et al. 2023）让这件事在分布上精确成立；EAGLE-3（2025）把 acceptance rates 推到每次 verify 约 4.5 tokens，在匹配输出分布的情况下带来 4-5x speedup。

**类型：** Build
**语言：** Python (with numpy)
**前置要求：** Phase 10 Lesson 12（Inference Optimization），Phase 10 Lesson 04（Pre-training Mini-GPT）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 speculative decoding 的 two-model setup，以及 target model 与 draft model 的职责划分
- 推导 exact rejection rule 为什么能保持 target distribution 不变
- 根据 acceptance rate `α` 和 draft length `K` 估算期望 speedup
- 理解 EAGLE 的 feature reuse、tree drafting 和 tree attention verification 如何提升 acceptance

## The Problem / 问题

70B-class model 在 H100 上的 decode throughput 通常是 40-80 tokens/second。每个 token 都需要一次完整 forward pass，从 HBM 读取全部模型权重。你不能在不改变输出的情况下缩小模型。也不能无限增大 batch size，因为内存会卡住。看起来陷住了，除非你能让模型每次 forward pass 输出不止一个 token。

Autoregressive generation 看起来天然串行：`x_{t+1} = sample(p(· | x_{1:t}))`。但这里有一个并发机会。如果你有一个便宜 predictor 说“接下来 4 个 tokens 可能是 [a, b, c, d]”，你就可以用大模型的**一次 forward pass** 同时 verify 5 个位置，并接受最长匹配 prefix。

Leviathan, Kalai, Matias（2023, "Fast Inference from Transformers via Speculative Decoding"）通过一个巧妙的 accept/reject rule 让它精确成立，并保留 target model 的 sampling distribution。同样的输出分布，快 2-4×。

## The Concept / 概念

### The Two-Model Setup / 双模型设置

- **Target model** `M_p`：你真正想从中采样的大、慢、高质量模型。分布：`p(x)`。
- **Draft model** `M_q`：小、快、质量较低的模型。分布：`q(x)`。通常小 5-30×。

每一步：

1. Draft model autoregressively 提出 `K` 个 tokens：`x_1, x_2, ..., x_K ~ q`。
2. Target model 对全部 `K+1` 个位置并行运行一次 forward pass，为每个 proposed token 产出 `p(x_k)`。
3. 使用下面的 modified rejection-sampling rule 从左到右 accept/reject 每个 token。接受最长匹配 prefix。
4. 如果任意 token 被拒绝，就从修正后的分布中采样 replacement 并停止。否则从 `p(· | x_1...x_K)` 采样一个 bonus token。

如果 draft 与 target 完美匹配，你每次 target-forward 得到 K+1 个 tokens。如果 draft 在第 1 个位置就错了，你只得到 1 个 token。

### The Exactness Rule / 精确性规则

Speculative decoding **在分布上可证明等价于从 p 采样**。rejection rule：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差的正部。当 draft 与 target 一致（`p ≈ q`）时，acceptance 接近 1。当二者不同意时，residual distribution 会被构造出来，使整体 sample 仍然精确等于 `p`。

**Greedy case。** temperature=0 sampling 时，只要检查 `argmax(p) == x_t`。如果是，接受；如果不是，输出 `argmax(p)` 并停止。

### Expected Speedup / 期望加速

如果 draft model 的 token-level acceptance rate 是 `α`，每次 target-forward pass 期望产生的 tokens 数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

当 `α = 0.8, K = 4`：`(1 - 0.8^5)/(1 - 0.8) = 3.36` tokens per forward。一次 target forward 的成本大约是 `cost_q * K + cost_p`（K 个 draft steps 加一次 target verify）。如果 `cost_p >> cost_q * K`，吞吐 speedup ratio 就是 `3.36× / 1 = 3.36×`。

真正的参数只有 `α`，它完全取决于 draft-target alignment。好的 draft 就是一切。

### Training the Draft: Distillation / 训练 Draft：蒸馏

随机小模型是糟糕的 draft。标准配方是从 target distill：

1. 选择一个小架构（70B target 配约 1B，7B target 配约 500M）。
2. 在大规模 text corpus 上运行 target model，存储它的 next-token distributions。
3. 用 KL divergence 训练 draft 去匹配 target distribution（而不是 ground-truth tokens）。

结果：coding 上 `α` 通常为 0.6-0.8，natural-language chat 上为 0.7-0.85。生产中 speedups 约 2-3×。

### EAGLE: Tree Drafting + Feature Reuse / EAGLE：树状 Drafting 与 Feature Reuse

Li, Wei, Zhang, Zhang（2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）观察到 standard speculative decoding 中有两个低效点：

1. draft 做 K 个串行 steps，每个 step 都跑 full-stack。但 draft 可以复用最近一次 verify 中 target 的 features（hidden states）；target 已经计算出丰富表示，draft 没必要从头再推一遍。
2. draft 输出一条 linear chain。如果 draft 能输出一棵 candidates *tree*（每个 node 多个 guesses），target 的一次 forward pass 就能通过 tree attention mask 并行 verify 多条 candidate paths，并选择最长 accepted branch。

EAGLE-1 改动：
- Draft input = 位置 t 的 target final hidden state，而不是 raw tokens。
- Draft architecture = 1 transformer decoder layer（不是单独小模型）。
- Output = 每个 depth K = 4-8 个 candidates、depth 4-6 的 tree。

EAGLE-2（2024）加入 dynamic tree topology：draft 不确定时 tree 更宽，有信心时更窄。在不增加 verify cost 的情况下提升 `α_effective`。

EAGLE-3（Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）移除固定 top-layer feature dependency，并用新的 "test-time simulation" loss 训练 draft，也就是让 draft 在匹配 target test-time distribution 的 outputs 上训练，而不是 teacher-forced training distribution。Acceptance rate 从 EAGLE-2 的 0.75 提升到 EAGLE-3 的 0.82，mean tokens/verify 从 3.0 提升到 4.5。

### Tree Attention Verification / Tree Attention 验证

当 draft 输出一棵 tree 时，target model 使用 **tree attention mask** 在一次 forward pass 中 verify 它。这个 causal mask 编码 tree topology，而不是纯线性序列。每个 token 只 attend 到它的 ancestors。verify pass 仍是一次 forward、一次 matmul；topological mask 只增加少量 KV entries。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的 first-token candidates，`c, d, e, f` 是 second-token candidates，六个位置会在一次 forward pass 中全部 verify。输出是任意 accepted path 上的最长 prefix。

### When It Wins, When It Doesn't / 什么时候赢，什么时候不赢

**Wins：**
- Chat / completion 中可预测文本（code、常见英语、structured output）。`α` 高。
- decode 阶段 GPU compute 未被充分利用的设置（memory-bound phase）。Tree drafting 会使用可用 FLOPs。

**Loses / no win：**
- 高 stochasticity 输出（高 temperature creative writing）。`α` 会接近 `1/|vocab|`。
- 极高并发 batch serving。batching 已经填满 FLOPs，tree verification 空间很小。
- 很小 target models，draft 并没有小很多。

Production shops 通常报告 chat 上 2-3× wall-clock speedup，code generation 上 3-5×，creative writing 上接近零。

```figure
speculative-decoding
```

## Build It / 动手构建

`code/main.py`：

- 一个参考 `speculative_decode(target, draft, prompt, K, temperature)`，实现 exact rejection rule，并验证它保留 target distribution（相对 plain target sampling 的 empirical KL < 0.01）。
- 一个 EAGLE-style tree drafter，使用 top-p branching 构建 depth-K tree。
- 一个 tree attention mask builder，为 verifier 生成正确 causal pattern。
- 一个 acceptance-rate harness，在 tiny LM 上运行二者（从 GPT-2-medium target distill 一个 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Use It / 使用它

- **vLLM** 和 **SGLang** 提供 first-class speculative decoding。Flags：`--speculative_model`、`--num_speculative_tokens`。EAGLE-2/3 可通过 `--spec_decoding_algorithm eagle` flag 支持。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE trees。
- **Reference draft models**：`Qwen/Qwen3-0.6B-spec`（drafts for Qwen3-32B）、`meta-llama/Llama-3.2-1B-Instruct-spec`（drafts for 70B）。
- **Medusa heads**（Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）：不使用 draft model，而是在 target 自身上添加 K 个 parallel prediction heads。部署更简单，acceptance 略低于 EAGLE。

## Ship It / 交付

本课会产出 `outputs/skill-speculative-tuning.md`，这是一个 skill，用于 profile target model workload，并选择 draft model、K（draft length）、tree width、temperature，以及何时 fallback 到 plain decode。

## Exercises / 练习

1. 实现 exact rejection rule 并经验验证。分别通过 `speculative_decode` 和 plain target sampling 运行 10K samples；计算两个 output distributions 的 TV distance。应 < 0.01。

2. 计算 speedup formula。给定固定 `α` 和 `K`，画出每次 target-forward 的 expected tokens。为 α ∈ {0.5, 0.7, 0.9} 找出最优 K。

3. 训练一个 tiny draft。取 124M GPT-2 target，并在 100M tokens 上用 KL loss distill 一个 30M GPT-2 draft。测量 held-out text 上的 `α`。预期：0.6-0.7。

4. 实现 EAGLE-style tree drafting。不要输出 chain，而是让 draft 在每个 depth 输出 top-3 branches。构建 tree attention mask。验证 target 会接受最长正确 branch。

5. 测量 failure modes。在 temperature=1.5（高 stochasticity）下运行 speculative decode。展示 α 崩溃，且由于 draft overhead，算法慢于 plain decode。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Target model | “The big model” | 你想从中采样的慢而高质量模型（p distribution） |
| Draft model | “The speculator” | 小而快的 predictor（q distribution）；通常小 5-30x |
| K / draft length | “Look-ahead” | 每次 verify pass 中 speculated tokens 的数量 |
| α / acceptance rate | “Hit rate” | draft proposal 被接受的 per-token probability |
| Exact rejection rule | “The accept test” | 保持 target distribution 的 r < p/q 比较 |
| Residual distribution | “Corrected p-q” | (p - q)+ / ||(p - q)+||_1，即 rejection 时采样的分布 |
| Tree drafting | “Branching speculation” | draft 输出 candidate tree，并用 tree-structured attention mask 在一次 pass 中 verify |
| Tree attention mask | “Topological mask” | 编码 tree topology 的 causal mask，使每个 node 只 attend 到 ancestors |
| Medusa heads | “Parallel heads” | target 自身上的 K 个额外 prediction heads；不需要独立 draft model |
| EAGLE feature reuse | “Hidden-state draft” | draft input 是 target 的 last hidden state，而不是 raw tokens，从而缩小 draft |
| Test-time simulation loss | “EAGLE-3 training” | 用匹配 target test-time distribution 的 outputs 训练 draft，而不是 teacher forcing |

## Further Reading / 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192)：exact rejection rule 和理论 speedup analysis
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318)：DeepMind 同期 speculative-sampling paper
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774)：parallel-heads alternative to a draft model
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077)：feature reuse 与 tree drafting
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858)：dynamic tree topology
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840)：train-time 与 test-time matching
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057)：Jacobi/lookahead decoding，一种不需要 speculator 的替代方案
