# Mixture of Experts (MoE) / 专家混合（MoE）

> Dense 70B transformer 会为每个 token 激活所有参数。671B MoE 每个 token 只激活 37B，却在每个 benchmark 上击败它。Sparsity 是这个十年最重要的 scaling idea。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 理解 MoE 如何解耦 total parameters 与 active parameters per token
- 实现 top-k router、gating weights 和 auxiliary-loss-free balancing 的核心逻辑
- 比较 shared experts、routed experts 与 fine-grained experts 的作用
- 判断 MoE 在 inference cost、VRAM、expert parallelism 和 latency 上的取舍

## The Problem / 问题

Dense transformer 的 inference FLOPs 大致等于参数量（forward pass 还要乘 2）。Dense model 一旦变大，每个 token 都要付全额账单。到 2024 年，frontier 已经撞上 compute wall：要明显更聪明，就需要每 token 指数级更多 FLOPs。

Mixture of Experts 打破了这个绑定。把每个 FFN 替换为 `E` 个 independent experts + 一个 router，router 为每个 token 选择 `k` 个 experts。Total parameters = `E × FFN_size`。Active parameters per token = `k × FFN_size`。2026 年典型配置是 `E=256`、`k=8`。Storage 随 `E` 增长，compute 随 `k` 增长。

2026 年 frontier 几乎全是 MoE：DeepSeek-V3（671B total / 37B active）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立 leaderboard 上，top 10 open-source models 全部是 MoE。

## The Concept / 概念

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### The FFN swap / 替换 FFN

Dense transformer block：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个 expert 都是一个 independent FFN（通常是 SwiGLU）。Router 是单个 linear layer。每个 token 选择自己的 `k` 个 experts，并得到它们输出的 gated mixture。

### The load-balancing problem / Load-balancing 问题

如果 router 把 90% tokens 都送进 expert 3，其他 experts 就会饿死。历史上尝试过三种修复：

1. **Auxiliary load-balancing loss**（Switch Transformer、Mixtral）。添加一个与 expert usage variance 成比例的 penalty。有效，但会引入 hyperparameter 和第二个 gradient signal。
2. **Expert capacity + token dropping**（早期 Switch）。每个 expert 最多处理 `C × N/E` tokens；overflow tokens 跳过该 layer。会伤害质量。
3. **Auxiliary-loss-free balancing**（DeepSeek-V3）。添加一个 learned per-expert bias，移动 router 的 top-k selection。Bias 在 training loss 之外更新。不对 main objective 加 penalty。2024 年的大突破。

DeepSeek-V3 的做法：每个 training step 后，对每个 expert 检查 usage 高于还是低于 target。用 `±γ` nudging bias。Selection 使用 `scores + bias`。用于 gating 的 expert probabilities 仍然使用未改动的 raw `scores`。这把 routing 与 expression 解耦了。

### Shared experts / 共享 experts

DeepSeek-V2/V3 还把 experts 分成 *shared* 和 *routed*。每个 token 都通过所有 shared experts。Routed experts 通过 top-k 选择。Shared experts 捕捉 common knowledge；routed experts 做 specialization。V3 运行 1 个 shared expert 加 256 个 routed experts 中的 top-8。

### Fine-grained experts / 细粒度 experts

Classic MoE（GShard、Switch）：每个 expert 和完整 FFN 一样宽。`E` 较小（8–64），`k` 较小（1–2）。

Modern fine-grained MoE（DeepSeek-V3、Qwen-MoE）：每个 expert 更窄（1/8 FFN size）。`E` 很大（256+），`k` 也更大（8+）。Total parameters 相同，但组合数增长快得多。`C(256, 8) = 400 trillion` 种可能的每-token “experts” 组合。Quality 上升，latency 保持平坦。

### The cost profile / 成本画像

Per token、per layer：

| Config | Active params / token | Total params |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 在几乎每个 benchmark 上都击败 Llama 3 70B（dense），同时每 token 使用**更少 active FLOPs**。更多 parameters = 更多 knowledge。更多 active FLOPs = 每 token 更多 compute。MoE 把二者解耦。

### The catch: memory / 代价：memory

无论哪些 experts 被激活，所有 experts 都必须驻留在 GPU 上。671B model 用 fp16 weights 需要约 1.3 TB VRAM。Frontier MoE deployment 需要 expert parallelism：把 experts shard 到多张 GPU，跨网络 route tokens。Latency 主要由 all-to-all communication 主导，而不是 matmul。

## Build It / 动手构建

见 `code/main.py`。一个 pure stdlib 的 compact MoE layer，包含：

- `n_experts=8` 个 SwiGLU-ish experts（为演示，每个只用一个 linear）
- top-k=2 routing
- softmax-normalized gating weights
- 通过 per-expert bias 实现 auxiliary-loss-free balancing

### Step 1: the router / 第 1 步：router

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias 影响 selection，不影响 gate weight。这就是 DeepSeek-V3 trick：bias 修正 load imbalance，但不 steering 模型预测。

### Step 2: run 100 tokens through the router / 第 2 步：让 100 个 tokens 通过 router

追踪哪些 experts 被激活以及频率。不加 bias 时，usage 会偏斜。加入 bias update loop（over-used experts `-γ`，under-used experts `+γ`）后，几轮内 usage 会收敛到近似 uniform distribution。

### Step 3: param count comparison / 第 3 步：参数量对比

打印一个 MoE config 的 “dense equivalent”。DeepSeek-V3-shaped：256 routed + 1 shared，8 active，d_model=7168。Total parameter count 非常夸张。Active count 只有 dense Llama 3 70B 的七分之一。

## Use It / 应用它

HuggingFace loading：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年 production inference：vLLM 原生支持 MoE routing。SGLang 有最快的 expert-parallel path。两者都会自动处理 top-k selection 和 expert parallelism。

**When to pick MoE / 什么时候选 MoE：**
- 你想要 frontier quality，但希望每 token inference cost 更低。
- 你有足够 VRAM / expert-parallel infrastructure。
- 你的 workload 是 token-heavy（chat、code），不是 context-heavy（long docs）。

**When NOT to pick MoE / 什么时候不选 MoE：**
- Edge deployment：任何 active FLOP 都要付完整 storage。
- Latency-critical single-user serving：expert routing 会增加 overhead。
- Small models（<7B）：MoE 的 quality advantage 只在超过某个 compute threshold（约 6B active params）后显现。

## Ship It / 交付它

见 `outputs/skill-moe-configurator.md`。这个 skill 会根据 parameter budget、training tokens 和 deployment target，为新 MoE 选择 E、k 和 shared-expert layout。

## Exercises / 练习

1. **Easy / 简单。** 运行 `code/main.py`。观察 auxiliary-loss-free bias update 如何在 50 iterations 内拉平 expert usage。
2. **Medium / 中等。** 用 hash-based router（deterministic、no learning）替换 learned router。比较 quality 和 balance。为什么 learned router 更好？
3. **Hard / 困难。** 实现 GRPO-style “rollout-matched routing”（DeepSeek-V3.2 trick）：记录 inference 时哪些 experts 被激活，在 gradient computation 时强制相同 routing。测量它对 toy policy-gradient setup 的影响。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Expert | “One FFN among many” | 一个 independent feed-forward network；参数只负责 FFN computation 的一个 sparse slice。 |
| Router | “The gate” | 一个 tiny linear layer，给每个 token 对每个 expert 打分；再做 top-k selection。 |
| Top-k routing | “k active experts per token” | 每个 token 的 FFN computation 经过恰好 k 个 experts，并由 gate 加权。 |
| Auxiliary loss | “Load-balance penalty” | 额外 loss term，用来惩罚 skewed expert usage。 |
| Auxiliary-loss-free | “DeepSeek-V3's trick” | 只对 router selection 施加 per-expert bias 来 balance；没有额外 gradient。 |
| Shared expert | “Always on” | 每个 token 都会通过的额外 expert；捕捉 common knowledge。 |
| Expert parallelism | “Shard by expert” | 把不同 experts 分发到不同 GPUs；tokens 跨网络 route。 |
| Sparsity | “Active params < total params” | 比例 `k × expert_size / (E × expert_size)`；DeepSeek-V3 约 37/671 ≈ 5.5%。 |

## Further Reading / 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 这个想法的源头。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，经典 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + auxiliary-loss-free MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — bias-based balancing 论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本课 router 使用的 fine-grained + shared-expert split。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 原始 shared-expert 论文。
