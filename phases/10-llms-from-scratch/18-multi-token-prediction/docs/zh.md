# Multi-Token Prediction (MTP) / 多 Token 预测

> 从 GPT-2 到 Llama 3，每个 autoregressive LLM 都按每个位置一个 loss 训练：预测下一个 token。DeepSeek-V3 在每个位置增加了第二个 loss：预测再后面的 token。在 671B 模型上，额外 14B 参数通过 gradient flow 蒸馏回主模型；训练好的 MTP heads 在 inference 时又被复用为 speculative-decoding drafters，acceptance 超过 80%。1.8× generation throughput 几乎白送。本课会构建 DeepSeek technical report 中的 sequential MTP module，计算 loss 与 shared-head 参数布局，并解释为什么 MTP 保留 causal chain，而 Gloeckle et al. 的原始 parallel MTP 破坏了它。

**类型：** Build
**语言：** Python (stdlib)
**前置要求：** Phase 10 · 04（pre-training a mini GPT），Phase 10 · 15（speculative decoding）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 陈述 MTP training objective，并推导跨 prediction depths 的 joint loss
- 解释 Gloeckle et al. 的 parallel MTP heads（2024）与 DeepSeek-V3 的 sequential MTP modules 的差异，以及为什么 sequential design 能保留 causal chain
- 计算为 pre-training run 添加 MTP modules 的参数与内存开销
- 从零实现一个 MTP module：shared embedding、per-depth transformer block、projection 和 shared output head

## The Problem / 问题

Next-token prediction 是标准 LLM training objective。每个 hidden state 只被监督预测一件事：紧随其后的 token。这其实是一个出人意料地弱的信号。序列中的多数信息都会跨越一个 token 以上，比如结构、连贯性、事实性、算术流。模型必须在 trillions of tokens 上累积许多 one-token signals 才能学到这些东西。

MTP 问的是：如果每个 hidden state 同时被监督预测多个 future tokens，会怎样？Gloeckle et al.（Meta, 2024）证明这样有帮助。他们的实现是在 backbone 之上放几个独立 output heads，每个 head 预测不同 offset。它并行、简单，但 heads 看到的是同一个 hidden state，没有 hierarchical refinement；预测之间也不形成 causal chain，因此不能用于 speculative decoding。

DeepSeek-V3（2024 年 12 月）把 MTP 重新设计为 sequential modules，在每个 prediction depth 保留 causal chain。模型先从 `h_i^(0)` 预测 `t+1`，然后从新的 hidden state `h_i^(1)` 预测 `t+2`；这个新状态结合了 `h_i^(0)` 与 `E(t+1)` embedding，依此类推。每个 depth 都有自己的小 transformer block。shared embedding 和 shared output head 让参数开销保持温和。在 DeepSeek-V3 的尺度上，MTP modules 在 671B main-model weights 之上增加了 14B 参数。2% overhead 换来了更密集的训练信号，以及 inference 时现成的 speculative-decoding draft。

本课从零构建一个 MTP module 和 D-depth loss。数学很干净，实现大约 150 行。

## The Concept / 概念

### The sequential MTP recipe / Sequential MTP 配方

DeepSeek-V3 在主模型之上添加 `D` 个 MTP modules。每个 module `k`（`k = 1..D`）预测 depth `k` 的 token，也就是给定到位置 `i` 为止的 prefix，预测 `t_{i+k}`。

Module `k` 包括：

- 一个 transformer block `T_k`，带自己的 attention 和 MLP。
- 一个 projection matrix `M_k`，把 previous-depth hidden state 与 next-depth ground-truth token 的 embedding 合并。
- shared embedding `E`（与主模型相同）。
- shared output head `Out`（与主模型相同）。

训练时，对经过位置 `i` 的 prefix，per-depth hidden state 为：

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

per-depth prediction 为：

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

per-depth loss 是对 ground-truth `t_{i+k}` 的 cross-entropy：

```
L_k = CE(logits_{i+k}, t_{i+k})
```

跨 depths 的 joint loss：

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` 是一个小 weighting factor。DeepSeek-V3 在训练前 10% 使用 0.3，之后使用 0.1。总训练 loss 是 `L_main + L_MTP`。

### Why sequential, not parallel / 为什么是 sequential，而不是 parallel

Gloeckle 的原始 parallel MTP 有 D 个 output heads，每个 head 都直接作用于 `h_i^(0)`。每个 head 从同一个 backbone hidden state 预测 `t_{i+k}`。这样训练没问题，但这些预测没有彼此条件化。你不能用 `head_1` 的输出帮助 `head_2`，因为这些 heads 是并行触发的。

DeepSeek-V3 的 sequential design 从 `h_i^(k-1)` 加实际 next-token embedding `E(t_{i+k})` 构建 `h_i^(k)`。这保留了 causal chain：要预测 `t_{i+k+1}`，depth `k+1` 的 module 会看到 `t_{i+k}` 的内容。结构上，这与 autoregressive decoder 消费自己输出的方式相同，因此 MTP modules 可以直接作为 speculative-decoding drafters 使用。

inference 时：把 `h_i^(k-1)` 和 drafted `t_{i+k}` 喂给 module `k+1`，得到 `t_{i+k+1}` 的预测。重复即可。这正是 EAGLE-style draft，只是 draft network 使用训练好的 MTP module。DeepSeek-V3 报告第一个 MTP module 的 acceptance 超过 80%，带来约 1.8× speedup。

### Parameter accounting / 参数核算

对 hidden 为 `h`、vocabulary 为 `V` 的模型：

- Main model：数十亿参数，加一个大小为 `V * h` 的 output head。
- Shared output head：复用主模型 head。无额外参数。
- Shared embedding：复用主模型 embedding。无额外参数。
- 每个 MTP module：
  - Projection `M_k`：`(2h) * h = 2h^2`。
  - Transformer block `T_k`：attention（MHA 约 `4h^2`）加 MLP（SwiGLU 且 ratio 8/3 时通常约 `8h^2`）。每个 block 约 `12h^2`。

每个 module 的额外参数总计：`~14h^2`。对 DeepSeek-V3 的 `h = 7168`、D = 1 module：纸面上 `~14 * 7168^2 = ~720M` 参数。DeepSeek-V3 报告 14B，差异主要来自 MTP module 中的 expert layers 也采用了 MoE。

### The speculative-decoding payoff / Speculative decoding 收益

pre-training 期间，MTP modules 会让训练慢约 10%（更多 forward compute、额外 loss）。回报有两层：

1. 更密集的训练信号。每个 hidden state 看到 D+1 个 supervision targets。DeepSeek-V3 ablations 在 MMLU、GSM8K、MATH、HumanEval 上测到稳定的几个百分点提升。

2. inference 时免费获得 speculative decoding draft。MTP module 已经被训练来预测接下来几个 token。复用为 draft network 时，它能提供 80%+ acceptance rates。在这个水平上，N=3 或 N=5 的 spec decoding 会带来 1.8× throughput。10% 的 training-time cost 在你第一次大规模 inference 时就能回本。

### Relation to EAGLE / 与 EAGLE 的关系

EAGLE 在 pre-training 之后单独训练一个小 draft model。MTP 把 draft 烧进 pre-training。两者通过不同 pipeline 收敛到相近 accept rates：

| Dimension | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| When trained | Post-pre-training | During pre-training |
| Backward-compatible with existing weights | Yes | No (need to re-train) |
| Draft params | 1-2 transformer layers | 1 transformer block + projection |
| Acceptance rate | 0.88-0.92 | 0.80+ at depth 1 |
| Benefit beyond speedup | Speculative decoding only | Denser training signal + speedup |

## Build It / 动手构建

`code/main.py` 端到端构建一个 MTP module：shared embedding、projection、transformer block、shared output head。然后它在一段短 synthetic sequence 上计算 per-depth cross-entropy loss，并打印按组件拆分的 parameter count。toy vocabulary 只有 32 个 tokens，让数字更容易看。

### Step 1: shared embedding table / 步骤 1：Shared embedding table

一个 `vocab_size x hidden` table 同时被 main model 和每个 depth 的每个 MTP module 使用。不是第二份拷贝，而是同一个 tensor。

### Step 2: the per-depth combination / 步骤 2：Per-depth 组合

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

真实 DeepSeek-V3 会把两个 RMSNormed vectors concat 成 `[2h]`，再用 `h x 2h` matrix 投影。toy 为了 stdlib 简洁，用 vector addition 代替。

### Step 3: the transformer block at depth k / 步骤 3：Depth k 的 transformer block

Self-attention 加 MLP。在 toy 中，一个 one-layer linear attention block 和一个 SwiGLU MLP 保留结构可见性，同时不依赖 numpy。

### Step 4: the shared output head / 步骤 4：Shared output head

复用主模型的 output projection。输出 vocabulary 上的 logits。

### Step 5: per-depth loss / 步骤 5：Per-depth loss

对 offset `k` 处的 ground-truth token，计算 softmax(logits) 的 cross-entropy。用 `lambda / D` scaling factor 聚合 across depths。

### Step 6: parameter accounting / 步骤 6：参数核算

打印 total parameter count、shared（embedding, head）count，以及 per-module extra count。展示 MTP extra 相对 main-model size 的比例。

## Use It / 使用它

MTP 已集成到 DeepSeek-V3（2024 年 12 月）和 DeepSeek-R1 系列。inference 时：

- DeepSeek 自家 serving stack 会开箱即用地把 MTP modules 当作 speculative decoders。
- 截至 2026 年 4 月，vLLM 和 SGLang 都有 DeepSeek-V3 MTP integration paths。
- AMD 的 ROCm SGLang tutorial 展示了具体 MTP speculative-decoding config，并在 V3 checkpoint 上测得 1.8× speedup。

什么时候在新的 pre-training run 中使用 MTP：

- 你控制完整 pre-training pipeline，并希望提前押注更密集训练信号。
- 你知道模型会被大规模 serving，并希望免费获得 speculative decoding。
- hidden size 至少为 4096。1B 规模上 overhead 往往比收益更痛。

什么时候不要用：

- fine-tuning 一个已有 pre-trained dense model。MTP module 并没有被训练过。
- 你想保留 clean baseline 做研究对照。MTP 会改变 architecture。

## Ship It / 交付

本课会产出 `outputs/skill-mtp-planner.md`。给定 pre-training run specification（model size、data、compute），它会返回 MTP integration plan：depths D 数量、`lambda` schedule、memory overhead，以及 inference-time speculative-decoding wiring。

## Exercises / 练习

1. 运行 `code/main.py`。展示当 synthetic signal 增强时，per-depth loss 单调下降。修改 synthetic 使其使用固定 pattern，并验证 depth-1 与 depth-2 losses 都会收敛。

2. 计算一个 dense 70B model（hidden 8192, 80 layers）使用 D=1 MTP module 的参数 overhead。与 DeepSeek-V3 报告的 14B overhead 对比。解释为什么 DeepSeek 的数字更高：MTP transformer block 继承了同样的 MoE structure，抬高了 per-module parameter count。

3. 在 toy 中实现 D=2：添加第二个 MTP module，接收 h^(1) 并预测 `t_{i+2}`。验证 joint loss 和 parameter accounting 与 DeepSeek paper 的 equations 19-21 匹配。

4. 把 toy 切换为 parallel MTP（Gloeckle-style）：在 main hidden state 之上添加 D 个 output heads，每个预测不同 offset。用同一个 synthetic signal 对比它与 sequential version 的 per-depth losses。对 k > 1，sequential version 应该产生更低 depth-k loss，因为它条件化在中间预测上。

5. 把训练好的 MTP module 当作 EAGLE-style draft：inference 时调用 module k 提议 `t_{i+k}`。在 held-out sequence 上测量这些 draft tokens 相对 main model predictions 的 acceptance rate。如果 toy 上达到 50%+，你就复现了 MTP-as-draft 的经验性质。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MTP module | “Extra loss block” | 一个小 transformer block 加 projection，用来预测主模型之后第 `k` 个位置的 token |
| Prediction depth | “Which offset” | 整数 `k`，表示 module `k` 从到位置 `i` 的 prefix 预测 `t_{i+k}` |
| Parallel MTP | “Gloeckle-style” | 同一个 backbone hidden state 上的 D 个独立 heads，没有 conditional chain |
| Sequential MTP | “DeepSeek-V3 style” | 每个 module 条件化在 previous depth hidden state 加 next token embedding 上，保留 causal chain |
| Shared output head | “Reuse the main head” | MTP modules 调用主模型 LM head，而不是单独 output projection |
| Shared embedding | “Reuse the main table” | 同一张 vocabulary embedding table 到处复用，没有重复参数 |
| Projection matrix M_k | “Combine hidden + next-token” | 一个 `h x 2h` linear layer，把 previous hidden state 和 target-token embedding 折叠进 next depth input |
| Joint loss L_MTP | “Averaged extra losses” | per-depth cross-entropy losses 的算术平均，并乘以 `lambda` |
| Acceptance rate at depth 1 | “How often MTP draft is right” | D=1 MTP module 的 top-1 prediction 等于 main model top-1 prediction 的比例；DeepSeek-V3 上超过 80% |
| Lambda weighting | “Extra-loss importance” | per-depth scaling factor；DeepSeek-V3 训练开始使用 0.3，之后使用 0.1 |

## Further Reading / 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)：完整 sequential MTP 描述（Section 2.2），包括 joint-loss equations 和 inference 的 1.8× speedup
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737)：DeepSeek 设计所改进的 parallel MTP baseline
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3)：685B total（671B main + 14B MTP），deployment notes
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192)：MTP 适配的 speculative-decoding framework
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840)：EAGLE 的 2025 draft architecture，是 MTP 的对应竞争方案
