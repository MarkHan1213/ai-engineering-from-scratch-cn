# Native Sparse Attention (DeepSeek NSA) / 原生稀疏 Attention（DeepSeek NSA）

> 到 64k tokens 时，attention 会吃掉 decode latency 的 70-80%。每个 open-model lab 都有修复它的方案。DeepSeek 的 NSA（ACL 2025 best paper）是最终站住脚的方案：三个并行 attention branches，分别是 compressed coarse-grained tokens、selectively retained fine-grained tokens，以及用于 local context 的 sliding windows，再通过 learned gate 合并。它 hardware-aligned（kernel-friendly）、natively trainable（用于 pre-training，而不是 inference 时硬接上），并且在 64k decodes 上比 FlashAttention 更快，同时匹配或超过 full attention 质量。本课会端到端构建这三个 branches，并说明为什么这种 sparsity 是端到端可微的。

**类型：** Build
**语言：** Python (stdlib)
**前置要求：** Phase 7 · 12（KV cache, flash-attention），Phase 7 · 15（attention variants），Phase 10 · 16（differential attention）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出 NSA attention 的三个 branches，以及每个 branch 捕获什么信息
- 解释为什么 NSA 是 "natively trainable"，而此前许多 sparse-attention 方法只是 inference-only
- 根据 compression block size 和 selection top-k，计算 NSA 相对 full attention 在 64k context 下的 attention compute savings
- 在短 synthetic sequence 上用 stdlib Python 实现三分支组合，并验证 gating weights 行为合理

## The Problem / 问题

序列长度为 N 时，full attention 的时间复杂度是 `O(N^2)`，每层 KV cache 是 `O(N)`。到 64k tokens 时，计算和内存带宽数字会变得灾难性。NSA 论文中的理论估计显示：在 64k 下，attention 占总 decode latency 的 70-80%。下游所有指标，TTFT、tokens/sec、每百万 tokens 成本，都会被 attention cost 主导。

Sparse attention 是显然答案。此前尝试大致分为两类。Fixed-pattern sparsity（sliding-window、strided、block-local）会丢信息，并在 long-range recall tasks 上失败。Inference-time sparsity（KV cache pruning、H2O、StreamingLLM）被应用到在 dense attention 上预训练的模型，因此只能拿回部分潜在加速，因为模型从未被要求通过 sparse pattern 路由信息。

Native Sparse Attention（Yuan et al., DeepSeek + PKU + UW, ACL 2025 best paper, arXiv:2502.11089）同时解决两件事：一种模型在 pre-training 中学习到的 sparsity pattern，以及一个 kernel-aligned algorithm，能够在 inference 时真实交付 compute savings。两年后，NSA 或它的直接后代会成为每个 frontier long-context model 的默认 attention。

## The Concept / 概念

### Three parallel branches / 三个并行分支

对每个 query，NSA 会对 KV cache 的三种不同视图分别运行 attention：

1. **Compressed branch。** Tokens 被分组成大小为 `l` 的 blocks（通常 32 或 64）。每个 block 通过一个小型 learned MLP 压缩成一个 summary token。query attend 到这些 compressed tokens，从而获得全序列的 coarse-grained view。

2. **Selected branch。** 使用 compressed branch 的 attention scores，识别当前 query 最相关的 top-k blocks。读取这些 blocks 中 fine-grained（未压缩）tokens，并让 query attend 到它们。可以把 compressed-branch attention 看成 selection 的 routing signal。

3. **Sliding-window branch。** query attend 到最近 `W` 个 tokens（通常 512），用于 local context。这个 branch 捕获短程但结构密集的模式（syntax、local coreference），避免另外两个 branch 漏掉。

三个 branch outputs 通过 learned per-position gate 组合：

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 是 query 上一个小 MLP 产生的 gate weights。它们不必和为 1，可以独立给 branch 加权。

### Why this is "natively trainable" / 为什么这是 natively trainable

selection step（top-k blocks）是离散操作。离散操作会打断梯度流。以前的 sparse-attention 工作要么跳过 selection 的反向传播（限制训练），要么使用 continuous relaxations，但 inference 时无法产生真实 sparsity。

NSA 绕开了这个问题：compressed-branch attention 本身就是全序列上的可微 coarse-grained attention。top-k 操作只是复用 compressed branch 中最高的 attention scores，决定加载哪些 fine-grained blocks。梯度会流经 compressed-branch scores（它们同时影响 compressed output 和 selection logic），selected blocks 对最终输出的贡献也是可微的。不可微的 `top_k` 操作在 forward computational graph 上只是 no-op，它只控制哪些 blocks 从内存加载。

这就是为什么 NSA 可以端到端用于 pre-training。模型会联合学习通过三个 branches 路由信息，产出一种 sparse pattern，并且这种 pattern 在 inference 时真的带来承诺的加速。

### Hardware-aligned kernel / 硬件对齐的 kernel

NSA 的 kernel 面向现代 GPU 内存层级设计。kernel 按 GQA groups 加载 queries（外层循环），获取每个 group 对应的 sparse KV blocks（内层循环），并在 SRAM 上运行 attention。由于每个 query group 看到同一组选中 blocks（selection 是 per-query-group，而不是 per-query-head），KV loads 能在 group 内摊销。Arithmetic intensity 保持较高。

论文报告 Triton kernels 在 64k decodes 上比 FlashAttention 快 9x，且 speedup ratio 会随 sequence length 增长。forward 和 backward kernels 都有提供。

### The compute budget / 计算预算

令 `N` 为 sequence length，`l` 为 compression block size，`k` 为 top-k selection count，`w` 为 sliding window，`b` 为 selected block size（通常等于 `l`）。

- Compressed branch：每个 query 有 `O(N/l)` 个 keys，因此总计 `O(N * N / l)`。
- Selected branch：每个 query 有 `O(k * b)` 个 keys，因此总计 `O(N * k * b)`。
- Sliding branch：每个 query 有 `O(w)` 个 keys，因此总计 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

当 `N = 64k, l = 64, k = 16, b = 64, w = 512`：每个 query 的成本是 `1000 + 1024 + 512 = 2536 keys`。Full attention 是 `64000 keys`。计算量降低 25x。

当 `N = 128k, l = 64, k = 16, b = 64, w = 512`：每个 query 的成本是 `2000 + 1024 + 512 = 3536 keys`。Full attention 是 `128000 keys`。降低 36x。收益会随 sequence length 增长，这正是设计目的。

### How does it compare / 与其他方法相比

| Method | Differentiable | Real inference speedup | Long-range recall |
|--------|---------------|----------------------|-------------------|
| Sliding window only | yes | yes | fails |
| Strided / block-sparse | yes | yes | partial |
| KV pruning (H2O, StreamingLLM) | N/A (inference-time) | yes | partial |
| MoBA (Moonshot) | partial | yes | good |
| NSA | yes (natively) | yes (9x at 64k) | matches full attention |

MoBA（Moonshot, arXiv:2502.13189）几乎同期发表，采取了类似“三个总比一个好”的思路，把 MoE 原则应用到 attention blocks 上。NSA 和 MoBA 是 2026 long-context pre-training 必须了解的两个架构。

```figure
sliding-window-attention
```

## Build It / 动手构建

`code/main.py` 在一段短 synthetic sequence 上实现三个 branches，并展示：

- Compression MLP（为教学清晰使用 simple mean-pool baseline；真实 NSA 使用 learned MLP）。
- 由 compressed-branch scores 驱动的 top-k block selection。
- 最近 `w` tokens 上的 sliding-window attention。
- Gated combination。
- 与 full attention 比较的 compute-count printout。

### Step 1: compress tokens into blocks / 步骤 1：把 tokens 压缩成 blocks

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Step 2: compressed-branch attention / 步骤 2：Compressed-branch attention

让 query 对 compressed keys 运行 softmax attention。compressed-branch scores 同时也是 top-k selection 的信号。

### Step 3: top-k block selection / 步骤 3：Top-k block selection

选择 compressed blocks 中分数最高的 `k` 个 indices。加载这些 blocks 中原始未压缩 tokens，并在它们上运行 attention。

### Step 4: sliding-window attention / 步骤 4：Sliding-window attention

取最近 `w` 个 tokens，并在它们上运行 standard attention。

### Step 5: gate + combine / 步骤 5：Gate 加组合

query 上的小 MLP 产生三个 gate weights。最终输出是三个 branch outputs 的 weighted sum。

### Step 6: compute counting / 步骤 6：计算量计数

打印每个 branch 每个 query attend 的 keys 数量和总数。与 `N`（full attention）对比。在 1024-token synthetic 上，若 `l = 32, k = 4, w = 128`，NSA 每个 query 看到 `32 + 128 + 128 = 288` keys，而 full attention 是 1024，减少约 3.5x。

## Use It / 使用它

NSA 已经在 DeepSeek 自己的 long-context pre-training pipeline 中使用。截至 2026 年 4 月，public inference stacks 的集成状态如下：

- **DeepSeek internal**：native，公开 weights 使用 NSA 或其后继 DSA（Deepseek Sparse Attention）。
- **vLLM**：针对 DeepSeek-V3.x weights 的实验性 NSA support 正在开发。
- **SGLang**：已发布 NSA benchmarks；production path 跟随 vLLM。
- **llama.cpp / CPU**：不支持；在 CPU throughput 下 kernel decomposition 的 overhead 不值得。

什么时候使用 NSA：

- 以 64k+ context 为目标、预算严肃的 pre-training 或 continued-training run。
- DeepSeek 自家 long-context checkpoints 的 inference。它们的 weights 是 NSA-native。

什么时候不要用：

- serving 一个已有 dense-attention pre-trained model。没有 continued training，无法 retrofitting NSA。
- context 小于 16k。三分支 overhead 会压过节省。
- Batch-1 interactive chat。latency-sensitive decode 会受益，但只在 long contexts 下明显。

## Ship It / 交付

本课会产出 `outputs/skill-nsa-integrator.md`。给定 long-context pre-training run specification，它会产出 NSA integration plan：compression block size、top-k、sliding window、gate MLP width、kernel choice，以及能证明架构变化值得的 specific long-context evals。

## Exercises / 练习

1. 在 1024-token synthetic 上运行 `code/main.py`。对三组 presets 扫 `(l, k, w)` 并打印 compute counts。找出在 needle-in-haystack test 上保持 95% recall 的同时，让每 query key-count 最低的 preset。

2. 用 tiny learned MLP（2-layer, hidden 32）替换 mean-pool compressor。在 signal 是某个 block 平均值的 synthetic task 上训练它。测量 held-out data 上与 mean-pool baseline 相比的 perplexity gap。

3. 实现 gate MLP。它接收 query 作为输入，并输出三个 scalars。展示 gate 行为合理：random queries 上接近 uniform weighting；当 query 命中 far-back block 时，对 selected branch 给出重权重。

4. 计算 NSA-enabled 70B model 在 128k context 下的 KV cache memory budget。KV heads 为 8，head dim 为 128，BF16。与 full attention 和 MLA 对比（Phase 10 · 14 展示了 MLA 的数字）。找出 NSA fine-grained branch KV cache 等于 full attention 的 sequence length。

5. 阅读 NSA paper（arXiv:2502.11089）的 Section 4，并用三句话解释为什么 compressed branch 的 attention scores 会被复用于 top-k selection，而不是单独计算 routing score。把答案与 gradient flow 联系起来。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Compressed branch | “Coarse view” | 对 block-averaged keys 做 attention，以每 query O(N/l) keys 提供 global context |
| Selected branch | “Top-k blocks” | 对 compressed-branch scores 最高的 `k` 个 blocks 做 fine-grained attention |
| Sliding window | “Local context” | 对最近 `W` 个 tokens 做 attention，用于 short-range patterns |
| Native trainability | “Pre-train with the sparsity on” | sparsity pattern 在 pre-training 中学习，而不是 inference 时硬接上 |
| Compression block size l | “Group size for coarse view” | 合并成一个 summary 的 token 数量；典型值 32-64 |
| Top-k | “Blocks to keep” | 会读取其 uncompressed tokens 的 compressed blocks 数量；典型值 16 |
| Sliding window W | “Local attention radius” | 通常为 512；太短伤害 local coherence，太长浪费 compute |
| Branch gate | “How to mix the three” | per-position MLP 输出，用于加权三个 branches 的贡献 |
| Hardware alignment | “Kernel-friendly sparsity” | 选择 sparse pattern 时确保实际 GPU kernel 能获得理论加速 |
| DSA | “NSA's successor” | Deepseek Sparse Attention，DeepSeek 谱系中 NSA 之后的架构 |

## Further Reading / 延伸阅读

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089)：原论文
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)：NSA 面向的架构家族
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189)：同期工作，用 MoE-style attention over blocks
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150)：sliding-window 起点
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453)：NSA 改进的 inference-time sparsity baseline
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691)：NSA kernels 在 64k 上超过的 full-attention baseline
