# Async and Hogwild! Inference / 异步与 Hogwild! 推理

> Speculative decoding（Phase 10 · 15）在单条序列内部并行化 tokens。Multi-agent frameworks 在完整序列之间并行化，但要求显式协调（voting、sub-task splitting）。Hogwild! Inference（Rodionov et al., arXiv:2504.06261）做的是另一件事：并行运行 N 个相同 LLM 实例，使用一个共享 key-value cache。每个 worker 会立即看到其他 worker 生成的 tokens。现代 reasoning models，如 QwQ、DeepSeek-R1，可以通过共享 cache 自我协调，无需 fine-tuning。这个方法仍处于实验阶段，但它打开了一个全新的 inference parallelism 轴，与 spec decode 正交。本课会用 stdlib Python 实现一个 two-worker Hogwild! simulator，并解释为什么 shared-cache collaboration 会从模型已有 reasoning abilities 中涌现出来。

**类型：** Build
**语言：** Python (stdlib)
**前置要求：** Phase 10 · 12（inference optimization），Phase 10 · 15（speculative decoding）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述三种常见 parallel-LLM topologies（voting、sub-task、Hogwild!），并说出各自适合的问题
- 陈述 Hogwild! 的核心设置：multiple workers、one shared KV cache、通过 self-prompting 涌现 coordination
- 根据 worker count `N`、task-level parallelism `p` 和 coordination overhead `c` 计算 Hogwild! 的 wall-time speedup
- 在 toy problem 上实现 two-worker Hogwild! simulator，并观察涌现的任务分工

## The Problem / 问题

现代 LLM 解决难题时，常常会生成很长的 reasoning chains：5,000 tokens 的 step-by-step logic 很常见，深度数学问题上数万 tokens 也会发生。70B 模型以 35 tokens/sec decode，50k tokens 需要 24 分钟。它就不再是 interactive model。

Speculative decoding（Phase 10 · 15）通过在单条序列内部并行化，能给你 3-5x speedup。超过这个范围后，autoregressive decoding 的 sequential dependency 就是硬天花板。每个新 token 都依赖所有 prior tokens。

显然的问题是：能不能跨序列并行？在同一个问题上运行同一模型的多个副本，让它们合作，让它们分工？

已有工作包括：voting ensembles（运行 N 个模型，选 majority answer）、tree-of-thought（分叉 reasoning paths 再 recombine）、multi-agent frameworks（给每个 agent 分配 sub-task，再用 coordinator）。它们在特定 task domains 上有帮助，但都引入显式 coordination machinery：voting rules、branch-and-prune logic、agent-to-agent messaging protocols。

Hogwild! Inference 走另一条路。N 个 workers 共享单个 KV cache。每个 worker 会立即看到其他 worker 生成的 tokens，就像这些 tokens 是自己的 context 一样。workers 在没有任何 training 或 fine-tuning 的情况下，自己学会分工。现代 reasoning models（QwQ、DeepSeek-R1、Claude-family reasoning mode）会读共享 cache，并说出类似“我看到 worker 2 已经处理了 base case，所以我来做 inductive step”的内容。

截至 2026 年 4 月，speedup 仍然强依赖 workload 且处于实验阶段。但这个想法值得了解，因为它打开了 inference parallelism 的新轴。

## The Concept / 概念

### The setup / 设置

初始化 N 个 worker processes，全部运行同一个 LLM。不使用 per-worker KV caches，而是维护一个共享 cache。当 worker `i` 生成 token `t_j` 时，该 token 被写到 shared cache 的下一个位置。当 worker `k` 进行下一步时，它读取 cache 当前状态，其中包含所有 N 个 workers 到目前为止生成的内容。

step 时，workers 会竞争写 tokens。没有 per-worker position index，cache 是单个增长序列。顺序由 write arrival time 决定。

### Why coordination emerges / 为什么 coordination 会涌现

workers 共享一个 prompt。通常类似：“你是 N 个一起解决此问题的实例之一。每个实例读取共享内存，并能看到其他实例写了什么。避免重复工作。” prompt 加 shared cache 就足够。reasoning models 会读 cache，注意到问题的哪些部分已经被尝试，然后经常（但并不总是）转向未探索部分。

Hogwild! paper（Rodionov et al., 2025）报告了这些观察：

- Workers 通过 cache 制定计划并向其他 workers 传达。
- Workers 注意到其他 workers reasoning 中的错误，并指出它们。
- Workers 在计划失败时会适应并提出 alternatives。
- 当 prompt 要求检查 redundancy 时，workers 会检测重复并转向。

这些都不需要 fine-tuning。涌现行为来自模型已经具备的 reasoning capabilities。

### The naming / 命名

论文名称借用了 Hogwild! SGD（Recht et al., 2011），一种 asynchronous-update optimizer。类比是：SGD 的 asynchronous workers 都写入共享 parameter vector；Hogwild! Inference 的 workers 都写入共享 KV cache。两者都依赖经验收敛，而不是 synchronization guarantees。

### RoPE makes this tractable / RoPE 让它可行

Rotary Position Embeddings（RoPE, Su et al. 2021）通过 Q 和 K 向量中的旋转编码位置信息。由于 positions 是 rotations，而不是 baked-in offsets，一个 token 的位置可以移动而无需重算 KV cache entry。当 worker `i` 写入 shared cache 的位置 `p` 时，其他 workers 读取该位置可以直接使用 cached entry，无需 re-rotation。

在 learned-position 或 absolute-position 模型中，Hogwild! 每次 concurrent write 都需要 cache invalidation。RoPE 让 cache 保持稳定。

### Wall-time math / Wall-time 数学

令 `T_serial` 为一个 worker 单独解决问题的时间。令 `p` 为 task-level parallelizable fraction。令 `c` 为 per-step coordination overhead（读取扩展后的 cache、决定写什么）。

单 worker 时间：`T_serial`。
N-worker Hogwild! 时间，如果 coordination 免费：`T_serial * ((1 - p) + p / N)`。这是经典 Amdahl。
带 coordination overhead：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

要让 worker 有生产力，`c` 必须相对 per-step decode time 足够小。reasoning models 生成 5k+ tokens 时，workers 可以承担数百 tokens 的 coordination overhead，仍然有收益。短 chat tasks 中，coordination 会主导，Hogwild! 比 serial 更差。

### Concrete example / 具体例子

Reasoning problem：10k tokens 的 chain-of-thought。假设问题有 `p = 0.7` 的 parallelizable content（不同 proof strategies、不同 case analyses），每个 worker 的 `c = 200` tokens coordination overhead。使用 `N = 4` workers：

- Serial time：10000 decode steps。
- Hogwild! time：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 decode steps。
- Speedup：10000 / 5550 = 1.8x。

这只是温和提升。但在更长 reasoning problems（50k tokens）上，coordination overhead 会被摊薄，speedup 会推到 2.5-3x。Hogwild! 是 inference 里的 thread-level parallelism，前提是语言本身能让你自然写出 multi-threaded code。

### When to reach for Hogwild! / 什么时候使用 Hogwild!

- 长 reasoning problems（数千 tokens），并且 task 可以分成独立 sub-goals 并行处理。
- 已经训练出 step-by-step thinking 的 reasoning models。非 reasoning models 不擅长 self-coordinate。
- 单节点部署，有足够 VRAM 容纳 shared cache 加 N 个 worker processes。cache 共享，但每个 worker 有自己的 activation memory。

### When not to / 什么时候不要用

- 短 interactive chat。coordination overhead 主导。
- 不能并行化的任务（单条 linear proof、单次 compilation）。N=1 就是最大值。
- 非 reasoning models。不会涌现 coordination。
- 多节点部署。shared cache 需要非常快的 cross-worker synchronization。intra-node 可行，cross-node 是 latency 灾难。

### The experimental status / 实验状态

截至 2026 年 4 月，Hogwild! 仍是带 open-source PyTorch implementation 的研究方法。还没有生产采用。三个 blockers：

1. 跨 concurrent processes 管理 shared KV cache 是不平凡工程。
2. Emergent coordination 依赖 task；benchmarks 仍在构建。
3. 与 speculative decoding 已经提供的加速相比，speedups 较温和；二者可以组合，但组合工程又多一层。

值得了解，值得实验，但还不值得押上产品。

```figure
continuous-batching
```

## Build It / 动手构建

`code/main.py` 实现一个 toy Hogwild! simulator：

- 两个 worker processes，每个都是 deterministic "LLM"，会以已知概率产生几类 tokens（work-token、observe-token、coordinate-token）。
- 一个 shared cache（只是 token list），两个 workers 都会读写。
- 简单 coordination logic：当 worker 看到另一个已经在某个 category 中产生了足够 work tokens，就选择另一个 category。

simulator 会按固定 step budget 运行，并报告：

- 产生的 total work-tokens。
- total wall time（worker steps 数）。
- 相对 single worker 的 effective speedup。
- 每个 worker 写入哪个 token 的 trace。

### Step 1: the shared cache / 步骤 1：Shared cache

两个 workers 都 append 的 list。真实实现会使用简单锁（Python `threading.Lock`）；这里用 counter 模拟。

### Step 2: the worker loop / 步骤 2：Worker loop

每个 worker 每一步：

- 读取当前 shared cache。
- 根据已有内容决定要写哪个 category 的 token。
- 写入一个 token。

### Step 3: the coordination heuristic / 步骤 3：Coordination heuristic

如果 category X 在 cache 中已经有 K 个 tokens，而 worker 原本打算写 X，worker 就切换到 category Y。这是 toy 版替身，用来模拟 reasoning model 的“注意到这部分已经覆盖，于是做别的事”。

### Step 4: measured speedup / 步骤 4：测量 speedup

用 N=1 worker 和 N=2 workers 运行 simulator，保持相同 total step budget。统计 work-tokens。N=2 应该由于 coordination-driven task division 产生大约 1.5-1.8x 更多 work-tokens。

### Step 5: stress the coordination / 步骤 5：压测 coordination

降低 coordination heuristic 的敏感度。再次运行。观察如果 coordination 不好，N=2 会重复生成同样 tokens，speedup 低于 1。这与论文观察一致：这个技巧只有在 workers 具备 self-coordinate 的 reasoning capacity 时才有效。

## Use It / 使用它

截至 2026 年 4 月，Hogwild! 的生产集成仍是 research-grade。Yandex/HSE/IST 的 reference implementation 基于 PyTorch，面向 DeepSeek-R1 和 QwQ models 的单节点多进程设置。

务实采用路径：

1. profile 你的 reasoning-task workload。测量 tokens 中有多少是 exploratory（multiple strategies、case analyses、search），多少是 linear。
2. 如果 exploration 占主导，运行 two-worker Hogwild! experiment，测量 wall-time improvement。
3. 如果 improvement 低于 1.3x，你处于 coordination-dominated regime。回退到 single-worker。
4. 如果 improvement 超过 1.5x，推到 N=4 再测。diminishing returns 通常在 N=4-8 附近出现。

与 speculative decoding 组合：每个 Hogwild! worker 可以独立使用 spec decode。两个 speedups 大致相乘，因此 3x spec decode 加 1.8x Hogwild!，有效上可达到相对 naive single-worker decoding 的 5.4x。

## Ship It / 交付

本课会产出 `outputs/skill-parallel-inference-router.md`。给定 reasoning workload profile（token budget、task parallelism profile、model family、deployment target），它会在 voting、tree-of-thought、multi-agent、Hogwild! 和 speculative decoding strategies 之间做 routing。

## Exercises / 练习

1. 使用默认设置运行 `code/main.py`。确认 N=2 Hogwild! configuration 在同样 wall time 下比 N=1 baseline 产生更多 work-tokens。

2. 降低 coordination heuristic 强度（设置 `coordination_weight=0.1`）。重新运行。展示 speedup 崩溃。解释原因：workers 无法协调时会重复劳动。

3. 为一个 50k-token reasoning task 计算期望 Hogwild! speedup，其中 `p=0.8, c=500`，N=4 workers。再为 1k-token chat task 计算，其中 `p=0.3, c=200`，N=4。为什么一个赢，一个输？

4. 阅读 Hogwild! paper 的 Section 4（preliminary evaluation）。识别作者报告的两个 failure modes。描述更好的 coordination prompt 如何缓解它们。

5. 在 toy 中组合 Hogwild! 和 speculative decoding：每个 worker 内部使用 2-token spec-decode。报告 multiplicative speedup。当两个 workers 都想扩展同一个 shared-cache prefix 时，会出现什么 bookkeeping 问题？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Hogwild! | “Parallel workers, shared cache” | N 个相同 LLM 实例并发运行，共享一个 KV cache；通过 self-prompting 涌现 coordination |
| Shared KV cache | “The coordination medium” | 一个持续增长的 KV buffer，所有 workers 都读写；让 workers 之间立即可见 token |
| Emergent coordination | “No training needed” | 具备 reasoning 能力的 LLM 可读取 shared cache 并分工，无需 fine-tuning 或显式协议 |
| Coordination overhead (c) | “Tokens spent orienting” | 每个 worker 读取扩展 cache 并决定下一步的成本；必须相对总 decode time 足够小 |
| Parallelizable fraction (p) | “What can run in parallel” | task-level parallelism：总工作中不是内在顺序的那部分比例 |
| RoPE enables Hogwild! | “Rotary positions are shift-invariant” | 因为 positions 是 rotations，写入 shared cache 不需要重算 prior tokens |
| Voting ensemble | “Run N, pick the majority” | 最简单的 parallel inference topology；适合 classification，不太适合 long-form reasoning |
| Tree of thought | “Branch and prune” | 探索多个 reasoning branches 并剪枝的策略；有显式 coordination logic |
| Multi-agent framework | “Assign sub-tasks” | 每个 agent 获得一个 role，由 coordinator 编排；protocol overhead 很重 |

## Further Reading / 延伸阅读

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261)：Hogwild! paper，包含 QwQ 和 DeepSeek-R1 上的 preliminary evaluation
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730)：原始 Hogwild!，命名来源
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864)：RoPE，让 shared-cache inference 可行的性质
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601)：tree-of-thought reasoning strategy，Hogwild! 与其正交
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192)：speculative decoding，Hogwild! 可组合的 within-sequence parallelism
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm)：论文实验的 single source of truth
