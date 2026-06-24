# DualPipe Parallelism / DualPipe 并行

> DeepSeek-V3 使用 2,048 张 H800 GPU 训练，MoE experts 分散在不同节点上。跨节点 expert all-to-all communication 的成本达到了每 1 GPU-hour compute 对应 1 GPU-hour comm。GPU 有一半时间在空等。DualPipe（DeepSeek, 2024 年 12 月）是一种双向 pipeline，会把 forward 和 backward computation 与它们触发的 all-to-all comms 重叠起来。Bubbles 下降，吞吐上升；保留两份 model-parameter copies（名字里的 "dual"）在 Expert Parallelism 已经把 experts 分散到 ranks 上之后并不昂贵。本课是 Learn 型走读，解释 DualPipe 实际做了什么，以及 Sea AI Lab 的 DualPipeV refinement 如何以稍微更紧的 bubble 为代价，去掉 2x parameter cost。

**类型：** Learn
**语言：** Python (stdlib, schedule simulator)
**前置要求：** Phase 10 · 05（distributed training, FSDP, DeepSpeed），Phase 10 · 14（open-model architectures and MoE）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出 DualPipe forward-backward chunk 的四个组件，以及为什么每个组件都需要自己的 overlap window
- 解释 scale 上的 pipeline bubble problem，以及 "bubble-free" 在实践与营销语言中分别意味着什么
- 手工追踪 8 个 PP ranks 和 16 个 micro-batches 的 DualPipe schedule，确认 forward 和 reverse streams 会填满彼此的 idle slots
- 陈述 DualPipeV（Sea AI Lab, 2025）的权衡：去掉 2x parameter replication，但当 Expert Parallelism 未激活时会付出略大的 bubble

## The Problem / 问题

在 2k H800 GPUs 上训练 671B MoE model，会遇到三个叠加瓶颈：

1. **Memory pressure。** 每张 GPU 持有模型切片。sequence 8k、61 layers、128 heads 下的 activation memory 极大。
2. **Pipeline bubbles。** 传统 pipeline parallelism（GPipe、1F1B）会让 GPU 在等待自身 stage 的 input 或 gradient 时空闲。8 个 stages 下，即使使用 1F1B scheduling，也可能有约 12% GPU time 成为 bubble。
3. **Cross-node all-to-all。** 带 expert parallelism 的 MoE 会把 experts 分散到节点之间。每个 forward pass 都会触发一次 all-to-all 去把 tokens 分发给 experts，再触发一次 all-to-all 做 combine。到 2k GPUs 时，这很容易变成 1:1 compute-to-comm ratio。

每个问题都有单独解法：memory 用 gradient checkpointing，pipeline bubbles 用 Zero Bubble（Sea AI Lab, 2023），all-to-all 用 expert-parallel comm kernels。DualPipe 做的是让它们彼此配合。这个 schedule 在一个 forward-backward chunk 内重叠 compute 和 comm，从 pipeline 两端同时注入 micro-batches，并用 resulting schedule 把 all-to-all 藏进 compute windows。

报告结果：DeepSeek-V3 的 14.8T-token training run 中，pipeline bubbles 接近消除，GPU utilization 超过 95%。

## The Concept / 概念

### Pipeline parallelism refresher / Pipeline parallelism 复习

把一个 N-layer model 拆到 P 个 devices 上。Device `i` 持有 layers `i * N/P .. (i+1) * N/P - 1`。micro-batch 从 device 0 到 P-1 做 forward，再从 P-1 到 0 做 backward。每个 device 只能在前一个 device 送来输出后开始自己的 forward stage，也只能在下游 device 送来 upstream gradient 后开始 backward。

GPipe（Huang et al., 2019）一次调度一个 micro-batch，会浪费大量 GPU 时间。1F1B（Narayanan et al., 2021）交错多个 micro-batches 的 forward 和 backward passes。Zero Bubble（Qi et al., 2023）把 backward pass 拆成两部分：backward-for-input（B）和 backward-for-weights（W），并安排它们填掉 bubble。Zero Bubble 之后，pipeline 已经接近紧凑。

DualPipe 是下一步。它在上面增加两个想法：

### Idea 1: chunk decomposition / 想法 1：Chunk 分解

每个 forward chunk 被拆成四个组件：

- **Attention。** Q/K/V projections、attention、output projection。
- **All-to-all dispatch。** 把 tokens 发送给对应 experts 的跨节点通信。
- **MLP。** MoE expert computation。
- **All-to-all combine。** 把 expert outputs 取回来的跨节点通信。

backward chunk 会为这些组件增加 gradient 版本。DualPipe 会这样调度它们：all-to-all dispatch 与下一个 chunk 的 attention compute 并行，all-to-all combine 与后续 chunk 的 MLP compute 并行。

### Idea 2: bidirectional scheduling / 想法 2：双向调度

多数 pipeline schedules 从 stage 0 注入 micro-batches，并流向 stage P-1。DualPipe 从两端同时注入 micro-batches。stage 0 看到从自己这里出发的 forward micro-batches；stage P-1 也看到从自己那里出发的 forward micro-batches。两条 streams 在中间相遇。

为了做到这一点，device `i` 必须同时持有 early-pipeline layer `i` 和 late-pipeline layer `P - 1 - i`。这就是 DualPipe 中 "dual" 的部分：每个 device 保留服务两个方向所需的两份 model layers。在 DeepSeek-V3 的 scale 下，这是 2x parameter replication cost。它可接受，是因为 Expert Parallelism 已经把 MoE experts 分得很薄，replicate 非 expert layers 两次只是小菜。

关键是：一个方向的 forward stream 和另一个方向的 backward stream 会恰好在单向 schedule 的 bubbles 位置重叠。bubbles 消失。

### A hand-traced schedule / 手工追踪一个 schedule

考虑 P = 4 ranks、8 micro-batches，分成 4 个 forward / 4 个 reverse。时间从左到右，行是 device ranks。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

读 "F4/F5R" 这种记号：rank 1 在同一个 time slot 中同时运行 micro-batch 4 的 forward（pipeline 中从左到右）和 micro-batch 5 的 forward（从右到左）。这就是 "bidirectional" 在操作上的含义。

在 rank 2 处，两条 streams 更早重叠；在 rank 0 和 P-1 处，重叠最晚。在 schedule 的稳定中段，每个 rank 都运行一个方向的 forward-of-X，并与另一个方向的 backward-of-Y 重叠。Compute 保持忙碌。forward pass 的 all-to-all dispatches 藏在 backward compute 里。all-to-all combines 藏在 forward compute 里。bubbles 被挤出去。

### Bubble accounting / Bubble 核算

标准 1F1B pipeline bubble（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble refinement 会把它降下来，但不是零。DualPipe 在稳定阶段，如果 micro-batch count 可被 2 倍 pipeline depth 整除，就有 zero bubble。在稳定阶段之外（warmup 和 cooldown），仍有一些 bubble，但不会随 micro-batch 数增长，这是论文强调的关键性质。

营销语言里叫 "bubble-free"。技术语言里，是 bubbles 不随 micro-batch count 增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）指出，只有当 Expert Parallelism 不是瓶颈时才有完整 zero-bubble；当 EP-driven all-to-all 存在时，调度妥协总会存在。

### DualPipeV — the refinement / DualPipeV：改进

Sea AI Lab（2025）观察到，当 EP comm overlap 不是重点时，2x parameter replication 很浪费。他们的 DualPipeV schedule 把 bidirectional injection 折叠成一个 "V-shape" schedule，只需单份参数。bubble 比 DualPipe 略大，但 memory savings 很可观。DeepSeek 在 open-source DualPipe implementation 中把 DualPipeV 作为 EP-off mode 采用。

权衡如下：

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Param copies per device | 2 | 1 | 1 | 1 |
| Bubble vs micro-batches | constant | small growth | grows | grows |
| Compute-comm overlap | full | partial | minimal | partial |
| Use when | EP-heavy MoE | dense or EP-light | baseline | any pipeline |

### What it means for a 14.8T-token run / 对 14.8T-token run 意味着什么

DeepSeek-V3 的 pre-training 在 2,048 张 H800 GPUs 上消耗 14.8T tokens，用时约 2.8M GPU-hours。若使用朴素 1F1B，他们会在 pipeline bubbles 上损失 12-15%，也就是 340-420K GPU-hours，足够训练一个完整 70B 模型。DualPipe 回收了其中大部分。没有内部 logs 很难直接量化贡献，但论文声称训练平均 GPU utilization 超过 95%。

对较小 run（小于 1k GPUs），DualPipe 可能过度；pipeline bubbles 相对于总成本更小，dense-model training 很少撞到 all-to-all 瓶颈。对多千 GPU 规模的 frontier MoE training，它几乎是必需项。

### Where it sits in the stack / 它位于栈的哪里

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 在 ranks 间 shard model parameters；DualPipe 在 ranks 间调度 compute。二者可以组合。
- 兼容 **ZeRO-3** gradient sharding。两份参数复制的 bookkeeping 需要与 ZeRO 的 sharded gradients 协作。
- 需要针对具体 cluster topology 调优的 **custom all-to-all kernels**。DeepSeek open-source kernels 是参考实现。

```figure
expert-capacity
```

## Build It / 动手构建

本课是 Learn 型，但 `code/main.py` 提供了一个 pipeline schedule simulator。它接收 `(P, n_micro_batches, schedule)`，并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 的 stable-phase utilization。这是教学工具；数字与论文中的定性主张一致，但不是 production measured speedup 的声明。

用它探索不同 pipeline depth、micro-batch count 和 schedule 的 bubble fraction，重点观察 1F1B 如何随 micro-batch 增长而扩张，而 DualPipe 不会。

## Use It / 使用它

`code/main.py` 是 pipeline schedule simulator。它接收 `(P, n_micro_batches, schedule)`，并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 的 stable-phase utilization。这是教学工具，数字匹配论文中的定性说法，但不声称是 production measured speedup。

这个 simulator 的价值在于：用不同 P 和 micro-batch counts 运行它，观察 1F1B 的 bubble fraction 如何增长，而 DualPipe 不会。

真实训练 run 中的集成注意事项：

- 选择能被 micro-batch count 整除的 pipeline-parallel depth。
- 确保 expert-parallel mesh 支持 bidirectional all-to-all。DeepSeek kernels 是参考。
- 第一次实现时，预期会在 schedule 本身上花一周 debugging time。bookkeeping 很繁琐。
- 监控每个 rank 的 GPU utilization，而不仅是聚合值。DualPipe 的收益来自收紧 stragglers。

## Ship It / 交付

本课会产出 `outputs/skill-dualpipe-planner.md`。给定 training cluster specification（GPU count、topology、interconnect、model shape），它会推荐 pipeline parallelism strategy、应使用的 scheduling algorithm，以及目标 scale 下的 expected bubble fraction。

## Exercises / 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU utilization difference，并换算成每百万 training tokens 回收的 GPU-hours。

2. 手工画出 `(P=4, micro_batches=8, schedule=dualpipe)` 的 schedule table。用 micro-batch ID 和 direction 标注每个 time slot。识别第一个没有 bubble 的 time slot。

3. 阅读 DeepSeek-V3 technical report（arXiv:2412.19437）的 Figure 5。识别 DualPipe forward chunk 中 all-to-all dispatch 的 overlap window。解释 compute schedule 如何隐藏它。

4. 分别计算 P=8 pipeline stages 的 70B dense model，以及 P=16 pipeline stages 的 671B MoE model 上 DualPipe 的 2x parameter overhead。说明为什么 MoE case 的 overhead 比例更小（大多数参数是 experts，并跨大型 EP group sharded）。

5. 对比 DualPipe 和 Chimera（2021 年的一个竞争性 bidirectional scheduler）。以论文 Section 3.4 为参考，识别 DualPipe 添加了 Chimera 不具备的两个具体性质。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Pipeline bubble | “Idle time per rank” | pipeline stage 等待 input 或 gradient 时浪费的 GPU cycles |
| 1F1B | “Default pipeline schedule” | one forward / one backward 交错调度；DualPipe 对比的 baseline |
| Zero Bubble | “Sea AI Lab 2023” | 把 backward 拆成 B（input gradient）和 W（weight gradient）；几乎完全收紧 pipeline |
| DualPipe | “DeepSeek-V3 schedule” | bidirectional pipeline 加 compute-comm overlap；bubbles 不随 micro-batch count 增长 |
| DualPipeV | “Cut-in-half” | V-shape refinement，去掉 2x parameter replication，代价是 bubbles 略大 |
| Chunk | “Unit of pipeline work” | 一个 micro-batch 通过一个 pipeline stage 的 forward 或 backward pass |
| All-to-all dispatch | “Send tokens to experts” | 把 tokens 路由到其 assigned MoE experts 的 cross-node comm |
| All-to-all combine | “Bring expert outputs back” | MLP 之后收集 expert outputs 的 cross-node comm |
| Expert Parallelism (EP) | “Experts across GPUs” | 把 MoE experts sharded 到 ranks 上，使不同 GPUs 持有不同 experts |
| Pipeline Parallelism (PP) | “Layers across GPUs” | 把 model layers sharded 到 ranks 上；DualPipe 调度的维度 |
| Bubble fraction | “Wasted GPU time” | (bubble_time / total_time)；DualPipe 试图推向零的比例 |

## Further Reading / 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437)：主要 DualPipe 参考
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe)：open-source reference implementation，包含 DualPipeV（Cut-in-half）mode
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241)：Zero Bubble 前身
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63)：启发 DeepSeek EP-off mode 的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377)：DualPipe 对比的 1F1B schedule
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965)：原始 pipeline parallelism 论文和 bubble problem
