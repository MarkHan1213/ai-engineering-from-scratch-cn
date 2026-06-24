# vLLM Serving Internals: PagedAttention, Continuous Batching, Chunked Prefill / vLLM Serving 内部机制：PagedAttention、Continuous Batching、Chunked Prefill

> vLLM 在 2026 年的主导地位来自三个叠加的默认能力，而不是某个单点技巧。PagedAttention 总是开启。Continuous batching 在 decode iterations 之间把新请求注入 active batch。Chunked prefill 把长 prompt 切片，让 decode tokens 不会饿死。三者都打开时，一张 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 concurrent 下可达到 2,200-2,400 tok/s，约比 vLLM 自身默认高 25%，是 naive PyTorch loop 的 3-4x。本课会把 scheduler 和 attention kernel 读到能画图的程度，并以 `code/main.py` 中的 toy continuous batcher 结束，它会像 vLLM 一样调度 prefill 和 decode。

**类型：** 学习
**语言：** Python（stdlib, toy continuous batching scheduler）
**前置知识：** 第 17 阶段 · 01（Model Serving）, 第 11 阶段（LLM Engineering）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 把 PagedAttention 解释为 KV cache allocator：blocks、block tables，以及为什么生产负载下 fragmentation 能保持在 4% 以下。
- 在 iteration level 画出 continuous batching：完成的 sequences 如何离开 batch，新 sequences 如何加入而不用 drain。
- 用一句话描述 chunked prefill，并指出它保护的 latency metric（提示：是 TTFT tail，不是 mean throughput）。
- 说出 2026 年 vLLM v0.18.0 的 gotcha：会咬到同时打开所有优化的团队。

## The Problem / 问题

Naive PyTorch serve loop 一次跑一个请求：tokenize、prefill、decode 直到 EOS、返回。一个用户时可以。100 个用户时，就是一队耐心排队的人。显而易见的修复是 static batching，但它会把每个请求 pad 到窗口里最长 prompt，把每个 decode pad 到最长预期 output，并让整个 batch 卡在最慢 sequence 上。你为从未使用的 padding 付费，快请求还要等慢请求。

vLLM 同时解决三个问题。PagedAttention 阻止 KV cache fragmentation 像经典 contiguous allocation 那样吃掉 60-80% GPU memory。Continuous batching 允许请求在每个 decode iteration 之间加入和离开 batch，让 batch 永远装满真实工作。Chunked prefill 把 32k-token prompt 切成约 512-token slices，与 decode 交错，避免一个长 prompt 冻住 GPU 上所有 decode token。

2026 年生产默认是三者全开。你需要理解每个能力做什么，因为它们的 failure modes 都在 scheduler 上，不在 model 上。

## The Concept / 概念

### PagedAttention as a virtual memory system / 把 PagedAttention 看成虚拟内存系统

KV cache 对每个 sequence 的大小是 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。Llama 3.3 70B 在 8192 tokens 下，每个 sequence 的 BF16 KV cache 大约 1.25 GB。如果你为每个 request 预留 8192 slots，但平均请求只使用 1500 tokens，你浪费了约 82% 预留 HBM。Classic batching 会为这个浪费买单。

PagedAttention 借用了 OS virtual memory 的想法。每个 sequence 的 KV cache 不是连续的，而是按固定大小 blocks 分配（默认 16 tokens）。每个 sequence 有一个 block table，把逻辑 token positions 映射到物理 block IDs。当 sequence 超过已分配 blocks，就再加一个 block。结束后，它的 blocks 回到池里。

Fragmentation 从 60-80%（classic）降到 4% 以下（PagedAttention）。你不需要用 flag 开启 PagedAttention，它是 vLLM 唯一的 allocator。真正的旋钮是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载 weights 和 activations 后，预留多少 HBM 给 KV blocks。

### Continuous batching at the iteration level / iteration 级 continuous batching

旧的 “dynamic batching” 等一个窗口（例如 10 ms）填满 batch，然后运行 prefill + decode + decode + decode，直到所有 sequence 结束。快 sequence 早早离开并空等，GPU 继续处理慢 sequence。

Continuous batching 在每个 decode step 之间运行。把运行中的 sequences 集合叫 `RUNNING`。每次 iteration：

1. `RUNNING` 中刚命中 EOS 或 max_tokens 的 sequence 会被移除。
2. Scheduler 查看 waiting queue。如果有空闲 KV blocks，就接纳新 sequences（prefill 或 resumed）。
3. Forward pass 在当前 `RUNNING` 上运行，每个 sequence 产生一个新 token。

Batch size 从不 pad 到固定数。输出位置不同的 sequences 共享一次 fused forward。2026 年 vLLM 称之为 `V1 scheduler`。关键不变式是：scheduler 每个 decode iteration 运行一次，而不是每个 request 运行一次。

### Chunked prefill protects TTFT tail / Chunked prefill 保护 TTFT tail

Prefill 是 compute-bound。Llama 3.3 70B 上一个 32k-token prompt，在一张 H100 上纯 prefill 约 800 ms。prefill 运行期间，batch 里其他所有 sequence 的 decode tokens 都要等。在 serving loop 中，一个长 prompt 的 first-token latency（TTFT）会变成几十个用户的 inter-token latency（ITL）抖动。

Chunked prefill 把 prefill 切成固定大小 chunks（默认 512 tokens），每个 chunk 作为一个调度单元。chunks 之间，scheduler 可以让 decode sequences 各前进一步。你用很小的 absolute prefill latency 增量（每个 chunk 几 ms）换取低得多的 decode-time jitter。已发布 benchmark 中，混合负载下 P99 ITL 从约 50 ms 降到约 15 ms。

### The three defaults interact / 三个默认能力会互相作用

三个特性彼此依赖。PagedAttention 给 scheduler 一个细粒度 KV resource，可以拿来做 tradeoff。Continuous batching 需要这个细粒度 resource，才不会在接纳新 sequence 时引发全局 reshuffle。Chunked prefill 是 scheduler 在同一个 `RUNNING` list 上做出的决策；它是又一个 scheduler policy，不是单独系统。

你不需要知道每个 flag。你需要知道 scheduler 优化什么：在 KV-block budget 下最大化 goodput，同时受 chunked prefill slicing 约束。

### The 2026 v0.18.0 gotcha / 2026 年 v0.18.0 的 gotcha

在 vLLM v0.18.0 中，`--enable-chunked-prefill` 不能与 draft-model speculative decoding（`--speculative-model`）组合。文档例外是 V1 scheduler 中的 N-gram GPU speculative decoding。团队如果不读 release notes 就把所有 flag 打开，会在启动时得到 run-time error，而不是温和退化。如果你的 speculative gain 值得开启 chunked prefill，就重新审视选择；2026 年的正确答案经常是 EAGLE-3 without chunked prefill，而不是 draft model plus chunked prefill 这种无法编译的组合。

### Numbers you should remember / 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 concurrent，三者全开：2,200-2,400 tok/s。
- 同模型，default vLLM（no chunked prefill）：约 1,800 tok/s。
- 同模型，naive PyTorch forward loop：约 600 tok/s。
- PagedAttention 在生产负载下的 KV fragmentation waste：<4%。
- 混合负载下 P99 ITL：with chunked prefill 约 15 ms；without 约 50 ms。

### What the scheduler looks like / scheduler 长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是这个 loop 的 stdlib Python 版本，只是使用假的 token counts 和假的 forward latency。运行它可以看到 chunked prefill 如何在长 prefill 期间保持 decode sequences 继续前进。

```figure
tensor-parallel
```

## Build It / 动手构建

从 `code/main.py` 的 toy scheduler 开始，切换 naive、static、continuous 和 chunked modes，直接观察 PagedAttention 之外的调度策略如何影响 TTFT 与 ITL。

## Use It / 应用它

`code/main.py` 模拟一个 vLLM-style scheduler，并允许切换功能。运行它观察：

- `NAIVE` mode：一次一个请求，不 batching。
- `STATIC` mode：pad and wait，经典 batching。
- `CONTINUOUS` mode：iteration-level admission and release。
- `CONTINUOUS + CHUNKED` mode：prefill slices 与 decode 交错。

输出包括 total throughput（tokens per virtual second）、TTFT mean 和 P99 ITL。`CONTINUOUS + CHUNKED` 这一行在 mixed traffic 上应该占优。

## Ship It / 交付它

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定 serving config（batch size、KV memory utilization、chunked prefill size、speculative config），它会输出 scheduler diagnosis，指出三种默认能力中哪一个正在成为瓶颈，以及该调哪个参数。

## Exercises / 练习

1. 运行 `code/main.py`。在短长请求混合的 workload 上比较 `STATIC` 和 `CONTINUOUS`。throughput gap 来自哪里：prefill efficiency、decode efficiency，还是 tail latency？
2. 修改 toy scheduler，加入 `--max-num-batched-tokens`。H100 上跑 Llama 3.3 70B FP8 时，合适值是什么？（提示：它是 KV block size 和 free blocks 数量的函数，不是 raw HBM。）
3. 重新阅读 vLLM v0.18.0 release notes。哪些 flag 组合互斥？列出来。
4. 对一个 1,000 requests 的 trace 计算 KV cache fragmentation waste：mean 1,500 output tokens、std 600 tokens。在（a）8192 max 的 contiguous per-request allocation，（b）16-token blocks 的 PagedAttention 下分别是多少。
5. 用一段话解释为什么 chunked prefill 帮助 P99 ITL，但单独看不提高 throughput。实际 throughput win 从哪里来？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| PagedAttention | “the KV trick” | KV cache 的 fixed-size block allocator；fragmentation <4% |
| Block table | “the page table” | 每个 sequence 从 logical token position 到 physical KV block 的映射 |
| Continuous batching | “dynamic batching, but right” | 每个 decode iteration 做 admit/release decision |
| Chunked prefill | “prefill splitting” | 把长 prefill 切成 512-token slices，与 decode 交错 |
| TTFT | “first token time” | prefill + queue + network；长 prompt 时由 prefill 主导 |
| ITL | “inter-token latency” | 相邻 decode tokens 之间的时间；由 batch size 主导 |
| Goodput | “throughput that meets SLO” | 每个请求仍满足 TTFT 和 ITL 目标时的 tokens/sec |
| V1 scheduler | “the new scheduler” | vLLM 2026 scheduler；N-gram spec decode 是兼容 chunked-prefill 的路径 |
| `--gpu-memory-utilization` | “the memory knob” | weights 和 activations 后为 KV blocks 预留的 HBM 比例 |

## Further Reading / 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — chunked-prefill 与 speculative-decoding compatibility 的官方来源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 release cadence 和 version-specific behavior。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 原始文章，至今仍定义如何理解 allocator。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — fragmentation analysis 和 scheduler design。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — V1 scheduler 细节 walkthrough 和 flame graphs。
