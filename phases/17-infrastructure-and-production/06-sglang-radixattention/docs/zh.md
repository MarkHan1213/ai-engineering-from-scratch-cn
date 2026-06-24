# SGLang and RadixAttention for Prefix-Heavy Workloads / 面向 Prefix-Heavy 工作负载的 SGLang 与 RadixAttention

> SGLang 把 KV cache 当成一等可复用资源，存放在 radix tree 里。vLLM 以 FCFS（first-come, first-served）调度请求，而 SGLang 的 cache-aware scheduler 会优先处理共享 prefix 更长的请求，本质上是 depth-first radix traversal，让 hot branches 常驻 HBM。在 Llama 3.1 8B、ShareGPT-like 1K prompts 上，SGLang 约 16,200 tok/s，vLLM 约 12,500，优势约 29%。在 prefix-heavy RAG workloads 上，优势可达 6.4x。在 voice-cloning-shaped workloads 上，cache hit rate 超过 86%。2026 年，它已经在 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS 等生产环境部署到 400,000+ GPUs。gotcha 是：当 prefix ordering 不一致时，6.4x 会消失；ordering 是工程师可控的杠杆。

**类型：** 学习
**语言：** Python（stdlib, toy radix-tree cache + cache-aware scheduler）
**前置知识：** 第 17 阶段 · 04（vLLM Serving Internals）, 第 14 阶段（Agentic RAG）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 画出 RadixAttention：prefixes 如何存入 radix tree，以及 KV blocks 如何在同一 branch 下的 sequences 间共享。
- 解释 cache-aware scheduling，以及为什么 FCFS 对 prefix-heavy traffic 是错的。
- 给定 prefix-cache hit rate 和 prompt length distribution，计算 expected speedup。
- 说出让 6.4x 真实发生而不是浪费掉的 prompt-ordering discipline。

## The Problem / 问题

经典 serving 把每个请求的 prompt 当成 opaque。即使 5,000 个 RAG requests 都以同一个 2,000-token system prompt 加同一个 retrieval preamble 开头，vLLM 也会把这 2,000-token prefix 预填 5,000 次。GPU 在反复做同一件事。

观察很简单：agentic 和 RAG workloads 的 prompts 几乎总是共享长 prefixes。System prompt、tool schemas、few-shot examples、retrieval headers、conversation history，都会跨请求重复。如果你把这个 prefix 的 KV cache 存一次并复用，就不用再 prefill。

RadixAttention 正是这么做的。Tokens 被索引到 radix tree；每个 node 拥有从 root 到该 node 路径上 token sequence 的 KV blocks。新请求沿树走：任何 token 匹配的 node 都复用该 node 的 KV blocks。Prefill cost 变成只与“新” suffix 成正比，而不是与完整 prompt 成正比。

挑战在 scheduling。如果两个请求共享 2,000-token prefix，第三个只共享同一 prefix 的 200 tokens，你应该把两个长共享请求一起服务，让长 prefix 留在 HBM。FCFS 会反过来：谁先到就服务谁，可能在下一个 long-prefix request 到达前把 hot branch 驱逐掉。

## The Concept / 概念

### The radix tree as a KV index / 把 radix tree 当成 KV 索引

Radix tree（compact trie）存储 token sequences。每个 node 拥有一个 token range，以及为这个 range 计算出的 KV blocks。Children 会把 sequence 延长一个或多个 tokens。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

一个新请求进来，内容是 system prompt + "Context: <doc A>" + "Question: Carol"。Scheduler 会走树：system prefix 匹配（复用 124 blocks），doc-A branch 匹配（复用 31 blocks），然后只为 "Question: Carol" 分配新 blocks（4 blocks）。Prefill cost：4 blocks 的新 tokens。没有树时：160 blocks。prefill 节省约 40x。

### Cache-aware scheduling / Cache-aware scheduling

Radix-tree-backed reuse 如果 cache 反复 churn，就没有意义。两个关键策略：

1. **Depth-first dispatch**。从 queue 中选择下一个请求时，优先选择和当前 running set rooted at same branch 的请求。这会把 hot branch pin 住。
2. **LRU at branch level, not block level**。驱逐整个 branch（从 shortest-used leaves 开始），而不是单独 blocks，让 cache 形状匹配 radix 形状。

FCFS 同时违反这两点。一个共享 2,000 tokens 的请求排在一个只共享 50 tokens 的请求后面，然后 2,000-token branch 可能为了容纳 50-token 请求而被驱逐。

### Benchmark numbers you should memorize / 你应该记住的 benchmark 数字

- Llama 3.1 8B，H100，ShareGPT 1K prompts：SGLang 约 16,200 tok/s，vLLM 约 12,500（约 29% edge）。
- Prefix-heavy RAG（same system + same doc，varying question）：SGLang 上最高 6.4x。
- Voice cloning workloads：86.4% prefix-cache hit rate。
- SGLang customers 的生产 hit rates：50-99%，取决于 prompt discipline。
- 2026 年部署规模：400,000+ GPUs。

### The ordering gotcha / ordering gotcha

6.4x 依赖稳定的 prompt-template ordering。如果你的 client 有时构造 `[system, tools, context, history, question]`，有时构造 `[system, context, tools, history, question]`，tree 就找不到共享 prefix。人眼看起来共享 prefix，对 radix tree 来说却是两条不同 token sequence。

工程师的杠杆：prompt template 就是 cache key。固定顺序。把所有 immutable 内容（system、tools、schemas）放在最前。retrieval context 放中间。user question 放最后。不要把 dynamic content 插入 prefix。

研究中的真实案例：把 dynamic content 移出 cacheable prefix，让一个部署的 cache hit rate 从 7% 提升到 74%。

### Where RadixAttention wins and loses / RadixAttention 赢在哪里、输在哪里

Wins:
- RAG（same retrieval preamble，varying question）。
- Agents（same tool schemas，varying query）。
- 带长 system prompt 的 chat。
- Voice / vision workloads 中重复 preambles。

Loses（退回到 vLLM-level throughput）:
- Unique prompts 的 single-shot generation（code completion、没有 system prompt 的 open-ended chat）。
- 每个请求都把 unique content 交错进 prefix 的 dynamic prompts。

### Why this is a scheduler problem, not just a kernel problem / 为什么这是 scheduler 问题，不只是 kernel 问题

你可以把 KV reuse 实现成一个 kernel trick。SGLang 的洞察是：只有当 scheduler 保持 hot branch resident，reuse 才真正有收益。朴素的 “reuse if available” policy 在混合负载下会让 cache churn。radix-tree-indexed scheduler 才是把 kernel trick 转成 29% production edge 的关键。

### Interplay with vLLM / 与 vLLM 的关系

两套系统不是绝对竞争。2026 年 vLLM 加入了 prefix caching（`--enable-prefix-caching`）和 cache-aware router（Rust 写的 vLLM Router）。差距缩小，但没有完全消失：SGLang 整个 stack 都是 radix-first；vLLM 是后来 grafted it on。对 prefix reuse 主导的 workloads，SGLang 仍是默认选择。对没有强 prefix patterns 的 general-purpose serving，vLLM 仍然相当或更好。

```figure
roofline
```

## Build It / 动手构建

在 `code/main.py` 里构建一个小型 radix-tree KV cache，再分别用 FCFS 和 cache-aware scheduling 跑同一批 prompts，验证 prefix ordering 对 hit rate 的影响。

## Use It / 应用它

`code/main.py` 实现一个 toy radix-tree KV cache，以及带两种 policies 的 scheduler：FCFS 和 cache-aware。它用同一 workload 跑两遍，报告 prefix-cache hit rate 和 throughput delta。然后运行一个 “scrambled ordering” workload，展示 6.4x 如何坍塌。

## Ship It / 交付它

本课产出 `outputs/skill-radix-scheduler-advisor.md`。给定 workload description（prompt-template shape、retrieval pattern、concurrent tenants 数量），它会产出 prompt-ordering prescription 和是否采用 SGLang 的 go/no-go。

## Exercises / 练习

1. 运行 `code/main.py`。在同一 workload 上比较 FCFS 和 cache-aware。delta 来自哪里：prefill savings、decode savings，还是 queue delay？
2. 修改 workload，让 prompts 随机排列 `[system, tools, context]`。重跑。hit rate 发生什么？为什么？
3. 计算在 Llama 3.1 8B 上，把 2,000-token system prompt 作为一个 radix branch 常驻 HBM 的成本。与 16-sequence batch 不做 prefix reuse 的成本比较。
4. 阅读 SGLang RadixAttention paper。用三句话解释为什么 tree-shaped LRU eviction 在 prefix-heavy load 下胜过 block-shaped LRU。
5. 客户报告只有 8% cache hit rate。说出三个可能原因，以及每个原因要跑什么诊断。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| RadixAttention | “the SGLang thing” | 把 KV cache 索引成 radix tree，让 shared prefixes 复用 blocks |
| Radix tree | “compact trie” | 每个 node 拥有 token range 和对应 KV blocks 的树 |
| Cache-aware scheduler | “hot-branch-first” | 优先调度共享 resident branch 的请求 |
| Prefix-cache hit rate | “how much of your prompt was free” | 从复用 KV blocks 服务的 prompt tokens 比例 |
| FCFS | “first-come first-served” | 默认调度，会破坏 prefix locality |
| Branch-level LRU | “evict the leaf” | 与 radix shape 匹配的驱逐策略 |
| Prompt template ordering | “the cache key” | prompt component 顺序决定 tree 能共享什么 |
| System prompt pinning | “resident prefix” | 保持 immutable system 部分常驻，避免 eviction thrash |

## Further Reading / 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — source 和 docs。
- [SGLang documentation](https://sgl-project.github.io/) — RadixAttention 和 scheduling details。
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — design reference。
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — benchmark numbers 和 scheduler rationale。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM 自己的 radix-like implementation，用于比较。
