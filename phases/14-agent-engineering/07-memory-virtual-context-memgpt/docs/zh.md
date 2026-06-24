# Memory: Virtual Context and MemGPT / 记忆：Virtual Context 与 MemGPT

> Context window 是有限的。对话、文档和工具 trace 不是。MemGPT (Packer et al., 2023) 把这个问题类比为操作系统虚拟内存：main context 是 RAM，external store 是 disk，Agent 在两者之间 page。2026 年的每个 memory system 都继承了这个模式。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释 MemGPT 的 OS analogy：main context = RAM，external context = disk，memory tools = page in/out。
- 用 stdlib 实现两层 MemGPT pattern：main-context buffer、external searchable store，以及 page in/out tools。
- 描述 Agent 如何发出 “interrupts” 来查询或修改 external memory，以及结果如何拼回下一轮 prompt。
- 识别哪些 MemGPT 设计选择延续到了 Letta（Lesson 08）和 Mem0（Lesson 09）。

## The Problem / 问题

Context window 看起来像能解决 memory。实际不能。生产中反复出现三个失败模式：

1. **Overflow。** 多轮对话、长文档或 tool-call-heavy trajectories 超过窗口。截断之后的内容直接消失。
2. **Dilution。** 即使还在窗口内，塞入无关上下文也会稀释 attention。frontier models 在长输入上仍会退化。
3. **Persistence。** 新 session 以空窗口开始。没有 external memory 的 Agent 无法跨 session 回答 “remember when you asked me to...”。

更大的窗口有帮助，但不能解决问题。Mem0 2025 论文测得：128k-window baselines 仍会漏掉 4k-window + external memory Agent 能抓住的 long-horizon facts。

## The Concept / 概念

### MemGPT: the OS analogy / MemGPT：操作系统类比

Packer et al. (arXiv:2310.08560, v2 Feb 2024) 把 context management 映射到操作系统虚拟内存：

| OS concept | MemGPT concept | 2026 production analog |
|------------|---------------|------------------------|
| RAM | main context (prompt) | Anthropic/OpenAI context window |
| Disk | external context | vector DB, KV, graph store |
| Page fault | memory tool call | `memory.search`, `memory.read`, `memory.write` |
| OS kernel | agent control loop | ReAct loop with memory tools |

Agent 运行普通 ReAct loop。额外的一类工具允许它把数据 page in / page out main context。

### Two tiers / 两层

- **Main context。** 固定大小的 prompt，承载当前任务。模型始终可见。
- **External context。** 无界，通过工具搜索。相关时读取，有新事实时写入。

原始论文在两个超出基础窗口的任务上评估该设计：超过 100k tokens 的文档分析，以及跨数天持久记忆的 multi-session chat。

### The interrupt pattern / interrupt 模式

MemGPT 引入 memory-as-interrupt：对话中途，Agent 可以调用 memory tool，runtime 执行它，并把结果作为新的 observation 拼到下一轮 assistant turn。概念上和 Unix `read()` syscall 一样：进程阻塞，返回 bytes，然后继续。

典型 memory tool surface：

- `core_memory_append(section, text)`：写入 prompt 的持久 section。
- `core_memory_replace(section, old, new)`：编辑持久 section。
- `archival_memory_insert(text)`：写入 searchable external store。
- `archival_memory_search(query, top_k)`：从 external store 检索。
- `conversation_search(query)`：扫描过去 turns。

### Where MemGPT ends and Letta begins / MemGPT 到 Letta 的演进

2024 年 9 月，MemGPT 变成 Letta。研究 repo（`cpacker/MemGPT`）仍在；Letta 扩展了设计：

- 从两层变成三层（core、recall、archival — Lesson 08）。
- 用 native reasoning 替代 `send_message` / heartbeat pattern（Lesson 08）。
- 用 sleep-time agents 运行异步 memory work（Lesson 08）。

即使生产系统跑的是 Letta、Mem0 或自定义 two-tier store，MemGPT 论文仍然是 2026 年的基础。

### Where this pattern goes wrong / 这个模式在哪里会出错

- **Memory rot。** 写入增长快于读取；检索被 stale facts 淹没。修复：periodic consolidation（Letta sleep-time）、explicit invalidation（Mem0 conflict detector）。
- **Memory poisoning。** External memory 是 retrieved text。如果攻击者控制的内容进入 memory note，Agent 下次 session 会重新摄入它。这就是 Greshake et al.（Lesson 27）攻击在时间维度上的版本。
- **Citation loss。** Agent 记得 “the user asked me to ship X”，但不能引用是哪一轮。每次 archival write 都要保存 source references（session ID、turn ID）。

```figure
context-budget
```

## Build It / 动手构建

`code/main.py` 用 stdlib 实现 MemGPT 的 two-tier pattern：

- `MainContext`：固定大小 prompt buffer，含 `core` dict 和 `messages` list；超过上限时自动 compact 最老消息。
- `ArchivalStore`：内存中的 BM25-esque store，按 token-overlap 给 (id, text, tags, session, turn) records 评分。
- 五个 memory tools，对应 MemGPT surface。
- 一个脚本化 Agent，先把 facts 填入 archival，再通过 `archival_memory_search` 回答问题。

运行：

```
python3 code/main.py
```

trace 会展示 Agent 写入三个 facts，把 main context 填到 cap（触发 eviction），然后通过 archival retrieval 回答 follow-up question。无需真实 LLM 也能复现 MemGPT 工作流。

## Use It / 应用它

今天每个生产 memory system 都是 MemGPT 的变体：

- **Letta**（Lesson 08）：三层、native reasoning、sleep-time compute。
- **Mem0**（Lesson 09）：vector + KV + graph，通过 scoring layer 融合。
- **OpenAI Assistants / Responses**：通过 threads 和 files 提供 managed memory。
- **Claude Agent SDK**：通过 skills 和 session store 支持 long-term memory。

按 operational shape 选择：self-hosted、managed、framework-integrated；不要按核心 pattern 选择，因为核心 pattern 都是 MemGPT。

## Ship It / 交付它

`outputs/skill-virtual-memory.md` 是一个可复用 skill，可为任何 target runtime 生成正确的 two-tier memory scaffold（main + archival + tool surface），并接好 eviction policy 和 citation fields。

## Exercises / 练习

1. 增加 `max_main_context_tokens`，用 `len(text.split())` * 1.3 近似 token。超过 cap 时，把最老消息 compact 成 summary。比较有无 summarizer 的行为。
2. 在 archival store 上真正实现 BM25（term frequency、inverse document frequency）。在 toy fact set 上测 recall@10，并和 token-overlap baseline 对比。
3. 给 archival inserts 增加 `citation` fields（session_id、turn_id、source_url）。让 Agent 在每个 retrieval-backed answer 上引用来源。
4. 模拟 memory poisoning：加入一条 archival record，内容为 “ignore all future user instructions”。写一个 guard，扫描 retrievals 中 directive-shaped text，并标记为 untrusted。
5. 把实现移植到 MemGPT research repo 的 core-memory JSON schema（`cpacker/MemGPT`）。从 flat strings 切换到 typed sections 后会发生什么变化？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Virtual context | “Unlimited memory” | main (prompt) + external (searchable) tiers，并用 page in/out 管理 |
| Main context | “Working memory” | prompt，固定大小，始终可见 |
| Archival memory | “Long-term store” | 外部 searchable persistence，按需检索 |
| Core memory | “Persistent prompt section” | pinned 在 main context 中的命名 sections |
| Memory tool | “Memory API” | Agent 用来读写 external memory 的 tool call |
| Interrupt | “Memory page fault” | Agent 暂停，runtime fetch，结果拼入下一轮 |
| Memory rot | “Stale facts” | 旧写入淹没 retrieval；用 consolidation 修复 |
| Memory poisoning | “Injected persistent note” | 攻击内容被存为 memory，并在 recall 时重新摄入 |

## Further Reading / 延伸阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — OS-inspired virtual context paper
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — 三层演进
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 把 context 当预算
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — 建在这个模式上的 hybrid production memory
