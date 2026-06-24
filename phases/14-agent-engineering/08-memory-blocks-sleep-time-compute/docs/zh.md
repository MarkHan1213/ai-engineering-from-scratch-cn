# Memory Blocks and Sleep-Time Compute (Letta) / Memory Blocks 与 Sleep-Time Compute（Letta）

> MemGPT 在 2024 年变成 Letta。2026 年的演进增加了两个想法：模型可以直接编辑的离散 functional memory blocks，以及在主 Agent 空闲时异步整理记忆的 sleep-time agent。这是把 memory 扩展到单次对话之外的方式。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 07 (MemGPT)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 说出 Letta 使用的三层 memory（core、recall、archival）以及每层的作用。
- 解释 memory-block pattern：Human block、Persona block 和用户自定义 blocks 都是一等 typed objects。
- 描述什么是 sleep-time compute，为什么它位于 critical path 之外，以及为什么它可以使用比 primary agent 更强的模型。
- 实现一个脚本化 two-agent loop：primary agent 服务响应，sleep-time agent 在 turns 之间整理 blocks。

## The Problem / 问题

MemGPT（Lesson 07）解决了 virtual-memory control flow。生产中又出现三个问题：

1. **Latency。** 每次 memory operation 都在 critical path 上。如果 Agent 在用户等待时 prune、summarize 或 reconcile，tail latency 会暴涨。
2. **Memory rot。** 写入会积累。矛盾事实会留下。retrieval 被 stale content 淹没。
3. **Structure loss。** flat archival store 无法表达 “Human block 总是在 prompt 中；Persona block 总是在 prompt 中；Task block 按 session 切换”。

Letta (letta.com) 是 2026 年的重写。Memory blocks 把结构显式化；sleep-time compute 把整理工作移出 critical path。

## The Concept / 概念

### Three tiers / 三层

| Tier | Scope | Where it lives | Written by |
|------|-------|----------------|------------|
| Core | Always visible | Inside the main prompt | Agent tool call + sleep-time rewrites |
| Recall | Conversation history | Retrievable | Automatic turn logging |
| Archival | Arbitrary facts | Vector + KV + graph | Agent tool call + sleep-time ingest |

Core 是 MemGPT core。Recall 是 conversation buffer 及其被 evict 的尾部。Archival 是 external store。这个拆分清理了 MemGPT two-tier 设计中的语义重载。

### Memory blocks / 记忆块

block 是 core tier 中 typed、persistent、editable 的 section。原始 MemGPT 论文定义了两个：

- **Human block**：关于用户的事实（姓名、角色、偏好、目标）。
- **Persona block**：Agent 的自我概念（身份、语气、约束）。

Letta 泛化为任意用户自定义 blocks：当前目标的 `Task` block、代码库事实的 `Project` block、硬约束的 `Safety` block。每个 block 都有 `id`、`label`、`value`、`limit`（字符上限）、`description`（让模型知道什么时候编辑它）。

Blocks 通过工具面编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)`：把接近上限的 block 压缩。

### Sleep-time compute / Sleep-time compute

Letta 2025 年的新增能力：在 background 跑第二个 Agent，位于 critical path 之外。Sleep-time agents 处理 conversation transcripts 和 codebase context，把 `learned_context` 写入 shared blocks，并合并或 invalidate archival records。

自然得到三个性质：

- **No latency cost。** Primary responses 不等待 memory ops。
- **Stronger model allowed。** Sleep-time agent 可以使用更贵、更慢的模型，因为它不受响应延迟约束。
- **Natural consolidation window。** 用户不等待时，可以 dedup、summarize、invalidate contradicted facts。

这个形状很像人类工作：先做任务，睡一觉，长期记忆在夜里沉淀。

### Letta V1 and native reasoning / Letta V1 与原生 reasoning

Letta V1 (`letta_v1_agent`, 2026) 弃用 `send_message` / heartbeat 和 inline `Thought:` tokens，改用 native reasoning。Responses API（OpenAI）和带 extended thinking 的 Messages API（Anthropic）都把 reasoning 放到单独 channel，并跨 turns 透传（生产中跨 provider 加密）。控制循环仍是 ReAct。thought trace 是结构化的，不再是 prompt-shaped。

### Where this pattern goes wrong / 这个模式在哪里会出错

- **Block bloat。** 无限 `block_append` 很快达到 limit。在会超限的写入前接入 block summarizer。
- **Silent drift。** Sleep-time agent 重写 block，primary agent 却没注意到。给 blocks 做 version，并在 trace 中展示 diff。
- **Poisoned consolidation。** Sleep-time agent 把攻击者可达内容处理进 core。Lesson 27 同样适用于 sleep-time surface。

## Build It / 动手构建

`code/main.py` 实现：

- `Block`：id、label、value、limit、description。
- `BlockStore`：CRUD + `near_limit(label)` helper。
- 两个脚本化 Agents：`PrimaryAgent` 服务一轮，`SleepTimeAgent` 在 turns 之间整理。
- 一条 trace：展示三轮对话中的 block writes，以及一次 sleep-time pass 如何 summarize block 并 invalidate stale fact。

运行：

```
python3 code/main.py
```

transcript 会展示拆分：primary turns 快速响应并产生 raw writes；sleep pass 负责压缩和清理。

## Use It / 应用它

- **Letta** (letta.com)：参考实现，可 self-host 或使用 managed cloud。
- **Claude Agent SDK skills**：可视为 block-shaped knowledge。skill 是命名、版本化、可检索的 instruction block，Agent 按需加载。
- **Custom builds**：适合想控制存储 backend 的团队。采用 Letta API contract，后续可以迁移。

## Ship It / 交付它

`outputs/skill-memory-blocks.md` 会为任意 runtime 生成 Letta-shaped block system，并带 sleep-time hooks、safety rules 和 citation wiring。

## Exercises / 练习

1. 增加一个 `block_summarize` 工具，在 `near_limit` 返回 true 时用模型生成的 summary 替换 block value。哪个 trigger threshold 能在 summarization calls 和 block overflow 之间取得最好平衡？
2. 对 archival 做 sleep-time dedup：两条 records 的 text token overlap 超过 90% 时合并成一条。只在 sleep pass 做，不要在 critical path 做。
3. 给 blocks 加 version。每次写入都记录 old value 和 diff。暴露 `block_history(label)`，帮助 operator 调试 “why did the agent forget X”。
4. 把 sleep-time agents 当成不可信 writers。当它们触碰 Persona 或 Safety block 时，要求第二个 Agent review 后才能 commit。
5. 把示例移植到 Letta API（`letta_v1_agent`）。block schema 有什么变化？native reasoning 如何改变 trace shape？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Memory block | “Editable prompt section” | core memory 中 typed、persistent、LLM-editable segment |
| Human block | “User memory” | pinned 在 core 中的用户事实 |
| Persona block | “Agent identity” | pinned 在 core 中的自我概念、语气、约束 |
| Sleep-time compute | “Async memory work” | 第二个 Agent 在 critical path 外做 consolidation |
| Core / Recall / Archival | “Tiers” | 三层记忆拆分：always-visible / conversation / external |
| Block limit | “Cap” | 每个 block 的字符上限，触发 summarization |
| Native reasoning | “Thinking channel” | provider-level reasoning output，而不是 prompt-level `Thought:` |
| Learned context | “Sleep output” | sleep-time agent 写入 shared blocks 的事实 |

## Further Reading / 延伸阅读

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — block pattern
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — async consolidation
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — native reasoning rewrite
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — 起点
