# Skill Libraries and Lifelong Learning (Voyager) / Skill Libraries 与终身学习（Voyager）

> Voyager (Wang et al., TMLR 2024) 把可执行代码视为 skill。Skills 有名称、可检索、可组合，并通过环境反馈 refinement。这是 Claude Agent SDK skills、skillkit 和 2026 skill-library pattern 的参考架构。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 说出 Voyager 的三个组件：automatic curriculum、skill library、iterative prompting，以及各自作用。
- 解释为什么 Voyager 把 action space 设为代码，而不是 primitive commands。
- 实现一个 stdlib skill library，包含 registration、retrieval、composition 和 failure-driven refinement。
- 把 Voyager 模式映射到 2026 年的 Claude Agent SDK skills 和 skillkit 生态。

## The Problem / 问题

每个 session 都从零重建能力的 Agent 会犯三类错误：

1. **Waste tokens。** 每个任务都重新诱导同样推理。
2. **Lose progress。** session A 学到的纠正无法迁移到 session B。
3. **Fail on long-horizon composition。** 复杂任务需要 capability hierarchies；one-shot prompts 表达不了。

Voyager 的答案是：把每个可复用能力作为一段命名代码存入 library，按 similarity 检索，可和其他 skills 组合，并通过执行反馈 refine。

## The Concept / 概念

### Three components / 三个组件

Voyager (arXiv:2305.16291) 围绕三个组件组织 Agent：

1. **Automatic curriculum。** curiosity-driven proposer 基于 Agent 当前 skill set 和 environment state 选择下一个任务。探索是 bottom-up 的。
2. **Skill library。** 每个 skill 是可执行代码。任务成功后新增 skill。按 query-to-description similarity 检索 skills。
3. **Iterative prompting mechanism。** 失败时，Agent 接收 execution errors、environment feedback 和 self-verification output，然后 refine skill。

Minecraft 评测（Wang et al., 2024）：相对 baselines，unique items 多 3.3 倍，stone tools 快 8.5 倍，iron tools 快 6.4 倍，map traversal 长 2.3 倍。数字是 Minecraft-specific，但模式可以迁移。

### Action space = code / 动作空间就是代码

多数 Agent 产出 primitive commands。Voyager 产出 JavaScript functions。一个 skill 是：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

它由 sub-skills 组合而来。以 description 和 embedding 为 key 存储。检索回来的是 program，而不是 prompt。

这就是 2026 年 Claude Agent SDK skill：一个命名、可检索的 code + instructions chunk，Agent 按需加载。

### Skill retrieval / Skill 检索

新任务 “make a diamond pickaxe”。Agent：

1. 嵌入任务描述。
2. 在 skill library 里查询 top-k similar skills。
3. 检索 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 用已检索 primitives + 新逻辑组合出新 skill。

这是 MCP resources（Phase 13）和 Agent SDK skills 实现的模式：在 knowledge/code surface 上做 retrieval，并按当前任务 scope 加载。

### Iterative refinement / 迭代 refinement

Voyager 的 feedback loop：

1. Agent 写一个 skill。
2. skill 在环境中运行。
3. 返回三类信号之一：`success`、`error`（带 stack trace）、`self-verification failure`。
4. Agent 用该信号作为上下文改写 skill。
5. 循环直到成功或达到 max rounds。

这是 Self-Refine（Lesson 05）应用在代码生成上，并由环境提供 verification。CRITIC（Lesson 05）是同一个模式，只是 verifier 是外部工具。

### Curriculum and exploration / Curriculum 与探索

Voyager 的 curriculum module 会基于 Agent 已拥有和未完成的能力，提出 “build a shelter near the lake” 这类任务。proposer 使用 environment state + skill inventory，选择略高于当前能力的任务，也就是探索的最佳难度区间。

对生产 Agent 来说，这对应一个 “what's missing” operator：给定当前 skill library 和领域，我们还缺哪些 skills？团队通常把它实现为人工 curriculum review。

### Where this pattern goes wrong / 这个模式在哪里会出错

- **Skill library rot。** 同一个 skill 以略有不同的 description 被添加 10 次。写入时做 dedup；检索只返回一个 canonical skill。
- **Composed-skill drift。** parent skill 依赖的 child 被 refine 了。给 skills 做 version；pin 到 v1 的 parent 不应自动拿到 v3。
- **Retrieval quality。** 当 library 超过几百个 skill 后，只按 description 做 vector retrieval 会退化。补充 tag filters 和硬约束（例如 “only skills with `category=tooling`”）。

## Build It / 动手构建

`code/main.py` 实现一个 stdlib skill library：

- `Skill`：name、description、code（字符串）、version、tags、dependencies。
- `SkillLibrary`：register、search（token overlap）、compose（dependency topological sort）、refine（更新时 version bump）。
- 一个脚本化 Agent：注册三个 primitive skills，组合第四个，遇到失败，然后 refine。

运行：

```
python3 code/main.py
```

trace 会展示 library writes、retrieval、composition、failed execution 和 v2 refinement，也就是 Voyager loop 端到端。

## Use It / 应用它

- **Claude Agent SDK skills**（Anthropic）：2026 年参考形态。每个 skill 有 description、code 和 instructions；在 Agent session 中按需加载。
- **skillkit**（npm: skillkit）：用于 32+ AI coding agents 的 cross-agent skill management。
- **Custom skill libraries**：领域特定，例如 data agents 的 SQL skills、infra agents 的 Terraform skills。Voyager 模式可以缩小使用。
- **OpenAI Agents SDK `tools`**：低配版本；每个 tool 都是 lightweight skill。

## Ship It / 交付它

`outputs/skill-skill-library.md` 会为任意 target runtime 生成 Voyager-shaped skill library，包含 registration、retrieval、versioning 和 refinement。

## Exercises / 练习

1. 给 `compose()` 增加 dependency-cycle detector。如果 skill A 依赖 B，而 B 又依赖 A，会发生什么？error 还是 warning？
2. 实现 per-skill version pinning。当 parent skill 组合 child `crafting@1` 时，`crafting@2` refinement 不应静默升级 parent。
3. 用 sentence-transformers embeddings（或 stdlib BM25 实现）替换 token-overlap retrieval。在 50-skill toy library 上测 retrieval@5。
4. 增加一个 “curriculum” agent：给定当前 library 和 domain description，提出 5 个缺失 skills。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK skill docs。把 toy library 移植到 SDK 的 skill schema。discoverability 有什么变化？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Skill | “Reusable capability” | 带 description 的命名代码块，可按 similarity 检索 |
| Skill library | “Agent memory of how-to” | skills 的持久 store，可搜索、可组合 |
| Curriculum | “Task proposer” | 由当前 capability gap 驱动的 bottom-up 目标生成器 |
| Composition | “Skill DAG” | skills 调用 skills；执行前拓扑排序 |
| Iterative refinement | “Self-correcting loop” | env feedback + errors + self-verification 折回下一版本 |
| Action-space-as-code | “Programmatic actions” | 发出 functions，而不是 primitive commands，以表达长时行为 |
| Dedup on write | “Skill collapse” | 近似重复 descriptions 合并到一个 canonical skill |

## Further Reading / 延伸阅读

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — 原始 skill-library paper
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — skills 的 2026 产品化形态
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — skills and subagents in practice
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — Voyager 底层 refinement loop
