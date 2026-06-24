# Hierarchical Architecture and Its Failure Mode / 层级架构及其失败模式

> Hierarchical 就是嵌套 supervisor：manager Agent 管 sub-manager，sub-manager 再管 worker。CrewAI `Process.hierarchical` 是教科书版本：`manager_llm` 动态委派任务并验证输出。LangGraph 等价写法是 `create_supervisor(create_supervisor(...))`。当任务本身像组织结构图时，它是自然模式；但它也是最容易塌成管理循环的模式：manager Agent 分配不当、误读下级输出，或无法达成共识。很多时候 sequential 反而更好。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 05（Supervisor Pattern）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 hierarchical architecture 如何由嵌套 supervisor 组成
- 判断任务是否真的需要层级，而不是线性流程伪装成树
- 识别 task assignment error、output misinterpretation、consensus loops 三类层级失败
- 设计 depth cap、reconciliation budget、provenance chain、decomposition drift alert 等防护

## The Problem / 问题

一旦 supervisor pattern 想通，自然下一步就是：“如果 worker 自己也是 supervisor 呢？”团队有子团队，公司有多层部门。层级架构正好映射这种结构。

问题是：LLM manager 不是人类 manager。人类 manager 对下属知道什么有稳定先验。LLM manager 每一轮都根据上下文重新推理组织结构。上下文发生微小 drift，整棵树就会错误分配工作。

## The Concept / 概念

### The shape / 形状

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

每个内部节点负责计划、委派、综合。只有叶子节点真正干活。

### Where it shines / 适合的场景

- **Clear org mapping.** 如果真实任务就是部门式的，例如“legal 审文档、finance 审文档、engineering 审文档，然后汇总给 exec”，层级是显式的。
- **Local summarization.** 每个 sub-manager 先综合本团队输出，再交给顶层 manager。顶层看到三个 sub-manager 摘要，而不是十五个 worker 输出。

### Where it breaks / 会坏在哪里

2026 年 post-mortem 反复出现三种失败：

1. **Task assignment error.** manager 读到目标后 hallucinate 出一个拆解，并把任务交给错误 sub-manager。sub-manager 会忠实处理被分到的任务，所以错误到顶层 synthesis 才暴露，而这时已经离本该发现问题的地方隔了一层。
2. **Output misinterpretation.** sub-manager 返回“unable to verify claim X”。顶层 manager 总结成“claim X not confirmed”。含义每一层都会漂移。
3. **Consensus loops.** 两个 sub-manager 不同意；顶层让它们 reconcile；它们再向下委派；worker 重跑；sub-manager 返回略不同的答案；循环。CrewAI 的 `Process.hierarchical` 用 step limit 防护，但 limit 本身又变成一个超参数。

### The deciding question / 决策问题

Sequential（线性 pipeline）还是 hierarchical：你的任务真的有独立子团队，还是一个线性流程假装成树？如果是后者，用 sequential。如果是前者，可以用 hierarchical，但必须预算明确 reconciliation rules。

### CrewAI's implementation / CrewAI 实现

`Process.hierarchical` 把 manager LLM 接到 specialist crews 之上。manager：

- 接收顶层任务，
- 把子任务分配给 crews，
- 评估 crew 输出，
- 决定接受、重新委派或迭代。

文档：https://docs.crewai.com/en/introduction（Core Concepts 里的 "Hierarchical Process"）。

### LangGraph's implementation / LangGraph 实现

LangGraph 使用嵌套 `create_supervisor`。内部 supervisor 有自己的 graph；外部 supervisor 把内部 graph 当作 opaque node。它比 CrewAI 更利于调试（可以分别 step through 每个 graph），但更难表达动态重塑树结构。

参考：https://reference.langchain.com/python/langgraph-supervisor。

## Build It / 动手构建

`code/main.py` 运行一个 3 层 hierarchy：

- top manager：把任务拆成 "engineering" 和 "legal" 分支，
- engineering sub-manager：拆成 "frontend" 和 "backend" workers，
- legal sub-manager：一个 worker。

demo 对比 happy path 与 **perturbed path**：perturbed path 里，top manager 把 "legal" 错标成 "finance"，于是错误级联。sub-manager 忠实做 finance 工作，top synthesizer 报告 finance findings，原始 legal 问题无人回答。

运行：

```
python3 code/main.py
```

输出会并排展示两条路径里的 “what was asked” 与 “what was delivered”。

## Use It / 应用它

`outputs/skill-hierarchy-fitness.md` 用来评估一个任务应该使用 hierarchical、sequential 还是 flat supervisor。输入包括：task description、org structure、reconciliation budget。输出是模式推荐，以及需要防护的具体失败模式。

## Ship It / 交付它

如果要上线 hierarchical：

- **Cap tree depth at 2.** 三层已经会把大多数错误藏出 observability。
- **Explicit reconciliation budget.** 顶层 manager 必须在最大轮数前 commit。通常是 2。
- **Provenance on every synthesis.** 每个节点的摘要都必须引用产生它的叶子输出。
- **Alert on decomposition drift.** 记录 manager 每一步 decomposition；和用户 query 做 diff。如果 decomposition 不再覆盖 query，发 alert。

## Exercises / 练习

1. 运行 `code/main.py`，比较 happy vs perturbed。经过多少层 manager hand-off 后，顶层输出完全偏离用户问题？
2. 增加第三层（top → sub → sub-sub → worker）。随着深度增长，perturbed path 有多大概率自我纠正 vs 完全发散？
3. 在每个 sub-manager 下增加一个 “canary” worker，始终原样询问最初用户问题。用 canary answer 检测 decomposition drift。当 canary 与 synthesized answer 不一致时，manager 应该如何反应？
4. 阅读 CrewAI 的 `Process.hierarchical` 文档。识别 CrewAI 应用的一条具体 guardrail（step limit、manager_llm constraint），并说明它针对什么失败模式。
5. 比较 nested LangGraph supervisors 和 CrewAI hierarchical。哪一种更便宜地检测 reconciliation loops？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Hierarchical | “Org chart pattern” | supervisor 之上还有 supervisor；只有叶子节点做实际工作。 |
| Manager LLM | “The boss” | 在内部节点负责拆解、分配和验证的 LLM。 |
| Decomposition drift | “老板跑偏了” | 顶层 manager 的拆分不再覆盖原始问题。 |
| Reconciliation loop | “无尽会议” | sub-manager 不同意，顶层重新委派，worker 重跑，直到预算耗尽。 |
| Depth-2 ceiling | “不要超过 2 层” | 经验 guardrail：3+ 层会让 observability 崩掉。 |
| Canary question | “每层的 ground truth” | 总是原样询问最初 query 的 worker，用来检测 drift。 |
| Provenance chain | “谁说了什么” | 从每个 synthesis 回溯到产生它的叶子输出。 |

## Further Reading / 延伸阅读

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) — 带 manager LLM 的教科书 hierarchical
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — 通过 `create_supervisor` 嵌套 supervisor
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 为什么 Anthropic 明确选择 flat supervisor 而不是 hierarchical
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy；coordination failures 章节记录 decomposition drift
