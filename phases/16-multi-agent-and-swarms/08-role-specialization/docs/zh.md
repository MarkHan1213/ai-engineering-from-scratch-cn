# Role Specialization — Planner, Critic, Executor, Verifier / 角色专门化：Planner、Critic、Executor、Verifier

> 2026 年最常见的多 Agent 拆解：一个 Agent 计划，一个执行，一个批判或验证。MetaGPT（arXiv:2308.00352）把它形式化为写进 role prompts 的 SOP：Product Manager、Architect、Project Manager、Engineer、QA Engineer，并提出 `Code = SOP(Team)`。ChatDev（arXiv:2307.07924）通过 “chat chain” 串起 designer、programmer、reviewer、tester，并加入 “communicative dehallucination”（Agent 显式请求缺失细节）。verifier 是承重角色：Cemri et al.（MAST, arXiv:2503.13657）显示每个多 Agent 失败都能追溯到缺失或破损的 verification。PwC 报告在 CrewAI 中用结构化验证循环把准确率提升 7 倍（10% → 70%）。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 04（Primitive Model）, 第 16 阶段 · 05（Supervisor）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 planner、executor、critic、verifier 四个经典角色及其输入/输出
- 解释 MetaGPT 的 SOP pattern 和 ChatDev 的 communicative dehallucination
- 明确 critic 与 verifier 的边界：主观质量判断 vs 确定性 pass/fail
- 设计至少包含一个 deterministic verifier 的多 Agent pipeline

## The Problem / 问题

泛化的多 Agent 系统会产生泛化的输出。三个 coder 在一个 group chat 里会写出三种差不多的平庸代码。你可以加更多 Agent、加更多轮次，仍然过不了质量线。

修复办法不是更多 Agent，而是 *不同* Agent。分配不同角色。给 critic planner 没有的工具。给 verifier 一个客观测试套件。这样系统内部有的是带证据的纠错，而不是并行猜测。

## The Concept / 概念

### The four canonical roles / 四个经典角色

**Planner.** 读取目标，生成步骤列表或 spec。工具：knowledge retrieval、docs。输出：structured plan。

**Executor.** 一次读取一个 plan step，产出 artifact。工具：真正干活的工具（code compiler、shell、API client）。输出：artifact。

**Critic.** 对照 planner 意图读取 executor 输出。工具：artifact 的只读访问、static analysis。输出：accept/reject 和理由。

**Verifier.** 读取 artifact 并运行确定性检查。工具：test runner、type checker、schema validator。输出：带证据的 pass/fail。

Critic 是主观的、有观点的，通常基于 LLM。Verifier 是客观的、确定性的，通常基于代码。它们不是同一个角色。

### MetaGPT's SOP pattern / MetaGPT 的 SOP 模式

MetaGPT（arXiv:2308.00352）把软件工程 SOP 编码为 role prompts：

- **Product Manager** 写 PRD。
- **Architect** 产出系统设计。
- **Project Manager** 拆任务。
- **Engineer** 实现。
- **QA Engineer** 跑测试。

每个角色都有严格输入/输出 schema。role prompt 说明该角色 *是什么*，以及 *必须产出什么*。`Code = SOP(Team)` 的含义是：确定性的 SOP 把 LLM 团队变成可预测 pipeline。

### ChatDev's communicative dehallucination / ChatDev 的 communicative dehallucination

ChatDev 增加了一个关键动作：当 executor 需要计划里没有的具体细节时，它会在继续前显式询问 designer。这避免了 LLM 的典型失败：看似合理地编造细节。

实现方式：role prompt 包含“当你需要未给出的具体信息时，先按名字询问相关角色，再产出输出”。

### Why verifier matters most / 为什么 verifier 最重要

Cemri et al.（MAST）追踪了 1642 个多 Agent 执行失败。21.3% 是 verification gaps：系统交付了没人检查的答案。剩下的 79% 也常常能追溯到“本该存在的检查失败、静默失败或从未运行”。Verification 是承重角色。

PwC 报告（CrewAI deployments, 2025）显示，加入结构化 validation loop 后，准确率从 10% 到 70%。单个角色带来 7 倍提升。

### Critic vs verifier / Critic 与 verifier

- critic 是 LLM，对 artifact 做质量 review。主观。能被看似合理的文字骗过。
- verifier 是在 artifact 上运行的确定性程序。客观。给出带证据的 pass/fail。

两者都要用。critic 抓 verifier 难以表达的 taste issues。verifier 抓 critic 看不见的 runtime bug。

### The anti-pattern / 反模式

系统里每个角色都是 LLM，每个角色输出都是 “looks good to me”。这是经典 MAST failure mode。至少加一个 verifier，它的 pass/fail 必须由代码而不是 LLM 决定。

### Framework mappings / 框架映射

- **CrewAI** — `Agent(role, goal, backstory)` 是教科书式 specialization surface。
- **LangGraph** — node 可以拥有 specialized prompt；edge 强制 pipeline。
- **AutoGen** — GroupChat 中有 role-specific ConversableAgents，通常用一词命名。
- **OpenAI Agents SDK** — role-specialized Agents 之间通过 handoff tools 转移。

## Build It / 动手构建

`code/main.py` 实现一个构建简单 Python 函数的 4-role pipeline：

- **Planner** 产出 spec。
- **Executor** 生成 code string。
- **Critic**（LLM-simulated）标出明显问题。
- **Verifier** 在 sandbox（`exec`）中对生成代码跑 test case。

demo 运行两次：一次 executor 生成正确代码（critic + verifier 都通过），一次 executor 生成 off-spec 代码（critic 因为看起来合理而漏掉 bug，verifier 因为测试失败而抓住）。

运行：

```
python3 code/main.py
```

## Use It / 应用它

`outputs/skill-role-designer.md` 接收一个任务，产出 role roster（3-5 个角色）、每个角色的输入/输出 schema，以及 verifier check。把 Agent 接入框架前先用它。

## Ship It / 交付它

Checklist：

- **At least one deterministic verifier.** 永远不要全是 LLM。
- **Explicit I/O schema per role.** planner 返回 spec，不是散文；executor 读取该 schema。
- **Communicative dehallucination.** executor 缺信息时必须问 planner；不要编造。
- **Critic/verifier ordering.** critic 先跑（便宜，抓设计问题），verifier 后跑（慢，抓 bug）。
- **Loop budget.** critic-executor 修订最多 2 轮，之后升级给人类。

## Exercises / 练习

1. 运行 `code/main.py`，观察 verifier 如何抓到 critic 漏掉的 bug。增加一个 static-analysis check（统计 `return` 出现次数）作为额外 verifier。它能抓到 runtime test 抓不到的什么？
2. 增加第 5 个角色："requirements analyst"，把用户愿望转换成 planner-ready spec。哪些 communicative dehallucination 请求应该向它上行？
3. 阅读 MetaGPT Section 3（"Agents"）。列出 MetaGPT 5 个角色的输入/输出 schema。
4. 阅读 ChatDev chat-chain diagram（arXiv:2307.07924 Figure 3）。找出 communicative dehallucination 在哪里打断了原本会无限循环的环。
5. PwC 的 7× accuracy gain 来自 verification loops。假设三个加入 verifier 也没帮助的任务：即确定性正确性检查不可能或成本过高的任务。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Role specialization | “不同 Agent 做不同工作” | 为 planner/executor/critic/verifier 等角色调优的不同 system prompt。 |
| SOP pattern | “编码标准操作流程” | MetaGPT 框架：每个角色严格 I/O schema，把团队变成 pipeline。 |
| Communicative dehallucination | “先问，不要编” | ChatDev 模式：executor 缺细节时询问 planner，而不是臆造。 |
| Critic | “LLM reviewer” | 主观、有观点的 reviewer。抓 taste issues。会被合理文字骗过。 |
| Verifier | “Deterministic check” | 基于代码的 pass/fail。test runner、type checker、schema validator。不会被说服。 |
| Verification gap | “没人检查” | MAST 失败的 21.3%。答案没有经过本可抓 bug 的检查就被交付。 |
| Revision loop | “Critic 打回” | critic 拒绝后触发 executor 带反馈重跑。需要预算。 |
| All-LLM anti-pattern | “Looks good to me” | 每个角色都是 LLM，没有确定性检查。经典 MAST 失败。 |

## Further Reading / 延伸阅读

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — SOP-as-role-prompt 参考论文
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — chat chain + communicative dehallucination
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy；verification gaps 占失败的 21.3%
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) — 生产 role specification surface
