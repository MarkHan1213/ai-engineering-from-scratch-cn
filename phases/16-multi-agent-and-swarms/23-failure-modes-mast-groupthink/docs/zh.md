# Failure Modes — MAST, Groupthink, Monoculture, Cascading Errors / 失败模式：MAST、群体思维、单一文化与级联错误

> 2026 年参考 taxonomy 是 **MAST**（Cemri et al., NeurIPS 2025, arXiv:2503.13657），来自 7 个 state-of-the-art open-source MAS 的 1642 条执行 trace，显示 **41–86.7% failure rate**。三个根类别：**Specification Problems**（41.77%）——角色歧义、任务定义不清；**Coordination Failures**（36.94%）——通信 breakdown、state desync；**Verification Gaps**（21.30%）——缺少 validation、质量检查缺失。**Groupthink** family（arXiv:2508.05687）补充：monoculture collapse（同 base model → correlated failures）、conformity bias（Agent 互相强化错误）、deficient theory of mind、mixed-motive dynamics、cascading reliability failures。级联例子：retry storm 中 payment failure 触发 order retries，order retries 触发 inventory retries，几秒内把 inventory service 压到 10x load，需要 circuit breakers。Memory poisoning：一个 Agent 的 hallucination 进入 shared memory，下游 Agent 当成事实；准确率慢慢衰减，根因诊断痛苦。**STRATUS**（NeurIPS 2025）报告，通过 specialized detection / diagnosis / validation agents，mitigation-success 提升 1.5x。本课把失败模式当成一等工程目标。

**类型：** 学习
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 13（Shared Memory）, 第 16 阶段 · 14（Consensus and BFT）, 第 16 阶段 · 15（Voting and Debate Topology）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 使用 MAST 三大类定位多 Agent 失败：specification、coordination、verification
- 识别 groupthink family：monoculture、conformity、deficient ToM、mixed-motive、cascading
- 设计 retry storm、memory poisoning、slow failure 的检测和缓解
- 建立 failure-mode audit、STRATUS trio、failure budget 等生产纪律

## The Problem / 问题

多 Agent 系统在真实任务上失败率高达 41-86.7%（Cemri et al. 2025 在 7 个开源 MAS 上测得）。这不是“再加几个 Agent”就能调试的问题。失败有结构性原因。MAST taxonomy 给出了类别。本课把每个类别映射到具体 detection、diagnosis 和 mitigation pattern，让这些数字不再像随机噪声。

2026 年生产实践是把 failure modes 当作设计输入。你的架构还不算“够好”，直到你能指着每个 MAST 类别，说出部署了什么缓解。

## The Concept / 概念

### MAST categories / MAST 分类

**Specification Problems (41.77% of failures).** Agent 的任务定义不够紧。例子：

- role ambiguity：两个 Agent 都以为自己是 reviewer。
- task underspecified：“summarize this”，但用户想要特定角度。
- success criteria implicit：Agent 无法判断自己是否成功。

缓解：

- 写显式 role contracts。每个 Agent prompt 说明自己做什么，也说明自己不做什么。
- 每个任务定义 acceptance tests。Agent 开始前先定义 “done looks like X”。
- pre-flight spec check：独立 Agent 在 dispatch 前 review task definition。

**Coordination Failures (36.94%).** 通信或状态 breakdown。

例子：

- 两个 Agent 无同步地更新 shared state。
- Agent 间消息丢失（queue failure、timeout）。
- state drift：Agent A 认为任务完成，Agent B 仍在执行。

缓解：

- 带 optimistic concurrency 的 versioned shared state。
- 关键消息必须 explicit acknowledgment（retry until acked）。
- 周期性 state-sync checkpoints，尽早发现 drift。

**Verification Gaps (21.30%).** 输出缺少独立检查。

例子：

- 一个 Agent 声称成功，没人验证。
- 链上每个 Agent 都信任前一个输出。
- emergent composed behavior 没有测试覆盖。

缓解：

- 独立 verifier Agent（Lesson 13）。只读、拥有独立 source access。
- 显式 handoff contract：“A 的输出必须通过 checker C，B 才能开始。”
- outcome logging 方便事后分析。

### Groupthink family (arXiv:2508.05687) / Groupthink 失败族

Agent 同质化或互相模仿时会出现五类相关失败：

**Monoculture collapse.** 同 base model 或训练数据 → correlated errors。三个 Agent 共享一个 LLM 时，也共享它的 hallucinations。

**Conformity bias.** Agent 向最大声或最自信的 peer 靠拢，即使 peer 错了。

**Deficient ToM.** Agent 无法建模彼此 beliefs；coordination 崩溃（Lesson 18）。

**Mixed-motive dynamics.** 部分对齐的 Agent 会走向折中中点，结果谁都不满意。

**Cascading reliability failures.** 一个组件的错误模式触发依赖组件的错误模式。

### Cascading example — the retry storm / 级联例子：重试风暴

2026 年经典 incident pattern：

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

修复是经典做法：**circuit breakers**。当 downstream error rate 超过阈值，短路返回缓存或默认结果。再加每个 request 的 capped retry budgets。

circuit breaker 是少数几乎可以直接从分布式系统借来的多 Agent 失败缓解。

### Memory poisoning (revisited) / Memory poisoning 回顾

来自 Lesson 13：一个 Agent 的 hallucination 变成 shared-memory fact；下游 Agent 在被污染事实上推理。在 MAST 术语里，这是 shared-memory layer 的 verification gap。

症状是准确率逐渐衰减。你不会得到 crash，只会得到难以追根溯源的慢性 drift。

缓解：append-only log、provenance、unwritable verifier。Lesson 13 已覆盖。

### STRATUS — specialized agents for failure detection / STRATUS：专门化失败检测 Agent

STRATUS（NeurIPS 2025）报告，部署下面三类角色时，mitigation-success 提升 1.5x：

- **Detection agent.** 监控 symptom patterns（high disagreement、retry spikes、accuracy drift）。
- **Diagnosis agent.** 给定 symptoms，从 MAST taxonomy 推断可能 root cause。
- **Validation agent.** mitigation 应用后，检查 symptoms 是否清除。

这是 SRE-style incident response，应用到 Agent systems。三个角色都可以是 specialized prompts 的 LLM Agent。

### The failure-mode audit / 失败模式审计

2026 年最佳实践是 annual（或每次 major release）failure-mode audit：

1. **Trace sample.** 收集约 1000 条真实 execution traces。
2. **Categorize.** 把每条 trace 的失败映射到 MAST + Groupthink categories。
3. **Compute failure-by-category rate.** 哪些类别主导你的系统？
4. **Rank mitigations.** 哪个 fix 能消除最多失败？
5. **Pick 2-3 mitigations.** 实施，下个季度重新审计。

纪律比具体选择更重要。没有 audit，失败会混成噪声，永远得不到系统处理。

### When systems fail silently / 静默失败

最危险的失败类别是 silent correctness failure。显式失败（crash、exception、alert）可以监控。产出 plausible-but-wrong outputs 的系统不能靠 exception logs 发现。这就是为什么 verification gaps 虽然按数量只有 21.30%，但单次成本最高。

投资方向：

- sample-based human review。
- golden-dataset regression tests。
- 重要输出上的 cross-agent cross-checking。

### Failure vs slow failure / 快失败与慢失败

有些失败立即发生，有些失败很慢。立即失败（timeout、schema mismatch、auth error）便宜易检测。慢失败（memory poisoning、monoculture drift、role ambiguity）检测和预防都贵。

2026 年工程动作：instrument slow-failure proxies，在 drift 变成可见错误前抓住它。agreement rate、retry rate、output-length distribution、连续 Agent 版本之间的 edit-distance 都是有用 proxy。

## Build It / 动手构建

`code/main.py` 实现：

- `FailureTaxonomy` — 把模拟 incident 分类到 MAST + Groupthink。
- `CircuitBreaker` — 经典 pattern；error rate 超阈值后 open。
- `RetryStormSimulator` — 展示级联失败；可切换 circuit breaker on / off。
- `DetectionAgent` — 脚本化 STRATUS-style symptom matcher。

运行：

```
python3 code/main.py
```

预期输出：

- 没有 circuit breaker：retry storm 让 inventory errors 暴涨（模拟）。
- 有 circuit breaker：错误被阈值封顶，并返回 degraded-mode responses。
- detection agent 标记 pattern 并命名 MAST category。

## Use It / 应用它

`outputs/skill-mast-auditor.md` 对多 Agent 系统运行 MAST-style failure-mode audit。输入 traces，输出 categorization 和 mitigation ranking。

## Ship It / 交付它

生产 failure-mode discipline：

- **MAST audit per quarter.** 不是 annual。系统增长时类别会变化。
- **Circuit breakers everywhere.** 每个 outbound call 到 dependent service 都要有。默认 open threshold 5-10% error rate。
- **Golden datasets.** 小而高质量、人工审过。每周 regression-test。
- **STRATUS trio.** Detection + Diagnosis + Validation agents 监控生产。先从 detection agent 开始；symptoms 噪声大时再加 diagnosis。
- **Failure budget.** 为每个 category 明确 SLO。超出 budget 触发 stop-shipping conversation。

## Exercises / 练习

1. 运行 `code/main.py`。确认 circuit breaker 限制 retry storm。改变 failure threshold，观察取舍。
2. 实现一个 **slow-failure proxy**：3 个并行 Agent 的 agreement rate。当它急剧下降时触发 alert。通过逐步相关化 Agent 输出模拟 monoculture drift。
3. 阅读 Cemri et al.（arXiv:2503.13657）。选择他们的 7 个 MAS 系统之一，映射其 top 3 failure categories。与 MAST 预测相比如何？
4. 阅读 Groupthink paper（arXiv:2508.05687）。识别五种 pattern 中生产里最难检测的一种，并提出 proxy metric。
5. 为你熟悉的某个多 Agent 系统设计 STRATUS-style detection-diagnosis-validation trio。detection 监控什么 symptom？diagnosis 推荐什么 mitigation？validation 如何确认有效？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MAST | “The 2026 taxonomy” | Cemri 2025；3 个根类别 + 14 个失败 subtype。 |
| Specification Problem | “Role ambiguity” | 任务或角色定义不足；Agent 不知道该做什么。 |
| Coordination Failure | “State drift” | Agent 之间通信或同步 breakdown。 |
| Verification Gap | “No one checked” | 输出没有 independent validation 就被接受。 |
| Groupthink family | “Homogeneity failures” | monoculture、conformity、deficient ToM、mixed-motive、cascading。 |
| Monoculture collapse | “Same model, same hallucinations” | base model 或训练数据共享导致 correlated errors。 |
| Retry storm | “Cascading error amplification” | 一个失败触发 retries，retries 又放大 downstream load。 |
| Circuit breaker | “Fail fast on error rate” | error rate 超阈值时 open，用 default 短路。 |
| STRATUS | “Incident response trio” | Detection + diagnosis + validation agents。mitigation success 提升 1.5x。 |
| Memory poisoning | “Hallucinations propagate” | shared-memory fact 被污染，下游 Agent 在 poison 上推理。 |

## Further Reading / 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy，NeurIPS 2025
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) — monoculture、conformity 和五类 taxonomy
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) — NeurIPS 2025 proceedings entry（detection + diagnosis + validation）
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — circuit breaker 的经典参考
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 生产 failure-mode notes
