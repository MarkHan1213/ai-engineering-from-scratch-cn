# Society of Mind and Multi-Agent Debate / Society of Mind 与多 Agent 辩论

> Minsky 1986 年的前提是：智能是专家社会。这个想法每十年都会被重新发现。2023 年 Du et al. 把它变成具体算法：多个 LLM instance 提出答案，阅读彼此答案，批判并更新。经过 N 轮，它们收敛到一个共识，在六个 reasoning 与 factuality 任务上超过 zero-shot CoT 和 reflection。两个发现最重要：**多个 Agent** 和 **多轮交流** 都各自贡献收益。社会优于单 Agent 独白；多轮交换优于一次性投票。

**类型：** 学习 + 构建
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 04（Primitive Model）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 self-consistency、reflection 与 multi-agent debate 的机制差异
- 解释 Du et al. 2023 算法中 agent count 和 round count 两个独立旋钮
- 识别 heterogeneous debate、sycophancy cascade、topic drift 与 compute blowup
- 为一个任务配置合理的 Agent 数、轮数、异质性和 adversarial role

## The Problem / 问题

Self-consistency 是最便宜的 reasoning 提升：让一个模型采样多次，然后取多数答案。它有效，但很快饱和。你可以把样本数翻倍，却看不到有意义的额外提升。

Debate 打破这种饱和。不是从一个模型独立采样 N 次，而是让 N 个 Agent 阅读彼此推理并修订。样本之间的相关性下降（不再是 i.i.d.），收敛点经常能修正 i.i.d. voting 自信错误的答案。

## The Concept / 概念

### The Du et al. 2023 algorithm / Du et al. 2023 算法

来自 arXiv:2305.14325（ICML 2024）：

1. N 个 Agent 分别为问题生成初始答案。
2. 对每一轮 r = 2..R：每个 Agent 看到其他 Agent 在 r-1 轮的答案，并被要求“considering these, give your updated answer.”
3. R 轮后，对最终答案做 majority vote。

论文在 MMLU、GSM8K、biographies、MATH 和 factuality benchmarks 上测试。debate 稳定超过 CoT 和 Self-Reflection。

### Two independent knobs / 两个独立旋钮

同一篇论文的 ablation 显示：

- **Agent count alone**（1 轮，N 个 Agent majority vote）在多数任务上超过单 Agent，但会 plateau。
- **Round count alone**（1 个 Agent 看自己的 prior reasoning）几乎没帮助，这也是 reflection 的已知弱点。
- **Both together** 才产生大幅提升。多个 Agent 之间的多轮交换是收益来源。

### Why it works / 为什么有效

两个机制：

1. **Exposure to disagreement.** 当 Agent 看见另一个 Agent 用不同推理链得到不同结论，它必须要么辩护，要么更新。不管哪种，r+1 轮上下文都比 r 轮更丰富。
2. **Correlated error reduction.** self-consistency 中所有样本来自同一个模型，错误相关；平均后可能得到自信错误。不同模型或不同 seed 能去相关。不同 *debated views* 进一步去相关。

### Heterogeneous debate / 异质辩论

A-HMAD 等后续工作让 *不同 base model* 扮演不同 Agent。Llama + Claude + GPT 辩论能降低 monoculture collapse（Lesson 26），因为一个模型族的相关错误不会被其他模型族共享。

代价：弱模型参与辩论可能把共识拖向自己的错误答案（见 "Should we be going MAD?", arXiv:2311.17371）。

### NLSOM — the 129-agent extension / NLSOM：129 Agent 扩展

Zhuge et al.（"Mindstorms in Natural Language-Based Societies of Mind," arXiv:2305.17066）把这个想法扩展到 129 个成员的 society。结果显示：规模上来后会涌现 specialization 和 self-organization，并在 visual question answering 等任务上超过单 Agent。

### Failure modes / 失败模式

- **Sycophancy cascade.** 所有 Agent 都服从最自信的 Agent。辩论塌缩成最大声的声音。给 adversarial roles 做 prompt（“一个 Agent 必须论证反方”）有帮助。
- **Topic drift.** 多轮辩论会从原始问题漂走。缓解：每轮重新注入问题。
- **Compute blowup.** N 个 Agent × R 轮 = N·R 次 LLM call，并且每次上下文都增长。5 Agent、5 轮辩论是 25 次调用，且上下文越来越长。单题成本可能超过一次 CoT 调用的 10 倍。

## Build It / 动手构建

`code/main.py` 在一个数学问题上运行 3-agent × 3-round debate。每个 Agent 初始答案不同（可能错误）。Agent 是脚本化的：它们按脚本化 confidence 对邻居答案做加权平均来“更新”。收敛过程能在逐轮日志里看见。

demo 展示两个关键效果：

- 一轮交换会让 Agent 更接近正确答案。
- 第 2 轮之后的额外轮数收益递减（对应 Du et al. 的 plateau）。

运行：

```
python3 code/main.py
```

## Use It / 应用它

`outputs/skill-debate-configurator.md` 为新任务配置辩论：Agent 数、轮数、异质性（同一模型 vs 混合模型）、角色分配（对称 vs 一个 adversarial）。它还会在运行前估算 token 成本。

## Ship It / 交付它

如果要上线 debate：

- **Cap rounds at 3.** Du et al. 显示 3 轮已经捕获大部分收益。更多轮数通常只是成本。
- **Cap agents at 5.** 超过 5 后，上下文膨胀和成本占主导。
- **Heterogeneous by default.** 池子里至少两个不同 base model。
- **Adversarial slot.** 一个 Agent 被提示始终提出反对意见。打破 sycophancy。
- **Log every round.** 隐藏中间轮次的 debate 系统无法调试和审计。

## Exercises / 练习

1. 运行 `code/main.py`，把轮数设为 5，观察收益递减。第几轮之后额外收敛停止？
2. 增加第四个 adversarial role：始终不同意当前多数。它破坏还是改善收敛？
3. 打印每轮 agreement score（多数答案占 Agent 比例）。什么时候达到 1.0？这是否等价于“正确”？
4. 阅读 Du et al. Section 4 ablations。用本课代码复现 “agents-only” vs “rounds-only” vs “both”。
5. 阅读 "Should we be going MAD?"（arXiv:2311.17371），列出两种 round-robin 之外的 debate variants，例如 judge-led、chain-of-debate、adversarial。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Society of Mind | “Minsky 的想法” | 智能来自交互专家；1986 年框架，现在通过 LLM debate 操作化。 |
| Multi-agent debate | “Agent 争论” | N 个 Agent 提案、互评、R 轮修订，最后 majority-vote。 |
| Consensus | “它们同意了” | 不是 epistemic truth，只是多数答案比例。可能自信错误。 |
| Rounds | “交换步骤” | 一轮 = 每个 Agent 读其他 Agent 输出并更新一次。 |
| Heterogeneous debate | “混合模型族” | 使用不同 base model 来去相关错误。 |
| Sycophancy cascade | “所有人同意最大声的人” | Agent 不管正确性，服从最自信 Agent 的失败模式。 |
| NLSOM | “129-Agent society” | Natural-language society of mind；Zhuge et al. 的规模化版本。 |
| Correlated error | “同一模型，同一 bug” | self-consistency 饱和的原因；不同视角辩论能降低相关性。 |

## Further Reading / 延伸阅读

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — 参考论文，ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — 129-agent NLSOM
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — benchmark debate variants
- [Debate project page](https://composable-models.github.io/llm_debate/) — Du et al. 的代码、demo 和 ablation 细节
