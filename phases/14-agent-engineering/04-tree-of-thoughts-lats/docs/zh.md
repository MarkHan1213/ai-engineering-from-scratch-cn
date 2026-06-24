# Tree of Thoughts and LATS: Deliberate Search / Tree of Thoughts 与 LATS：有意搜索

> 单条 chain-of-thought trajectory 没有回溯空间。ToT (Yao et al., 2023) 把推理变成一棵树，并对每个节点做 self-evaluation。LATS (Zhou et al., 2024) 用 Monte Carlo Tree Search 统一 ToT、ReAct 和 Reflexion。Game of 24 从 4% (CoT) 到 74% (ToT)；LATS 在 HumanEval 上达到 92.7% pass@1。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 把 reasoning 表述为 search：节点是 “thoughts”，边是 “expansions”，value 是 “有多 promising”。
- 实现一个 stdlib ToT-style BFS tree search，并用 self-evaluation scoring 排序。
- 扩展成 toy LATS MCTS loop，包含 select / expand / simulate / backpropagate。
- 判断什么时候 search 值得付 token multiplier（Game of 24、code generation），什么时候单条 trajectory 已足够（简单 Q&A）。

## The Problem / 问题

Chain-of-thought 是线性行走。如果第一步错了，后面的每一步都在错误前提上继续。在 Game of 24（用四个数字通过 + - * / 组成 24）上，GPT-4 CoT 只有 4% accuracy。模型早早选错子表达式，之后无法恢复。

推理需要的是提出多个候选、评估候选、选择有希望的分支，并在遇到死路时回溯的能力。这就是 search。Tree of Thoughts 和 LATS 是两个标准表述。

## The Concept / 概念

### Tree of Thoughts (Yao et al., NeurIPS 2023) / Tree of Thoughts

每个节点都是一个连贯中间步骤（一个 “thought”）。每个节点可以展开为 K 个 child thoughts。LLM 用 scoring prompt 对每个节点自评。搜索算法探索整棵树，可以是 BFS、DFS 或 beam。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Self-evaluation 是承重件。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` numeric score，以及 candidates 之间投票。三者在 Game of 24 上都显著超过 CoT（GPT-4 从 4% 到 74%）。

### LATS (Zhou et al., ICML 2024) / LATS

LATS 在 MCTS 下统一 ToT、ReAct 和 Reflexion。LLM 扮演三个角色：

- **Policy**：提出候选 next actions（ReAct-style）。
- **Value function**：给 partial trajectory 打分（ToT-style self-eval）。
- **Self-reflector**：失败时写自然语言 reflection（Reflexion-style），并用它重新播种 future rollouts。

Environment feedback（observations）会混进 value function，因此 search 会被真实 tool results 引导，而不只是模型自评。论文当时结果：HumanEval pass@1 用 GPT-4 达到 92.7%（SOTA），WebShop 用 GPT-3.5 达到 75.9 average（接近 gradient-based fine-tuning）。

### MCTS, minimally / 最小 MCTS

每次迭代有四个阶段：

1. **Select**：用 UCT（upper confidence bound for trees）从 root 走到 leaf。
2. **Expand**：通过 policy 生成 K 个 children。
3. **Simulate**：从某个 child 用 policy rollout 到 leaf，并用 value function（或 environment reward）评分。
4. **Backpropagate**：把 visit counts 和 value estimates 沿路径向上更新。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是 exploitation，第二项是 exploration。`c` 需要按任务调。

### The cost reality / 成本现实

Search 会让 token 爆炸。Game of 24 上的 ToT 使用的 token 是 CoT 的 100-1000 倍。LATS 也类似。这不是免费能力；只应保留给：

- 单条 trajectory 明显不够的任务（Game of 24、复杂代码）。
- 正确性比 wall-clock 更重要的任务。
- 有便宜且可靠 value function 的任务（代码单元测试、数学明确目标）。

如果任务只有一个正确答案，而且 evaluator 噪声很大，search 往往会让事情更糟：它会找到一个 “得分高” 的错误答案。

### 2026 positioning / 2026 年定位

大多数生产 Agent 不跑 LATS。它们跑的是 ReAct 加 tool-grounded verification（CRITIC，Lesson 05）。Search 会出现在专门场景：

- 用测试作为 value function 的 coding agents（HumanEval-style）。
- 探索多条 query paths 的 deep-research agents。
- LangGraph subgraphs 内的 planning-heavy workflows。

AlphaEvolve（Lesson 11）是 2025 年的极端形态：对代码做 evolutionary search，用机器可检查 fitness，获得 frontier gains（56 年来第一次改进 4x4 matmul）。

## Build It / 动手构建

`code/main.py` 实现：

- 一个 tiny ToT BFS，用在风格化的 “pick arithmetic ops” 任务上。
- 同一任务上的 toy LATS MCTS loop（Select / Expand / Simulate / Backpropagate），使用 UCT selection。
- 一个把 symbolic score 和 self-eval score 组合起来的 value function。

运行：

```
python3 code/main.py
```

trace 会展示 ToT 每个节点用 BFS 展开三个候选，并对比 LATS 如何通过 MCTS 收敛到最佳 rollout。两者都会打印 token counts。

## Use It / 应用它

LangGraph 把 ToT-style exploration 作为 subgraph patterns 提供；LangChain 团队 2024 年 5 月关于 LATS 的博客是参考教程。LlamaIndex 提供 `TreeOfThoughts` agent。对于多数 2026 生产 Agent，这个模式通常藏在一个 `if task_complexity > threshold: use_search()` gate 后面；参考 Lesson 05 的 evaluator-optimizer pattern。

## Ship It / 交付它

`outputs/skill-search-policy.md` 会根据任务形状、预算和 evaluator fidelity，在 linear ReAct、ToT、LATS 和 evolutionary search 之间做选择。

## Exercises / 练习

1. 用 UCT c=0.1 和 c=2.0 分别运行 toy LATS。trace 有什么变化？
2. 把 value function 换成更噪声的 scorer（加入 random jitter）。MCTS 还能找到最佳 leaf 吗？它能容忍的最低 signal-to-noise 是多少？
3. 实现 beam-search ToT（每层保留 top-k），并和 BFS 对比。在紧 token budget 下哪种更好？
4. 阅读 LATS Section 5.1。复现 HumanEval trajectory count：要多少 rollouts 才达到报告的 pass@1？
5. 阅读 LATS 论文关于 “when LATS helps less” 的讨论。写一段 decision rule，把任务形状映射到 search strategy。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Tree of Thoughts | “Branching CoT” | Yao et al.：带 self-evaluation 的 thought node tree |
| LATS | “MCTS for LLMs” | Zhou et al.：在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | “Upper confidence bound” | 平衡 exploitation (Q) 和 exploration (ln N / n) 的选择公式 |
| Value function | “How good is this state” | Prompted LLM score 或 environment reward；喂给 backprop |
| Policy | “Action proposer” | ReAct-style generator，产生候选 next thoughts / actions |
| Rollout | “Simulated trajectory” | 从某节点用 policy 走到 leaf，并用 value 评分 |
| Backpropagate | “Update ancestors” | 把 leaf reward 沿路径上推，更新 visit counts 和 Q |
| Search cost | “Token explosion” | Game of 24 上是 CoT 的 100-1000 倍；采用前先预算 |

## Further Reading / 延伸阅读

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — 标准论文
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — 带 Reflexion feedback 的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — search 的 subgraph patterns
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — 带 programmatic evaluator 的 evolutionary search
