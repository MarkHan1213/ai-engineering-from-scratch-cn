# Capstone 05 — Autonomous Research Agent (AI-Scientist Class) / 自主研究 Agent（AI-Scientist 级）

> Sakana 的 AI-Scientist-v2 已经生成并发表完整论文。Agent Laboratory 能跑实验。Allen AI 公开了 traces。2026 年的形态是：围绕实验做 plan-execute-verify tree search、有预算的成本控制、沙箱化代码执行、带视觉反馈的 LaTeX writer，以及自动化 NeurIPS-style reviewer ensemble。本 capstone 要你构建一个这样的系统，在每篇论文 $30 内端到端运行，并通过 Sakana 记录过的 sandbox-escape red team。

**类型：** 综合项目
**语言：** Python（agent + sandbox）, LaTeX（output）
**前置知识：** 第 02 阶段（ML）, 第 03 阶段（deep learning）, 第 07 阶段（transformers）, 第 10 阶段（LLMs from scratch）, 第 14 阶段（agents）, 第 15 阶段（autonomous）, 第 16 阶段（multi-agent）, 第 18 阶段（safety）
**Phases exercised:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**时间：** 40 小时

## Learning Objectives / 学习目标

- 把 autonomous research agent 建模为围绕实验节点的 best-first tree search
- 设计有预算、可复现、带 resource cap 的实验 sandbox
- 实现 writer-reviewer loop：生成 LaTeX、编译 PDF、视觉审稿、自动评分
- 建立 red-team range，验证 network egress、fork bomb 和 filesystem escape 被阻断
- 交付带 paper、review、trace 和 reproducibility bundle 的研究 Agent skill

## Problem / 问题

自主研究 Agent 在 2026 年跨过了一个门槛。Sakana AI 的 AI-Scientist-v2 登上 Nature，生成论文通过了 workshop peer review。ShinkaEvolve（ICLR 2026）把这条路线扩展到 evolving hypotheses。AMD 的 Agent Laboratory 发布了可复现 traces。这些 Agent 并不神秘：它们是在候选实验树上运行 plan-execute-verify loop，带成本上限、种子固定的沙箱和自动审稿。真正的工艺在 loop、budget 和 safety story。

你要用一个窄领域的 seed idea 实现这套 loop（例如在 100M-parameter transformer 上做 attention-sparsity ablations）。价值不在第一次运行就发现新东西，而在基础设施：tree-search、experiment sandbox、writer-reviewer loop、red-team report。Sakana 团队记录过 sandbox-escape failures；你的 Agent 必须通过同一类 red team。

## Concept / 概念

这个 Agent 是 best-first tree search。节点是 experiment specifications：(hypothesis, config, code, expected outcome)。expand step 用小改动生成 children（换 optimizer、移动 batch size、ablate 一个组件）。每个 child 在全新的 sandbox 中运行，并有硬 resource cap。结果回流到 scoring function，按 (novelty × quality × remaining budget) 给节点排序。树一直增长到预算耗尽，然后把最佳分支写成论文。

writer 是多模态的。它生成 LaTeX draft，编译，渲染 figures，再把渲染后的 PDF 送回 Claude Opus 4.7 的 vision mode，让它批判 layout、figure legibility 和 claim-evidence alignment。五个 LLM judges 组成 reviewer ensemble，输出 NeurIPS-style scores（novelty、rigor、clarity、reproducibility、impact）；平均分低于阈值时，论文带着 critique 回到 writer。

Safety 是承重结构。每个实验都在 E2B 或 Daytona sandbox 中运行，没有 network egress，有 bounded wall-clock 和 pinned resource limits。Agent 的 code-generation step 经过 policy layer，阻断可能逃逸 sandbox 的 syscalls。red-team report 复现 Sakana 记录过的 attack surface（fork bombs、filesystem escapes、LLM-written network calls）。

## Architecture / 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## Stack / 技术栈

- Orchestration: 带 checkpointing 和 human-approval gates 的 LangGraph
- Tree search: 自定义 best-first experiment nodes（Sakana v2 的 AB-MCTS-style）
- Sandbox: 每个实验一个 E2B，Docker-in-Docker fallback；通过 cgroups 做 resource caps
- Literature: Semantic Scholar Graph API + OpenAlex + abstracts 的本地 FAISS cache
- Writer: LaTeX template + Claude Opus 4.7（vision mode）用于 figure critique 和 layout
- Reviewer: 5 个 judges 的 ensemble（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max），加权聚合
- Experiment framework: 实际实验用 PyTorch 2.5，W&B 记录日志
- Observability: Langfuse agent traces，每篇论文 $30 hard budget

## Build It / 动手构建

1. **Seed and domain scoping.** 接收一个 seed idea（例如 “investigate sparsity patterns in attention maps of sub-1B transformers”）。定义 search space：models、datasets、compute budget。

2. **Literature pass.** 查询 Semantic Scholar + OpenAlex，取 50 篇最相关且被引用最多的论文；本地缓存 abstracts；生成 1-page domain digest。

3. **Tree scaffolding.** 用 seed hypothesis 初始化 root。实现 `expand(node) -> children`，用小编辑生成 proposals（每个 child 只改一个 config）。实现 `score(node)`，作为 weighted novelty × quality × budget term。

4. **Sandbox wrapping.** 每个实验都运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等价 E2B policy）。seeds 写入 sandbox；outputs 以 read-only mount 方式导出。

5. **Plan-execute-verify loop.** `plan` 提出 children。`execute` 运行 sandbox，捕获 logs 和 metrics。`verify` 对 metrics 运行 unit checks（loss 是否下降？ablation 是否隔离了效应？）。失败节点把 failure reason 存到 tree 上。

6. **Writer.** 预算结束后选择最佳分支。用 matplotlib 渲染 figures。把 branch trace 放进上下文，让 Claude Opus 4.7 生成 LaTeX draft。编译。把编译后的 PDF 回传给 Opus 4.7 vision 做 critique。迭代。

7. **Reviewer ensemble.** 五个 judges 按 (novelty, rigor, clarity, reproducibility, impact) 用 NeurIPS-style rubrics 给 draft 打分。若 mean < 4.0/5，则带 critique 返回 writer。最多重写 3 次。

8. **Red team.** 构建或接入一组针对 sandbox 的 adversarial tasks：fork bombs、network exfiltration attempts、filesystem escapes、LLM-written shell metacharacters。确认全部被阻断。写出 findings。

9. **Reproducibility.** 每篇论文都附带 tree-search trace JSON、seeds、W&B run links、sandbox configs，以及一个能端到端复现的 README。

## Use It / 应用它

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## Ship It / 交付它

`outputs/skill-ai-scientist.md` 是交付物。给定 seed idea + domain + $30 budget，它会运行完整 pipeline，并输出可审阅 paper 与 reproducibility bundle。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Paper quality | 与已发表 workshop papers 对比的 blind rubric review |
| 20 | Experimental rigor | Baselines、seeds、ablations；每个 claim 都有 results table 单元格支撑 |
| 20 | Cost and compute discipline | $30/paper ceiling 被执行，并由 Langfuse trace |
| 20 | Safety | Sandbox red team 通过；network policy 和 kill-switch 已验证 |
| 15 | Reproducibility | One-command rerun 使用相同 seeds 复现论文 |
| **100** | | |

## Exercises / 练习

1. 在同一 domain 下对三个不同 seed ideas 运行 pipeline。比较 tree-search 的重叠部分，识别重复浪费的 compute。

2. 在执行前为预计超过 $5 的节点添加 human-in-the-loop gate。测量总成本下降多少。

3. 把 reviewer ensemble 换成 single judge。用 known-bad papers 的 held-out set 测量 false-accept rate。

4. 引入 network-exfiltration red team test：Agent 写代码尝试 `curl` 外部地址。确认 `--network=none` policy 阻断它，并记录尝试。

5. 用 flat random baseline（相同预算、没有 expansion strategy）对比你的 tree-search。报告 novelty × quality gain。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Tree search | “AB-MCTS-style expansion” | 在 experiment nodes 上按 novelty×quality×budget score 做 best-first exploration |
| Sandbox | “Experiment isolation” | 无网络、有 CPU/memory 边界、固定 seeds、read-only inputs 的容器 |
| Vision critique | “Render-then-read” | 把论文编译成 PDF，再把 PDF 喂给 VLM 做 layout 和 claim-evidence critique |
| Reviewer ensemble | “Automated peer review” | 多个 LLM judges 用 NeurIPS rubric 给论文评分；加权聚合后 gate pipeline |
| Novelty score | “Is this new?” | 对接近 50-paper literature cache 的方案施加惩罚的 heuristic |
| Cost ceiling | “$ budget” | 每篇论文总花费硬上限；由 Langfuse counters + pre-run estimates 执行 |
| Red team | “Sandbox-escape audit” | 如果 policy 错误就能逃逸 sandbox 的 adversarial tasks |

## Further Reading / 延伸阅读

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — reference production research agent
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — original methodology
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — evolutionary extension
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — multi-role research-lab framework
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — reference orchestration layer
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — literature search
- [E2B sandboxes](https://e2b.dev) — reference experiment isolation
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — reviewer ensemble 编码的 rubric
