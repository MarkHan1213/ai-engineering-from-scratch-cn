# Long-Context Evaluation — NIAH, RULER, LongBench, MRCR / 长上下文评估：NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称 10M tokens context。在 1M tokens 时，8-needle MRCR 掉到 26.3%。Advertised ≠ usable。Long-context evaluation 告诉你要上线的模型实际容量。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 NIAH、RULER、LongBench v2、MRCR、NoLiMa、HELMET 与 BABILong 测量的能力
- 构建自定义 NIAH、多 needle 和 RULER-style variable tracing 测试
- 报告 advertised context、effective retrieval length、effective reasoning length 与 degradation curve
- 识别 NIAH-only、单一 depth、lexical overlap、latency 与 vendor self-report 等评估陷阱

## The Problem / 问题

你有一份 200 页合同。模型声称支持 1M-token context。你把合同贴进去并问："What is the termination clause?" 模型回答了，但它答的是封面页内容，因为 termination clause 位于 120k tokens 深处，已经超过模型真实可关注的位置。

这就是 2026 年的 context-capacity gap。规格表写 1M 或 10M。现实是其中 60-70% 才可用，而且“可用”取决于任务。

- **Retrieval (single needle in haystack):** 前沿模型在标称最大值内接近完美。
- **Multi-hop / aggregation:** 多数模型在约 128k 后急剧退化。
- **Reasoning over dispersed facts:** 最先失败的任务。

Long-context evaluation 衡量这些轴。本课会点名 benchmarks、它们实际测什么，以及如何为你的领域构建自定义 needle test。

## The Concept / 概念

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** 把一个事实（"the magic word is pineapple"）放在长 context 的受控深度。要求模型检索它。扫 depth × length。原始 long-context benchmark。前沿模型现在已经在它上面饱和；它是必要但不充分的 baseline。

**RULER (Nvidia, 2024).** 13 种 task types，覆盖 4 类：retrieval（single / multi-key / multi-value）、multi-hop tracing（variable tracking）、aggregation（common word frequency）、QA。Context length 可配置（4k 到 128k+）。它会揭示那些在 NIAH 饱和、但在 multi-hop 失败的模型。2024 发布时，在 17 个声称 32k+ context 的模型里，只有一半能在 32k 保持质量。

**LongBench v2 (2024).** 503 个 multiple-choice questions，8k-2M words contexts，六类任务：single-doc QA、multi-doc QA、long in-context learning、long dialogue、code repo、long structured data。用于真实世界 long-context behavior 的生产 benchmark。

**MRCR (Multi-Round Coreference Resolution).** 大规模多轮 coreference。8-needle、24-needle、100-needle variants。暴露模型能同时处理多少事实，再开始 attention degradation。

**NoLiMa.** “Non-lexical needle”。Needle 与 query 没有字面重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET.** 拼接大量文档，并从任意一个文档中提问。测试 selective attention。

**BABILong.** 把 bAbI reasoning chains 嵌入无关 haystacks。测试 reasoning-in-a-haystack，而不只是 retrieval。

### What to actually report / 实际应该报告什么

- **Advertised context window.** 规格表数字。
- **Effective retrieval length.** NIAH 在某个阈值（例如 90%）下通过的长度。
- **Effective reasoning length.** Multi-hop 或 aggregation 在该阈值下通过的长度。
- **Degradation curve.** Accuracy vs context length，按 task type 绘制。

给你的规格表两个数字：retrieval-effective 和 reasoning-effective。通常 reasoning-effective 只有 advertised window 的 25-50%。

## Build It / 动手构建

### Step 1: a custom NIAH for your domain / 第 1 步：为你的领域构建自定义 NIAH

见 `code/main.py`。骨架如下：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

扫 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。画 heatmap。这就是你的目标模型 NIAH card。

### Step 2: a multi-needle variant / 第 2 步：multi-needle 变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像 "What are the three magic words?" 这样的问题要求取回全部三个事实。Single-needle success 不能预测 multi-needle success。

### Step 3: multi-hop variable tracing (RULER-style) / 第 3 步：multi-hop variable tracing（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要串联三个 assignments。前沿模型在 128k 上经常会掉到 50-70% accuracy。

### Step 4: LongBench v2 on your stack / 第 4 步：在你的 stack 上跑 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按 category 报告 accuracy。Aggregate scores 会隐藏巨大的任务级差异。

## Pitfalls / 常见坑

- **NIAH-only evaluation.** 在 1M tokens 上通过 NIAH，不能说明 multi-hop 能力。始终运行 RULER 或自定义 multi-hop test。
- **Uniform depth sampling.** 很多实现只测 depth=0.5。要测 depth=0、0.25、0.5、0.75、1.0——"lost in the middle" effect 是真实存在的。
- **Lexical overlap with filler.** 如果 needle 和 filler 共享关键词，检索会变简单。使用 NoLiMa-style non-overlapping needles。
- **Ignoring latency.** 1M-token prompts prefill 需要 30-120 秒。把 time-to-first-token 与 accuracy 一起测。
- **Vendor-self-reported numbers.** OpenAI、Google、Anthropic 都发布自己的分数。始终在你的用例上独立重跑。

## Use It / 应用它

2026 stack：

| Situation / 场景 | Benchmark |
|-----------|-----------|
| 快速 sanity check | Custom NIAH at 3 depths × 3 lengths |
| 生产模型选择 | RULER（13 tasks）at your target length |
| 真实世界 QA 质量 | LongBench v2 single-doc-QA subset |
| Multi-hop reasoning | BABILong 或 custom variable-tracing |
| Conversational / dialogue | MRCR 8-needle at your target length |
| 模型升级 regression | Fixed in-house NIAH + RULER harness，每个新模型都跑 |

生产经验法则：在目标长度上跑完 NIAH + 1 个 reasoning task 之前，不要信任任何 context window。

## Ship It / 交付它

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Exercises / 练习

1. **Easy / 简单。** 构建 3 个 depths（0.25、0.5、0.75）× 3 个 lengths（1k、4k、16k）的 NIAH。任意模型上运行。把 pass rate 画成 3×3 heatmap。
2. **Medium / 中等。** 增加 3-needle 变体。测量每个长度上全部 3 个事实都取回的比例。与同长度 single-needle pass rate 对比。
3. **Hard / 困难。** 构建一个嵌入 64k filler 的 variable-tracing task（X1 → X2 → X3，3 hops）。在 3 个前沿模型上测 accuracy。报告每个模型的 effective reasoning length。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | Needle in haystack | 在 filler 中放入事实，再问模型取回。 |
| RULER | 加强版 NIAH | 13 种 task types，覆盖 retrieval / multi-hop / aggregation / QA。 |
| Effective context | 真实容量 | Accuracy 仍高于阈值的长度。 |
| Lost in the middle | Depth bias | 模型对长输入中间部分关注不足。 |
| Multi-needle | 多个事实同时存在 | 多个植入事实；测试 attention juggling，不只是 retrieval。 |
| MRCR | Multi-round coref | 8、24 或 100-needle coreference；暴露 attention saturation。 |
| NoLiMa | Non-lexical needle | Needle 和 query 没有字面 token 重叠；需要推理。 |

## Further Reading / 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 原始 NIAH repo。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — multi-task benchmark。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 真实世界 long-context eval。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 更难的 needles。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — reasoning-in-haystack。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — depth-bias 论文。
