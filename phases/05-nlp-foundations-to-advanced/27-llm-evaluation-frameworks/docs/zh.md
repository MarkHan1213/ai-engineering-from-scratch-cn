# LLM Evaluation — RAGAS, DeepEval, G-Eval / LLM 评估：RAGAS、DeepEval、G-Eval

> Exact-match 和 F1 会错过语义等价。人工 review 无法扩展。LLM-as-judge 是生产答案，但必须有足够校准才能信任数字。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 区分 RAGAS、DeepEval 与 G-Eval 的用途和指标
- 从零实现 NLI-style faithfulness 与 answer relevance 近似
- 使用 G-Eval 与 DeepEval 构建自定义评估和 CI gate
- 识别 judge bias、JSON parsing failure、self-evaluation、dataset rot 与 judge drift 风险

## The Problem / 问题

你的 RAG 系统回答："June 29th, 2007."
Gold reference 是："June 29, 2007."
Exact Match 得 0。F1 约 75%。人类会给 100%。

现在把它乘以 10,000 个测试用例。再乘以每次 retriever、chunking、prompt 或 model 改动。你需要一个 evaluator：懂语义、能便宜地规模化运行、不会对 regression 说谎，并能暴露正确失败模式。

2026 年有三个框架负责这个问题。

- **RAGAS.** Retrieval-Augmented Generation ASsessment。四个 RAG metrics（faithfulness、answer-relevance、context-precision、context-recall），底层用 NLI + LLM-judge。研究支撑，轻量。
- **DeepEval.** LLMs 的 pytest。G-Eval、task-completion、hallucination、bias metrics。原生支持 CI/CD。
- **G-Eval.** 一种方法（也是 DeepEval metric）：带 chain-of-thought、自定义 criteria、0-1 score 的 LLM-as-judge。

三者都依赖 LLM-as-judge。这一课会建立方法直觉，以及围绕它的信任层。

## The Concept / 概念

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge.** 用 LLM 根据 rubric 给输出打分，替代静态指标。给定 `(query, context, answer)`，prompt 一个 judge LLM："Score 0-1 on faithfulness." 返回分数。

它为什么有效：LLMs 能以极低成本近似人类判断。GPT-4o-mini 约 ~$0.003 每条评分，能让 1000-sample regression eval run 成本低于 $5。

它为什么会静默失败：

1. **Judge bias.** Judge 偏好更长答案、来自自己 model family 的答案，以及匹配 prompt 风格的答案。
2. **JSON parsing failures.** Bad JSON → NaN score → 从 aggregate 中静默排除。RAGAS 用户很熟悉这个痛点。用 try/except + explicit failure mode gate。
3. **Drift over model versions.** 升级 judge 会改变每个 metric。冻结 judge model + version。

**The RAG four / RAG 四指标。**

| Metric / 指标 | Question / 问题 | Backend / 后端 |
|--------|----------|---------|
| Faithfulness | 答案中的每个 claim 是否来自 retrieved context？ | NLI-based entailment |
| Answer relevance | 答案是否回答了问题？ | 从 answer 生成 hypothetical questions，再与真实 question 比较 |
| Context precision | Retrieved chunks 中有多少比例相关？ | LLM-judge |
| Context recall | Retrieval 是否返回了全部所需信息？ | LLM-judge against gold answer |

**G-Eval.** 定义自定义 criterion："Did the answer cite the correct source?" 框架会自动扩展为 chain-of-thought evaluation steps，再给 0-1 分。适合 RAGAS 未覆盖的 domain-specific quality dimensions。

**Calibration / 校准。** 在与人工标签做相关性前，不要信任原始 judge score。跑 100 个手工标注样本。画 judge vs human。计算 Spearman rho。如果 rho < 0.7，说明 judge rubric 需要重做。

## Build It / 动手构建

### Step 1: faithfulness with NLI (RAGAS-style) / 第 1 步：用 NLI 做 faithfulness（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

把 answer 拆成 atomic claims。用 NLI 检查每个 claim 是否被 retrieved context 支持。Faithfulness = supported 的比例。

### Step 2: answer relevance / 第 2 步：answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果 answer 暗示的问题不同于原问题，relevance 会下降。

### Step 3: G-Eval custom metric / 第 3 步：G-Eval 自定义 metric

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Evaluation steps 就是 rubric。显式步骤比隐式的 "score 0-1" prompt 更稳定。

### Step 4: CI gate / 第 4 步：CI gate

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

把它作为 pytest 文件交付。每个 PR 都运行。Regression 阻塞合并。

### Step 5: toy eval from scratch / 第 5 步：从零 toy eval

见 `code/main.py`。Stdlib-only 近似 faithfulness（answer claims 与 context overlap）和 relevance（answer tokens 与 question tokens overlap）。不是生产方案，只展示形状。

## Pitfalls / 常见坑

- **No calibration.** 与人工标签相关性只有 0.3 的 judge 是噪声。上线前必须做 calibration run。
- **Self-evaluation.** 用同一个 LLM 生成并评判，会把分数抬高 10-20%。Judge 要用不同 model family。
- **Positional bias in pairwise judging.** Judge 偏好第一个选项。始终随机化顺序，并双向运行。
- **Raw aggregate hides failures.** Mean score 0.85 经常隐藏 5% 灾难性失败。始终检查 bottom quantile。
- **Golden dataset rot.** 未版本化 eval sets 随时间漂移，会破坏纵向比较。每次变更都给 dataset 打 tag。
- **LLM cost.** 规模化后，judge calls 会成为主要成本。使用能达到 calibration threshold 的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## Use It / 应用它

2026 stack：

| Use case / 用例 | Framework / 框架 |
|---------|-----------|
| RAG quality monitoring | RAGAS（4 metrics） |
| CI/CD regression gates | DeepEval + pytest |
| Custom domain criteria | DeepEval 内的 G-Eval |
| Online live-traffic monitoring | RAGAS with reference-free mode |
| Human-in-the-loop spot checks | LangSmith 或 Phoenix with annotation UI |
| Red-teaming / safety eval | Promptfoo + DeepEval |

典型 stack：RAGAS 做 monitoring，DeepEval 做 CI，G-Eval 做新维度。三者都跑；它们的分歧很有价值。

## Ship It / 交付它

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## Exercises / 练习

1. **Easy / 简单。** 在 10 个包含已知 hallucinations 的 RAG examples 上使用 RAGAS。验证 faithfulness metric 能抓住每个问题。
2. **Medium / 中等。** 手工给 50 个 QA answers 按 correctness 标 0-1。用 G-Eval 打分。测量 judge 与 human 之间的 Spearman rho。
3. **Hard / 困难。** 用 DeepEval 构建 pytest CI gate。故意让 retriever regression。验证 gate 失败。通过检查最低 10% 分数增加 bottom-quantile alerting。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| LLM-as-judge | 用 LLM 打分 | Prompt judge model 根据 rubric 给输出打 0-1 分。 |
| RAGAS | RAG metric library | 开源 eval framework，提供 4 个 reference-free RAG metrics。 |
| Faithfulness | 答案是否 grounded？ | Answer claims 中被 retrieved context entail 的比例。 |
| Context precision | Retrieved chunks 是否相关？ | Top-K chunks 中真正有用的比例。 |
| Context recall | Retrieval 是否找全？ | Gold-answer claims 中被 retrieved chunks 支持的比例。 |
| G-Eval | Custom LLM judge | Rubric + chain-of-thought eval steps + 0-1 score。 |
| Calibration | 信任但验证 | Judge score 与 human score 之间的 Spearman correlation。 |

## Further Reading / 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — 开源生产 stack。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — biases、calibration、limits。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — 集成 RAGAS、DeepEval、Phoenix 的统一框架。
