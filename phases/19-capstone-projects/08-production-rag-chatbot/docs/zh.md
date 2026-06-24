# Capstone 08 — Production RAG Chatbot for a Regulated Vertical / 受监管行业的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年运行的是同一种生产形态：用 docling 或 Unstructured 摄取文档，用 ColPali 处理视觉内容，hybrid search，用 bge-reranker-v2-gemma 重排序，用 Claude Sonnet 4.7 生成并通过 prompt caching 达到 60-80% hit rate。再用 Llama Guard 4 和 NeMo Guardrails 防护，用 Langfuse 和 Phoenix 观察，用 RAGAS 在 200-question golden set 上评分。在受监管领域（legal、clinical、insurance）构建一个这样的系统，本 capstone 的目标是通过 golden set、red team 和 drift dashboard。

**类型：** 综合项目
**语言：** Python（pipeline + API）, TypeScript（chat UI）
**前置知识：** 第 05 阶段（NLP）, 第 07 阶段（transformers）, 第 11 阶段（LLM engineering）, 第 12 阶段（multimodal）, 第 17 阶段（infrastructure）, 第 18 阶段（safety）
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18
**时间：** 30 小时

## Learning Objectives / 学习目标

- 为 legal、clinical 或 insurance 这类受监管领域构建高保真 ingestion 与 hybrid retrieval
- 实现 role、jurisdiction、citation enforcement、guardrails 和 PII scrub 组成的合规链路
- 利用 prompt caching、rerank 和 drift monitoring 控制成本、延迟和质量衰退
- 构建 golden set、red-team suite、RAGAS online eval 和 Phoenix drift dashboard
- 交付一个可部署、可审计、带 compliance labels 的 production RAG chatbot

## Problem / 问题

Regulated-domain RAG（legal contracts、clinical trial protocols、insurance policies）是 2026 年最常落地的生产形态之一，因为 ROI 明确、风险具体。Harvey（Allen & Overy）为 legal 构建了这种系统。Mendable 提供 developer-docs 版本。Glean 覆盖 enterprise search。模式是：高保真 ingestion，hybrid retrieve with rerank，带 citation enforcement 和 prompt caching 的 synthesis，多层 safety guard，并持续监控 drift。

难点不在模型，而在 jurisdiction-aware compliance（HIPAA、GDPR、SOC2）、citation-level auditability、cost control（prompt caching 在 high hit rate 下能带来 60-90% 折扣）、通过 RAGAS faithfulness 做 hallucination detection，以及当源文档更新但 index 没跟上时的 drift detection。本 capstone 要你在 200-question golden set 上交付这一切，并附带 red-team suite。

## Concept / 概念

pipeline 有两侧。**Ingestion**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；chunks 带 summaries、tags 和 role-based access labels。vectors 写入 pgvector + pgvectorscale（低于 50M vectors）或 Qdrant Cloud；sparse BM25 并行运行。**Conversation**：LangGraph 处理 memory 和 multi-turn；每个 query 先 hybrid retrieval，用 bge-reranker-v2-gemma-2b rerank，再用 Claude Sonnet 4.7（prompt-cached）synthesize，通过 Llama Guard 4 和 NeMo Guardrails，最后输出 citation-anchored response。

eval stack 有四层。**Golden set**（200 个带 citations 的标注 Q/A）用于 correctness。**Red team**（jailbreaks、PII extraction attempts、off-domain questions）用于 safety。**RAGAS** 自动按 turn 评分 faithfulness / answer relevance / context precision。**Drift dashboard**（Arize Phoenix）每周观察 retrieval quality 和 hallucination score。

Prompt caching 是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存 system prompts + retrieved context。当 hit rate 达到 60-80% 时，per-query cost 能下降 3-5x。pipeline 必须为 stable prefixes 设计（system prompt + reranked context first），才能获得高 cache hit rate。

## Architecture / 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## Stack / 技术栈

- Ingestion: Unstructured.io 或 docling 处理 structured documents；ColPali 处理 visually-rich PDFs
- Vector DB: 低于 50M vectors 使用 pgvector + pgvectorscale；否则使用 Qdrant Cloud
- Sparse: Tantivy BM25 with field weights
- Orchestration: LlamaIndex Workflows（ingestion）+ LangGraph（conversation）
- Re-ranker: 自托管 bge-reranker-v2-gemma-2b 或 hosted Voyage rerank-2
- LLM: Claude Sonnet 4.7 with prompt caching；fallback 为 self-hosted Llama 3.3 70B
- Eval: RAGAS 0.2 online，DeepEval 用于 hallucination 和 jailbreak suites
- Observability: self-hosted Langfuse with annotation queue；Arize Phoenix for drift
- Guardrails: Llama Guard 4 input/output classifier、NeMo Guardrails v0.12 policy、Presidio PII scrub
- Compliance: chunks 上的 role-based access labels；GDPR/HIPAA jurisdiction tags

```figure
canary-rollout
```

## Build It / 动手构建

1. **Ingestion.** 用 Unstructured 或 docling 解析你的 corpus（严肃构建建议 1000-10000 documents）。对 scanned / visual-heavy pages 走 ColPali。生成带 summaries、role-labels、jurisdiction tags 的 chunks。

2. **Index.** dense embeddings（Voyage-3 或 Nomic-embed-v2）写入 pgvector + pgvectorscale。BM25 side-index 使用 Tantivy。role 和 jurisdiction filters 作为 payload。

3. **Hybrid retrieve.** 先按 role+jurisdiction filter；再并行 dense + BM25；用 reciprocal rank fusion 合并；top-20 进 reranker；top-5 进 synth。

4. **Synthesize with prompt caching.** System prompt + static policies 放入 cache header；reranked context 作为 cache extension；user question 作为 uncached suffix。steady state 目标 60-80% cache hit rate。

5. **Guardrails.** Llama Guard 4 检查 input；NeMo Guardrails rails 阻止 off-domain questions 或 policy-forbidden topics；Presidio 清理输出中意外出现的 PII；citation enforcement post-filter。

6. **Golden set.** 由 domain expert 标注 200 个 Q/A pairs，包含 (answer, citations)。用 exact-citation match、answer correctness、faithfulness（RAGAS）给 Agent 打分。

7. **Red team.** 50 个 adversarial prompts：jailbreaks（PAIR、TAP）、PII exfiltration attempts、off-domain、cross-jurisdiction leaks。按 pass/fail 和 severity 打分。

8. **Drift dashboard.** Arize Phoenix 每周追踪 retrieval quality（nDCG、citation faithfulness）。下降 5% 时告警。

9. **Cost report.** Langfuse 输出 prompt-caching hit rate、tokens per query、按 stage 拆分的 $/query。

## Use It / 应用它

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## Ship It / 交付它

`outputs/skill-production-rag.md` 描述交付物：一个带 compliance labels 的 regulated-domain chatbot，已通过 rubric，并带 live drift monitoring。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | golden set（200 Q/A）上的 online scores |
| 20 | Citation correctness | 带可验证 source anchors 的 answers 占比 |
| 20 | Guardrail coverage | Llama Guard 4 pass rate + jailbreak suite results |
| 20 | Cost / latency engineering | Prompt-cache hit rate、p95 latency、$/query |
| 15 | Drift monitoring dashboard | Phoenix live dashboard 展示 weekly retrieval-quality trend |
| **100** | | |

## Exercises / 练习

1. 在另一个 jurisdiction 下构建第二个 corpus slice（例如 HIPAA 与 GDPR 并存）。用 20-question cross-jurisdiction probe 证明 role+jurisdiction filtering 防止 cross-leak。

2. 在一周生产流量上测量 prompt-cache hit rate。找出哪些 query 破坏 cache prefix，并重构。

3. 添加带 10k-token summary buffer 的 multi-turn memory。测量 conversation 增长时 faithfulness 是否下降。

4. 把 Claude Sonnet 4.7 换成自托管 Llama 3.3 70B。测量 $/query 和 faithfulness delta。

5. 添加 “unsure” mode：如果 top reranked scores 低于阈值，Agent 说 “I do not have confident citations”，而不是强答。测量 false-confidence reduction。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Prompt caching | “Cached system + context” | Claude/OpenAI 功能：命中时 cached prefix tokens 享受 60-90% 折扣 |
| RAGAS | “RAG evaluator” | 自动评分 faithfulness、answer relevance、context precision |
| Golden set | “Labeled eval” | 200+ expert-labeled Q/A with citations；作为 ground truth |
| Jurisdiction tag | “Compliance label” | 附在 chunks 上的 GDPR/HIPAA/SOC2 scope；由 retrieval filter 强制执行 |
| Citation faithfulness | “Grounded answer rate” | 有可检索 source spans 支撑的 claim 占比 |
| Drift | “Retrieval quality decay” | nDCG 或 citation score 的周变化；5% 为 alert threshold |
| Red team | “Adversarial eval” | pre-release jailbreak、PII extraction、off-domain probes |

## Further Reading / 延伸阅读

- [Harvey AI](https://www.harvey.ai) — legal production stack reference
- [Glean enterprise search](https://www.glean.com) — enterprise scale RAG reference
- [Mendable documentation](https://mendable.ai) — developer-docs RAG reference
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — managed ingestion
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — cost-lever reference
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — canonical RAG eval framework
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — drift observability reference
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 safety classifier
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — policy rail framework
