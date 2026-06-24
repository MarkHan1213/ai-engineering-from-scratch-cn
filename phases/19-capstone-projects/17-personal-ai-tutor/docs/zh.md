# Capstone 17 — Personal AI Tutor (Adaptive, Multimodal, with Memory) / 个人 AI 导师（自适应、多模态、带记忆）

> Khanmigo（Khan Academy）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 在 2026 年都已经大规模交付自适应多模态 tutoring。共同形态是 Socratic policy（绝不直接甩答案）、每次交互后更新的 learner model（Bayesian knowledge tracing 风格）、voice + text + photo-math input、curriculum graph retrieval、spaced-repetition scheduling，以及面向年龄适配内容的强 safety filters。本 capstone 要交付一个 subject-specific tutor（K-12 algebra 或 intro Python），用 10 名学习者做两周 efficacy study，并通过 content-safety audit。

**类型：** 综合项目
**语言：** Python（backend, learner model）, TypeScript（web app）, SQL（curriculum graph via Postgres + Neo4j）
**前置知识：** 第 05 阶段（NLP）, 第 06 阶段（speech）, 第 11 阶段（LLM engineering）, 第 12 阶段（multimodal）, 第 14 阶段（agents）, 第 17 阶段（infrastructure）, 第 18 阶段（safety）
**Phases exercised:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**时间：** 30 小时

## Learning Objectives / 学习目标

- 为 K-12 algebra 或 intro Python 构建带 prerequisites 的 curriculum graph
- 实现 Socratic tutor policy、Bayesian knowledge tracing 和 spaced repetition
- 接入 text、voice、photo math 三种输入，以及 privacy-respecting memory
- 设计 Llama Guard 4、age-appropriate filter 和 COPPA-aware retention policy
- 用两周 10 learner efficacy study 评估 learning gain delta

## Problem / 问题

Adaptive tutoring 曾经只是 ed-tech research niche。到 2026 年，它已经成为消费级产品。Khanmigo 部署到美国多数 school districts。Duolingo Max 达到数千万 MAUs。Google LearnLM / Gemini for Education 驱动 Google Classroom 中的 tutoring。Quizlet Q-Chat 与 flashcards 并排出现。Synthesis Tutor 因 tutor-for-curious-kids 走红。共同元素包括：multimodal input（type、speak、photograph equations）、Socratic pedagogy（先提问，再解释）、每次交互后更新的 learner model，以及严格的 age-appropriate safety。

你将为一个具体 cohort 构建这样的系统。测量标准是真实 efficacy study：10 名学习者，两周内做 pre-test 和 post-test。voice loop 必须自然（复用 capstone 03 子栈）。memory 必须尊重隐私。safety filter 必须通过面向 K-12 的 COPPA-aware red-team。

## Concept / 概念

四个组件。**Tutor policy** 是 Socratic loop：学习者索要答案时，policy 先问引导问题；答对时，移动到下一个 concept；卡住时，提供 scaffolded hint。**Learner model** 是 Bayesian knowledge tracing（或简单变体），在每次交互后更新每个 curriculum node 的 mastery probability。**Curriculum graph** 是概念 Neo4j，带 prerequisite edges；policy 遍历 graph 选择下一个 concept。**Memory** 是 episodic + semantic store（agentmemory-style），保存过往交互、错误和偏好。

UX 是多模态的。typed answers 使用 text input。voice input 通过 LiveKit + Whisper（复用 capstone 03）。math problems 的 photo input 使用 dots.ocr 或 PaliGemma 2。voice output 使用 Cartesia Sonic-2。Safety 使用 Llama Guard 4 加 age-appropriate filter（阻断成人内容、暴力、自伤），并采用 COPPA-aware memory retention policy。

efficacy study 是交付物。10 名学习者，pre-test 和 post-test，两周。报告 learning gain delta 和 confidence interval。与 non-adaptive baseline 对比（同样内容，但按线性顺序交付，不使用 tutor policy）。

## Architecture / 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## Stack / 技术栈

- Subject choice: K-12 algebra 或 intro Python（二选一，做深）
- Tutor policy: LangGraph over Claude Sonnet 4.7（with prompt caching）
- Learner model: Bayesian knowledge tracing（classic）或用于 spacing 的 FSRS
- Curriculum graph: 概念与 prerequisite edges 的 Neo4j，并附 OER content
- Memory: agentmemory-style persistent vector + episodic + semantic store
- Voice: LiveKit Agents 1.0 + Cartesia Sonic-2（复用 capstone 03 子栈）
- Photo math: dots.ocr 或 PaliGemma 2 做 equation recognition
- Safety: Llama Guard 4 + custom age-appropriate filter
- Eval: Bloom-level question generation、pre/post test harness、efficacy study tooling

## Build It / 动手构建

1. **Curriculum graph.** 构建包含 50-150 个 concept nodes 的 Neo4j（例如 K-12 algebra，从 “number line” 到 “quadratic formula”），并添加 prerequisite edges。给每个 node 附 OER content（Open Textbook、OpenStax）。

2. **Learner model.** 用 priors 初始化 Bayesian knowledge tracing：guess、slip、learn-rate。每次交互后更新 per-concept mastery，并按 learner 持久化。

3. **Tutor policy.** LangGraph nodes: `read_signal`（学习者答案是 correct / partial / stuck？）、`select_concept`（遍历 curriculum graph 选择最高优先级 concept）、`scaffold`（Socratic prompt）、`update_mastery`。

4. **Memory.** 每次交互写入 episodic store。错误和偏好提升到 semantic memory。COPPA-aware retention policy：1 年后自动删除，且家长可访问。

5. **Voice path.** LiveKit Agents worker 挂到 tutor policy。ASR 使用 Whisper-v3-turbo。TTS 使用 Cartesia Sonic-2。支持 barge-in（复用 capstone 03 mechanics）。

6. **Photo-math path.** 上传或拍摄图片；运行 dots.ocr 或 PaliGemma 2 识别 equation；把 structured input 交给 tutor。

7. **Safety.** 每个 model output 通过 Llama Guard 4 + age-appropriate filter（阻断 self-harm、adult content、violence）。Memory access 按 learner ID scope 限制；提供 parental access surface 用于删除。

8. **Efficacy study.** 10 名学习者，pre-test（标准化 30-question baseline），两周 tutor interaction（每周 3 次 session），post-test。与 10 名学习者的 non-adaptive baseline cohort 对比，内容相同。

9. **Weekly progress reports.** 每个 learner 自动生成 PDF summary：topics explored、mastery trajectories、recommended next steps。

## Use It / 应用它

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## Ship It / 交付它

`outputs/skill-ai-tutor.md` 是交付物。它是一个 subject-specific adaptive tutor，具备 multimodal input、learner model、memory、safety 和 measured efficacy。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Learning gain delta | 10-learner two-week study 中的 pre/post-test delta |
| 20 | Socratic fidelity | transcript samples 上的 rubric score |
| 20 | Multimodal UX | Voice + photo + text coherence end to end |
| 20 | Safety + privacy posture | Llama Guard 4 pass rate + COPPA-aware retention |
| 15 | Curriculum breadth and graph quality | Concept coverage + prerequisite graph consistency |
| **100** | | |

## Exercises / 练习

1. 分别在有无 adaptive learner model（random concept order）的情况下运行 efficacy study。报告 delta。adaptive 应该胜出，但幅度才是重点。

2. 添加 multimodal probe：同一 concept question 分别用 text、voice 和 photo 交付。测量学习者是否在偏好的 modality 下收敛更快。

3. 构建 parent dashboard：练习过的 topics、mastery trajectories、upcoming concepts、safety events（任何 guardrail hits）。与 COPPA 对齐。

4. 添加 language-switch mode：tutor 接受 Spanish input，并用 Spanish 教学。测量 X-Guard coverage。

5. 压测 memory privacy：验证 learner A 即使通过 voice-clip re-ingest attack，也无法看到 learner B 的数据。记录 attempted access 并 alert。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Socratic policy | “Ask, do not dump” | Tutor 先问引导问题，而不是直接给答案 |
| Bayesian knowledge tracing | “BKT” | 按 concept 估计 mastery probability 的经典 learner-model equations |
| FSRS | “Free Spaced Repetition Scheduler” | 2024 spaced-repetition scheduler，比 SM-2 更好 |
| Curriculum graph | “Concept DAG” | 带 prerequisite edges 的 Neo4j 概念图 |
| Episodic memory | “Per-interaction log” | 每次交互都保存下来供后续检索 |
| Semantic memory | “Learned pattern store” | 从 episodic 中压缩提升出来的错误模式与偏好 |
| COPPA | “Kids privacy law” | 美国限制收集 13 岁以下儿童数据的法律 |

## Further Reading / 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — consumer K-12 tutor reference
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — language-learning tutor reference
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — hosted reference model
- [Quizlet Q-Chat](https://quizlet.com) — alternate reference
- [Synthesis Tutor](https://www.synthesis.com) — startup reference
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — spaced-repetition scheduler
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — learner-model classic
- [LiveKit Agents](https://github.com/livekit/agents) — voice stack
