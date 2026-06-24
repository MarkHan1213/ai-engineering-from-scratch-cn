# Chatbots — Rule-Based to Neural to LLM Agents / 聊天机器人：从规则到神经网络再到 LLM Agents

> ELIZA 用 pattern match 回复。DialogFlow 映射 intents。GPT 从权重里回答。Claude 调用工具并验证。每个时代都在解决上一个时代最糟糕的失败。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 理解 rule-based、retrieval-based、neural generation 与 LLM agent 四类 chatbot 架构
- 实现规则匹配、FAQ retrieval、神经生成 baseline 与 agent loop
- 设计 hybrid routing，把 destructive actions、FAQ 与开放任务分流到不同路径
- 识别 prompt injection、confident fabrication、scope creep、infinite loop 与 context exhaustion 等生产风险

## The Problem / 问题

用户说 "I want to change my flight." 系统必须判断用户想做什么、还缺哪些信息、如何获取信息，以及如何完成动作。然后用户又说 "wait, what if I cancel instead?" 系统必须记住上下文、切换任务，并保留状态。

对 ML 系统来说，对话很难。输入是开放的。输出必须跨多轮保持连贯。系统可能需要对真实世界采取动作（改签航班、扣款）。每一步错误都会被用户看见。

Chatbot architectures 经历了四种范式，每一种都是因为前一种失败得太明显才出现。这一课会按顺序走一遍。2026 年生产形态是最后两种的混合。

## The Concept / 概念

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

**Rule-based (ELIZA, AIML, DialogFlow).** 手写 patterns 匹配用户输入并生成 responses。Intent classifiers 路由到预定义 flows。Slot-filling state machines 收集必需信息。它在设计范围内非常强，一出范围就立刻失败。仍然会在 safety-critical domains（银行认证、航班预订）上线，因为这些场景不能容忍 hallucination。

**Retrieval-based.** FAQ 风格系统。编码每对（utterance, response）。运行时编码用户消息，并检索最近的存储 response。类似 Zendesk 经典的 "similar articles" 功能。比规则更能处理 paraphrases。没有生成，所以没有 hallucination。

**Neural (seq2seq).** 在 conversation logs 上训练 encoder-decoder。从零生成 responses。流畅，但容易给出通用回答（"I don't know"）和事实漂移。永远无法可靠保持主题。这就是 Google、Facebook、Microsoft 在 2016-2019 年的 chatbot 都令人失望的原因。

**LLM agents.** 语言模型被包在一个会计划、调用工具、验证结果的循环里。它不是一个带长 prompt 的 chatbot，而是一个 agent loop：plan → call tool → observe result → decide next step。Retrieval-first grounding（RAG）防止它 hallucinate。Tool calls 让它真的能做事。这是 2026 年架构。

四种范式不是线性替换关系。2026 年生产 chatbot 会同时路由经过四者：rule-based 用于认证和破坏性动作，retrieval 用于 FAQ，neural generation 用于自然措辞，LLM agent 用于模糊开放查询。

## Build It / 动手构建

### Step 1: rule-based pattern matching / 第 1 步：基于规则的 pattern matching

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行 ELIZA。Reflection trick（"I feel sad" → "Why do you feel sad"）是 Weizenbaum 1966 年心理治疗师 demo 的经典技巧。今天仍然有教学价值。

### Step 2: retrieval-based (FAQ) / 第 2 步：retrieval-based（FAQ）

这个示例片段需要 `pip install sentence-transformers`（会带入 torch）。本课可运行的 `code/main.py` 改用 stdlib Jaccard similarity，因此不依赖外部包也能运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒答是关键设计。如果最佳匹配不够近，就返回 `None`，让系统升级到其他路径。

### Step 3: neural generation (baseline) / 第 3 步：神经生成 baseline

使用小型 instruction-tuned encoder-decoder（FLAN-T5）或 fine-tuned conversational model。2026 年它单独上生产不可用（矛盾、跑题、事实胡说），但会在 hybrid systems 中承担自然措辞。DialoGPT 风格 decoder-only models 需要显式 turn separators 和 EOS handling 才能生成连贯回复；FLAN-T5 text2text pipeline 作为教学例子开箱能用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Step 4: LLM agent loop / 第 4 步：LLM agent loop

2026 年生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

三件事要点名。Tools 是 LLM 可以调用的函数。LLM 返回 final answer 而不是 tool call 时，循环结束。Step budget 防止模糊任务上的无限循环。

真实生产还会加入：retrieval-first grounding（每次 LLM call 前注入相关 docs）、guardrails（破坏性动作必须确认）、observability（记录每一步）、evaluations（自动检查 agent behavior 是否仍在规格内）。

### Step 5: hybrid routing / 第 5 步：hybrid routing

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式是：任何 destructive 都走确定性规则，canned FAQs 走 retrieval，其余交给 LLM agents。这就是 2026 customer-support systems 的上线形态。

## Use It / 应用它

2026 stack：

| Use case / 用例 | Architecture / 架构 |
|---------|---------------|
| 预订、付款、认证 | Rule-based state machines + slot filling |
| 客服 FAQs | 在 curated answers 上做 retrieval |
| 开放式帮助聊天 | LLM agent with RAG + tool calls |
| 内部工具 / IDE assistants | LLM agent with tool calls（search, read, write） |
| 陪伴 / 角色聊天机器人 | Tuned LLM with persona system prompt, retrieval on knowledge |

生产中始终使用 hybrid routing。没有单一架构能处理好所有请求。Routing layer 本身通常是一个小型 intent classifier。

## Failure modes that still ship / 仍然会上线的失败模式

- **Confident fabrication / 自信编造。** LLM agent 声称完成了实际上没完成的动作。缓解：验证结果、记录 tool calls、没有成功 tool return 时绝不允许 LLM 声称已经执行。
- **Prompt injection.** 用户插入覆盖 system prompt 的文本。在 OWASP Top 10 for LLM Applications 2025 中排名 LLM01。两种形式：direct injection（直接粘进 chat）和 indirect injection（藏在 agent 读取的文档、邮件或 tool outputs 中）。

  Attack rates 因场景而异。前沿模型在通用 tool-use 和 coding benchmarks 中的测量成功率约为 0.5-8.5%。特定高风险设置（针对 AI coding agents 的 adaptive attacks、脆弱 orchestration）达到过约 84%。生产 CVEs 包括 EchoLeak（CVE-2025-32711，CVSS 9.3）—— Microsoft 365 Copilot 中由 attacker-controlled email 触发的 zero-click data-exfiltration 漏洞。

  缓解：在整个 loop 中把 user input 视为不可信；tool calls 前做 sanitize；把 tool outputs 与 main prompt 隔离；使用 Plan-Verify-Execute (PVE) pattern，让 agent 先规划，再在执行前把每个动作与计划对照验证（这能阻止 tool results 注入新的未计划动作）；destructive actions 要求用户确认；对 tool scopes 应用 least-privilege。

  任何 prompt engineering 都无法完全消除这个风险。需要外部 runtime defense layers（LLM Guard、allowlist validation、semantic anomaly detection）。
- **Scope creep / 范围蔓延。** Tool call 返回了切题但偏离任务的信息，agent 开始跑题。缓解：收窄 tool contracts；保持 system prompt 聚焦；增加 off-task rate evaluations。
- **Infinite loops / 无限循环。** Agent 反复调用同一个 tool。缓解：step budget、tool-call deduplication、LLM judge 判断“我们是否有进展”。
- **Context window exhaustion / 上下文窗口耗尽。** 长对话把最早 turns 挤出上下文。缓解：总结旧 turns、按相似度检索相关历史 turns，或使用 long-context model。

## Ship It / 交付它

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## Exercises / 练习

1. **Easy / 简单。** 用上面的 rule-based respond 实现一个咖啡店点单 bot，写 10 个 patterns。测试边界情况：重复下单、修改、取消、意图不清。
2. **Medium / 中等。** 构建 hybrid FAQ + LLM fallback。为一个 SaaS 产品准备 50 条 canned FAQ entries，LLM fallback 对 docs site 做 retrieval。用 100 个真实 support questions 测量 refusal rate 和 accuracy。
3. **Hard / 困难。** 用三个 tools（search、read-user-data、send-email）实现上面的 agent loop。用 50 个测试场景做 evaluation，其中包含 prompt injection attempts。报告 off-task rate、failed task rate 和任何 injection success。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Intent | 用户想要什么 | 类别标签（book_flight, reset_password）。路由到 handler。 |
| Slot | 一块信息 | Bot 需要的参数（date, destination）。Slot filling 是一连串追问。 |
| RAG | Retrieval plus generation | 检索相关 docs，再 ground LLM response。 |
| Tool call | 函数调用 | LLM 发出带 name + args 的结构化调用。Runtime 执行并返回结果。 |
| Agent loop | Plan, act, verify | Controller 交替运行 LLM calls 和 tool calls，直到任务完成。 |
| Prompt injection | 用户攻击 prompt | 试图覆盖 system prompt 的恶意输入。 |

## Further Reading / 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始 rule-based chatbot 论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google 晚期 neural-chatbot 论文，随后 LLM agents 接管。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名 agent loop pattern 的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年生产指导，到 2026 年仍然成立。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — prompt-injection 论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 让 prompt injection 成为最高安全关注项的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 实用 orchestration-layer defenses，包括 Plan-Verify-Execute 和 user-confirmation flows。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — indirect prompt injection 的 canonical zero-click data-exfiltration CVE。说明有写权限的 agents 为什么需要 runtime defenses。
