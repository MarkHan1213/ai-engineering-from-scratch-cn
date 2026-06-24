# Context Engineering: Windows, Budgets, Memory, and Retrieval / Context Engineering：窗口、预算、记忆与检索

> Prompt engineering 是子集。Context engineering 才是全局。Prompt 是你输入的字符串。Context 是进入模型窗口的一切：system instructions、retrieved documents、tool definitions、conversation history、few-shot examples，以及 prompt 本身。2026 年最好的 AI engineers 是 context engineers。他们决定什么进入窗口，什么留在外面，以及以什么顺序进入。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 10 (LLMs from Scratch), Phase 11 Lesson 01-02
**Time / 时间：** 约 90 分钟
**Related / 相关：** Phase 11 · 15 (Prompt Caching) — cache-friendly layout 是 context engineering 的延伸。Phase 5 · 28 (Long-Context Evaluation) 讲如何用 NIAH/RULER 测量 lost-in-the-middle。

## Learning Objectives / 学习目标

- 计算 context window 各组件的 token budgets（system prompt、tools、history、retrieved docs、generation headroom）
- 实现 context window management strategies：truncation、summarization、sliding window for conversation history
- 对 context components 进行优先级排序和顺序安排，让模型 attention 最大程度落在最相关信息上
- 构建 context assembler，根据 query type 和可用窗口空间动态分配 tokens

## The Problem / 问题

Claude Opus 4.7 有 200K token window（beta 可到 1M）。GPT-5 有 400K。Gemini 3 Pro 有 2M。Llama 4 声称 10M。这些数字听起来巨大，直到你开始填它们。

看一个 coding assistant 的真实拆分：System prompt 500 tokens；50 个 tool definitions 8,000 tokens；retrieved documentation 4,000 tokens；conversation history（10 turns）6,000 tokens；current user query 200 tokens；generation budget（max output）4,000 tokens。总计 22,700 tokens。对 128K window 来说只占 18%。

但 attention 不会随 context length 线性扩展。128K context 的模型要支付 quadratic attention cost（vanilla transformers 是 O(n^2)，虽然多数生产模型使用 efficient attention variants）。更重要的是，retrieval accuracy 会下降。“Needle in a Haystack” 测试显示，模型很难找到 long contexts 中间位置的信息。Liu et al. (2023) 研究表明，LLMs 对 long contexts 开头和结尾的信息 retrieval accuracy 接近完美，但位于中间（context 40–70% 位置）的信息会下降 10–20%。这种 “lost-in-the-middle” effect 因模型而异，但影响所有当前架构。

实践教训是：有 200K tokens 可用，不等于使用 200K tokens 就有效。精心策划的 10K-token context，经常胜过直接倾倒的 100K-token context。Context engineering 是在 context window 内最大化 signal-to-noise ratio 的学科。

你放进窗口的每个 token，都会挤掉一个可能承载更相关信息的 token。每个无关 tool definition、每个过时 conversation turn、每个不能回答问题的 retrieved text chunk，都会让模型在当前任务上稍微变差。

## The Concept / 概念

### The Context Window is a Scarce Resource / Context window 是稀缺资源

把 context window 当成 RAM，而不是 disk。它快，并且模型可以直接访问，但有限。你放不下所有东西。你必须选择。

```mermaid
graph TD
    subgraph Window["Context Window (128K tokens)"]
        direction TB
        S["System Prompt\n~500 tokens"] --> T["Tool Definitions\n~2K-8K tokens"]
        T --> R["Retrieved Context\n~2K-10K tokens"]
        R --> H["Conversation History\n~2K-20K tokens"]
        H --> F["Few-shot Examples\n~1K-3K tokens"]
        F --> Q["User Query\n~100-500 tokens"]
        Q --> G["Generation Budget\n~2K-8K tokens"]
    end

    style S fill:#1a1a2e,stroke:#e94560,color:#fff
    style T fill:#1a1a2e,stroke:#0f3460,color:#fff
    style R fill:#1a1a2e,stroke:#ffa500,color:#fff
    style H fill:#1a1a2e,stroke:#51cf66,color:#fff
    style F fill:#1a1a2e,stroke:#9b59b6,color:#fff
    style Q fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#0f3460,color:#fff
```

每个 component 都在争夺空间。增加 tool definitions，就减少 conversation history 的空间。增加 retrieved context，就减少 few-shot examples 的空间。Context engineering 的艺术，就是分配预算以最大化 task performance。

### Lost-in-the-Middle / 中间遗失效应

这是 context engineering 中最重要的经验发现。模型更关注 context 开头和结尾的信息。中间的信息得到更低 attention scores，更容易被忽略。

Liu et al. (2023) 系统测试了这一点。他们把一个 relevant document 放在 20 个 irrelevant documents 的不同位置，并测量 answer accuracy。当 relevant document 在第一或最后时，accuracy 是 85–90%。当它在中间（20 个中的第 10 个位置）时，accuracy 降到 60–70%。

这有直接工程含义：

- 把最重要信息放在最前面（system prompt、critical instructions）
- 把当前 query 和最相关 context 放在最后（recency bias 有帮助）
- 把 context 中间当成最低优先级区域
- 如果必须把信息放在中间，就在末尾重复关键点

```mermaid
graph LR
    subgraph Attention["Attention Distribution Across Context"]
        direction LR
        P1["Position 0-20%\nHIGH attention\n(system prompt)"]
        P2["Position 20-40%\nMODERATE"]
        P3["Position 40-70%\nLOW attention\n(lost in middle)"]
        P4["Position 70-90%\nMODERATE"]
        P5["Position 90-100%\nHIGH attention\n(current query)"]
    end

    style P1 fill:#51cf66,color:#000
    style P2 fill:#ffa500,color:#000
    style P3 fill:#ff6b6b,color:#fff
    style P4 fill:#ffa500,color:#000
    style P5 fill:#51cf66,color:#000
```

### Context Components / Context 组件

**System prompt**：设定 persona、constraints 和 behavioral rules。它放在最前面，并跨 turn 保持稳定。Claude Code 的 system prompt 约 6,000 tokens，包含 tool definitions 和 behavioral instructions。保持紧凑。System prompt 中的每个 word 都会在每次 API call 重复。

**Tool definitions**：每个 tool 增加 50–200 tokens（name、description、parameter schema）。50 个 tools，每个 150 tokens，在任何 conversation 发生前就要 7,500 tokens。Dynamic tool selection（只纳入与当前 query 相关的 tools）可以减少 60–80%。

**Retrieved context**：来自 vector database、search results、file contents 的 documents。Retrieval 质量直接决定 response 质量。坏 retrieval 比没有 retrieval 更糟，因为它会用 noise 填满窗口，还主动误导模型。

**Conversation history**：所有之前的 user messages 与 assistant responses。随 conversation length 线性增长。50-turn conversation，每 turn 200 tokens，就是 10,000 tokens history，其中多数与当前 query 无关。

**Few-shot examples**：展示期望 behavior 的 input/output pairs。两三个选得好的 examples，往往比几千 tokens instruction 更能提升 output quality。但它们占空间。

**Generation budget**：为模型 response 预留的 tokens。如果你把窗口塞满，模型就没有空间回答。至少为 generation 保留 2,000–4,000 tokens。

### Context Compression Strategies / Context 压缩策略

**History summarization**：不要保留所有 previous turns 原文，而是周期性总结 conversation。用 100 tokens 写 “We discussed X, decided Y, and the user wants Z”，替代 10 turns 的 2,000 tokens。当 history 超过阈值（例如 5,000 tokens）时运行 summarization。

**Relevance filtering**：根据 current query 给每个 retrieved document 打分，丢掉低于 threshold 的 documents。如果 retrieve 了 10 个 chunks，只有 3 个相关，就丢弃另外 7 个。3 个高度相关 chunks 胜过 10 个平庸 chunks。

**Tool pruning**：分类 user query intent，只 include 与该 intent 相关的 tools。Code question 不需要 calendar tools。Scheduling question 不需要 file system tools。这可以把 tool definitions 从 8,000 tokens 降到 1,000。

**Recursive summarization**：对非常长 documents 分阶段摘要。先总结每个 section，再总结这些 summaries。50 页 document 变成 500-token digest，仍然捕获关键点。

### Memory Systems / 记忆系统

Context engineering 跨越三个时间尺度。

**Short-term memory**：当前 conversation。直接存在 context window 里。每轮增长。通过 summarization 和 truncation 管理。

**Long-term memory**：跨 conversations 保持的 facts 和 preferences。例如 “The user prefers TypeScript.”、“The project uses PostgreSQL.” 存在数据库里，在 session start 时 retrieve。Claude Code 存在 CLAUDE.md files 中；ChatGPT 存在 memory feature 中。

**Episodic memory**：可能相关的具体过去 interactions。例如 “Last Tuesday, we debugged a similar issue in the auth module.” 以 embeddings 存储，当当前 conversation 与过去 episode 相似时 retrieve。

```mermaid
graph TD
    subgraph Memory["Memory Architecture"]
        direction TB
        STM["Short-term Memory\n(current conversation)\nDirect in context window"]
        LTM["Long-term Memory\n(facts, preferences)\nDB -> retrieved on session start"]
        EM["Episodic Memory\n(past interactions)\nEmbeddings -> retrieved on similarity"]
    end

    Q["Current Query"] --> STM
    Q --> LTM
    Q --> EM

    STM --> CW["Context Window"]
    LTM --> CW
    EM --> CW

    style STM fill:#1a1a2e,stroke:#51cf66,color:#fff
    style LTM fill:#1a1a2e,stroke:#0f3460,color:#fff
    style EM fill:#1a1a2e,stroke:#e94560,color:#fff
    style CW fill:#1a1a2e,stroke:#ffa500,color:#fff
```

### Dynamic Context Assembly / 动态 context 组装

核心洞察是：不同 queries 需要不同 context。Static system prompt + static tools + static history 是浪费。最好的系统会按 query 动态组装 context。

1. Classify query intent
2. Select relevant tools（不是所有 tools）
3. Retrieve relevant documents（不是固定集合）
4. Include relevant history turns（不是全部 history）
5. Add few-shot examples that match the task type
6. 按重要性排序：critical first，important last，optional in the middle

这是好 AI application 与顶级 AI application 的分界线。模型相同，context 才是差异来源。

## Build It / 动手构建

### Step 1: Token Counter / 第 1 步：Token counter

你无法预算无法测量的东西。先构建一个简单 token counter（用 whitespace splitting 近似，因为精确数量取决于 tokenizer）。

```python
import json
import numpy as np
from collections import OrderedDict

def count_tokens(text):
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

def count_tokens_json(obj):
    return count_tokens(json.dumps(obj))
```

### Step 2: Context Budget Manager / 第 2 步：Context budget manager

核心抽象。Budget manager 追踪每个 component 使用多少 tokens，并强制 limits。

```python
class ContextBudget:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.max_tokens = max_tokens
        self.generation_reserve = generation_reserve
        self.available = max_tokens - generation_reserve
        self.allocations = OrderedDict()

    def allocate(self, component, content, max_tokens=None):
        tokens = count_tokens(content)
        if max_tokens and tokens > max_tokens:
            words = content.split()
            target_words = int(max_tokens / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        used = sum(self.allocations.values())
        if used + tokens > self.available:
            allowed = self.available - used
            if allowed <= 0:
                return None, 0
            words = content.split()
            target_words = int(allowed / 1.3)
            content = " ".join(words[:target_words])
            tokens = count_tokens(content)

        self.allocations[component] = tokens
        return content, tokens

    def remaining(self):
        used = sum(self.allocations.values())
        return self.available - used

    def utilization(self):
        used = sum(self.allocations.values())
        return used / self.max_tokens

    def report(self):
        total_used = sum(self.allocations.values())
        lines = []
        lines.append(f"Context Budget Report ({self.max_tokens:,} token window)")
        lines.append("-" * 50)
        for component, tokens in self.allocations.items():
            pct = tokens / self.max_tokens * 100
            bar = "#" * int(pct / 2)
            lines.append(f"  {component:<25} {tokens:>6} tokens ({pct:>5.1f}%) {bar}")
        lines.append("-" * 50)
        lines.append(f"  {'Used':<25} {total_used:>6} tokens ({total_used/self.max_tokens*100:.1f}%)")
        lines.append(f"  {'Generation reserve':<25} {self.generation_reserve:>6} tokens")
        lines.append(f"  {'Remaining':<25} {self.remaining():>6} tokens")
        return "\n".join(lines)
```

### Step 3: Lost-in-the-Middle Reordering / 第 3 步：Lost-in-the-middle 重排

实现 reordering strategy：最重要 items 放在开头和结尾，最不重要的放中间。

```python
def reorder_lost_in_middle(items, scores):
    paired = sorted(zip(scores, items), reverse=True)
    sorted_items = [item for _, item in paired]

    if len(sorted_items) <= 2:
        return sorted_items

    first_half = sorted_items[::2]
    second_half = sorted_items[1::2]
    second_half.reverse()

    return first_half + second_half

def score_relevance(query, documents):
    query_words = set(query.lower().split())
    scores = []
    for doc in documents:
        doc_words = set(doc.lower().split())
        if not query_words:
            scores.append(0.0)
            continue
        overlap = len(query_words & doc_words) / len(query_words)
        scores.append(round(overlap, 3))
    return scores
```

### Step 4: Conversation History Compressor / 第 4 步：Conversation history compressor

总结旧 conversation turns，回收 token budget。

```python
class ConversationManager:
    def __init__(self, max_history_tokens=5000):
        self.turns = []
        self.summaries = []
        self.max_history_tokens = max_history_tokens

    def add_turn(self, role, content):
        self.turns.append({"role": role, "content": content})
        self._compress_if_needed()

    def _compress_if_needed(self):
        total = sum(count_tokens(t["content"]) for t in self.turns)
        if total <= self.max_history_tokens:
            return

        while total > self.max_history_tokens and len(self.turns) > 4:
            old_turns = self.turns[:2]
            summary = self._summarize_turns(old_turns)
            self.summaries.append(summary)
            self.turns = self.turns[2:]
            total = sum(count_tokens(t["content"]) for t in self.turns)

    def _summarize_turns(self, turns):
        parts = []
        for t in turns:
            content = t["content"]
            if len(content) > 100:
                content = content[:100] + "..."
            parts.append(f"{t['role']}: {content}")
        return "Previous: " + " | ".join(parts)

    def get_context(self):
        parts = []
        if self.summaries:
            parts.append("[Conversation Summary]")
            for s in self.summaries:
                parts.append(s)
        parts.append("[Recent Conversation]")
        for t in self.turns:
            parts.append(f"{t['role']}: {t['content']}")
        return "\n".join(parts)

    def token_count(self):
        return count_tokens(self.get_context())
```

### Step 5: Dynamic Tool Selector / 第 5 步：Dynamic tool selector

只 include 与当前 query 相关的 tools。先 classify intent，再 filter。

```python
TOOL_REGISTRY = {
    "read_file": {
        "description": "Read contents of a file",
        "tokens": 120,
        "categories": ["code", "files"],
    },
    "write_file": {
        "description": "Write content to a file",
        "tokens": 150,
        "categories": ["code", "files"],
    },
    "search_code": {
        "description": "Search for patterns in codebase",
        "tokens": 130,
        "categories": ["code"],
    },
    "run_command": {
        "description": "Execute a shell command",
        "tokens": 140,
        "categories": ["code", "system"],
    },
    "create_calendar_event": {
        "description": "Create a new calendar event",
        "tokens": 180,
        "categories": ["calendar"],
    },
    "list_emails": {
        "description": "List recent emails",
        "tokens": 160,
        "categories": ["email"],
    },
    "send_email": {
        "description": "Send an email message",
        "tokens": 200,
        "categories": ["email"],
    },
    "web_search": {
        "description": "Search the web for information",
        "tokens": 140,
        "categories": ["research"],
    },
    "query_database": {
        "description": "Run a SQL query on the database",
        "tokens": 170,
        "categories": ["code", "data"],
    },
    "generate_chart": {
        "description": "Generate a chart from data",
        "tokens": 190,
        "categories": ["data", "visualization"],
    },
}

def classify_intent(query):
    query_lower = query.lower()

    intent_keywords = {
        "code": ["code", "function", "bug", "error", "file", "implement", "refactor", "debug", "test"],
        "calendar": ["meeting", "schedule", "calendar", "appointment", "event"],
        "email": ["email", "mail", "send", "inbox", "message"],
        "research": ["search", "find", "what is", "how does", "explain", "look up"],
        "data": ["data", "query", "database", "chart", "graph", "analytics", "sql"],
    }

    scores = {}
    for intent, keywords in intent_keywords.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[intent] = score

    if not scores:
        return ["code"]

    max_score = max(scores.values())
    return [intent for intent, score in scores.items() if score >= max_score * 0.5]

def select_tools(query, token_budget=2000):
    intents = classify_intent(query)
    relevant = {}
    total_tokens = 0

    for name, tool in TOOL_REGISTRY.items():
        if any(cat in intents for cat in tool["categories"]):
            if total_tokens + tool["tokens"] <= token_budget:
                relevant[name] = tool
                total_tokens += tool["tokens"]

    return relevant, total_tokens
```

### Step 6: Full Context Assembly Pipeline / 第 6 步：完整 context assembly pipeline

把所有东西接起来。给定 query，动态组装最优 context。

```python
class ContextEngine:
    def __init__(self, max_tokens=128000, generation_reserve=4000):
        self.budget = ContextBudget(max_tokens, generation_reserve)
        self.conversation = ConversationManager(max_history_tokens=5000)
        self.system_prompt = (
            "You are a helpful AI assistant. You have access to tools for "
            "code editing, file management, web search, and data analysis. "
            "Use the appropriate tools for each task. Be concise and accurate."
        )
        self.knowledge_base = [
            "Python 3.12 introduced type parameter syntax for generic classes using bracket notation.",
            "The project uses PostgreSQL 16 with pgvector for embedding storage.",
            "Authentication is handled by Supabase Auth with JWT tokens.",
            "The frontend is built with Next.js 15 using the App Router.",
            "API rate limits are set to 100 requests per minute per user.",
            "The deployment pipeline uses GitHub Actions with Docker multi-stage builds.",
            "Test coverage must be above 80% for all new modules.",
            "The codebase follows the repository pattern for data access.",
        ]

    def assemble(self, query):
        self.budget = ContextBudget(self.budget.max_tokens, self.budget.generation_reserve)

        system_content, _ = self.budget.allocate("system_prompt", self.system_prompt, max_tokens=1000)

        tools, tool_tokens = select_tools(query, token_budget=2000)
        tool_text = json.dumps(list(tools.keys()))
        tool_content, _ = self.budget.allocate("tools", tool_text, max_tokens=2000)

        relevance = score_relevance(query, self.knowledge_base)
        threshold = 0.1
        relevant_docs = [
            doc for doc, score in zip(self.knowledge_base, relevance)
            if score >= threshold
        ]

        if relevant_docs:
            doc_scores = [s for s in relevance if s >= threshold]
            reordered = reorder_lost_in_middle(relevant_docs, doc_scores)
            doc_text = "\n".join(reordered)
            doc_content, _ = self.budget.allocate("retrieved_context", doc_text, max_tokens=3000)

        history_text = self.conversation.get_context()
        if history_text.strip():
            history_content, _ = self.budget.allocate("conversation_history", history_text, max_tokens=5000)

        query_content, _ = self.budget.allocate("user_query", query, max_tokens=500)

        return self.budget

    def chat(self, query):
        self.conversation.add_turn("user", query)
        budget = self.assemble(query)
        response = f"[Response to: {query[:50]}...]"
        self.conversation.add_turn("assistant", response)
        return budget


def run_demo():
    print("=" * 60)
    print("  Context Engineering Pipeline Demo")
    print("=" * 60)

    engine = ContextEngine(max_tokens=128000, generation_reserve=4000)

    print("\n--- Query 1: Code task ---")
    budget = engine.chat("Fix the bug in the authentication module where JWT tokens expire too early")
    print(budget.report())

    print("\n--- Query 2: Research task ---")
    budget = engine.chat("What is the best approach for implementing vector search in PostgreSQL?")
    print(budget.report())

    print("\n--- Query 3: After conversation history builds up ---")
    for i in range(8):
        engine.conversation.add_turn("user", f"Follow-up question number {i+1} about the implementation details of the system")
        engine.conversation.add_turn("assistant", f"Here is the response to follow-up {i+1} with technical details about the architecture")

    budget = engine.chat("Now implement the changes we discussed")
    print(budget.report())

    print("\n--- Tool Selection Examples ---")
    test_queries = [
        "Fix the bug in auth.py",
        "Schedule a meeting with the team for Tuesday",
        "Show me the database query performance stats",
        "Search for best practices on error handling",
    ]

    for q in test_queries:
        tools, tokens = select_tools(q)
        intents = classify_intent(q)
        print(f"\n  Query: {q}")
        print(f"  Intents: {intents}")
        print(f"  Tools: {list(tools.keys())} ({tokens} tokens)")

    print("\n--- Lost-in-the-Middle Reordering ---")
    docs = ["Doc A (most relevant)", "Doc B (somewhat relevant)", "Doc C (least relevant)",
            "Doc D (relevant)", "Doc E (moderately relevant)"]
    scores = [0.95, 0.60, 0.20, 0.80, 0.50]
    reordered = reorder_lost_in_middle(docs, scores)
    print(f"  Original order: {docs}")
    print(f"  Scores:         {scores}")
    print(f"  Reordered:      {reordered}")
    print(f"  (Most relevant at start and end, least relevant in middle)")
```

## Use It / 应用它

### Claude Code's Context Strategy / Claude Code 的 context strategy

Claude Code 使用分层方式管理 context。System prompt 包含 behavioral rules 和 tool definitions（约 6K tokens）。当你打开文件时，文件内容会作为 context 注入。当你搜索时，结果会加入 context。旧 conversation turns 会被总结。CLAUDE.md 提供跨 sessions 持久化的 long-term memory。

关键工程决策是：Claude Code 不会把整个 codebase 倾倒进 context。它按需 retrieve relevant files。这就是 context engineering 的实践形态。

### Cursor's Dynamic Context Loading / Cursor 的动态 context loading

Cursor 会把整个 codebase index 成 embeddings。当你输入 query，它会用 vector similarity retrieve 最相关 files 和 code blocks。只有这些片段进入 context window。500K-line codebase 被压缩成 5–10 个最相关 code blocks。

模式就是：embed everything，retrieve on demand，只 include 真正重要的内容。

### ChatGPT Memory / ChatGPT memory

ChatGPT 把用户 preferences 和 facts 作为 long-term memory 存储。每次 conversation start 时，相关 memories 会被 retrieve 并纳入 system prompt。“The user prefers Python” 只花 5 tokens，却能跨 conversations 节省数百 tokens 的重复 instructions。

### RAG as Context Engineering / 把 RAG 看作 context engineering

Retrieval-Augmented Generation 是形式化的 context engineering。你不把知识塞进模型 weights（training），也不放进 system prompt（static context），而是在 query time retrieve relevant documents 并注入 context window。整个 RAG pipeline，包括 chunking、embedding、retrieval、reranking，都为解决一个问题：把正确的信息放进 context window。

## Ship It / 交付它

本课产出 `outputs/prompt-context-optimizer.md`：一个可复用 prompt，用来审计 context assembly strategy 并建议优化。给它 system prompt、tool count、average history length 和 retrieval strategy，它会识别 token waste 并提出改进。

它还产出 `outputs/skill-context-engineering.md`：一个 decision framework，根据 task type、context window size 和 latency budget 设计 context assembly pipelines。

## Exercises / 练习

1. 给 `ContextBudget` class 增加 “token waste detector”。它应该标记使用超过 30% budget 的 components，并针对每个 component type 建议 compression strategy（summarize history、prune tools、re-rank documents）。

2. 为 retrieved context 实现 semantic deduplication。如果两个 retrieved documents 超过 80% 相似（按 word overlap 或 embedding cosine similarity），只保留 score 更高的那个。测量回收了多少 token budget。

3. 构建 “context replay” tool。给定 conversation transcript，让它通过 `ContextEngine` replay，并可视化每轮 budget allocation 如何变化。绘制每个 component 的 token usage over time。识别 context 开始被压缩的 turn。

4. 实现 priority-based tool selector。不要二值 include/exclude，而是给每个 tool 一个与 current query 的 relevance score。按 relevance descending include tools，直到 tool budget 耗尽。比较 include 5、10、20、50 tools 时的 task performance。

5. 构建 multi-strategy context compressor。实现三种 compression strategies（truncation、summarization、key sentences extraction），并在 20 个 documents 上 benchmark。测量 compression ratio 与 information retention 的 tradeoff（compressed version 是否仍然包含 query answer）。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Context window | “模型能读多少” | 模型在一次 forward pass 中处理的最大 tokens（input + output）；GPT-5 为 400K，Claude Opus 4.7 为 200K（1M beta），Gemini 3 Pro 为 2M。 |
| Context engineering | “Advanced prompt engineering” | 决定什么进入 context window、以什么顺序、按什么优先级进入的学科；覆盖 retrieval、compression、tool selection 与 memory management。 |
| Lost-in-the-middle | “模型忘掉中间内容” | 经验发现：LLMs 更关注 context 开头和结尾，中间信息会有 10–20% accuracy drop。 |
| Token budget | “还剩多少 tokens” | 对 context window capacity 的显式分配，覆盖 system prompt、tools、history、retrieval、generation 等组件，并设置 per-component limits。 |
| Dynamic context | “按需加载内容” | 根据 intent classification、relevant tool selection 和 retrieval results，为每个 query 以不同方式组装 context window。 |
| History summarization | “压缩 conversation” | 用 concise summary 替换旧 conversation turns 原文，在保留关键信息的同时降低 token cost。 |
| Tool pruning | “只包含相关 tools” | 分类 query intent，只 include 匹配的 tool definitions，可降低 60–80% tool token cost。 |
| Long-term memory | “跨 sessions 记忆” | 存在数据库中并在 session start retrieve 的 facts 和 preferences，例如 CLAUDE.md、ChatGPT Memory 与类似系统。 |
| Episodic memory | “记住具体过去事件” | 以 embeddings 存储过去 interactions，当当前 query 与过去 conversation 相似时 retrieve。 |
| Generation budget | “留给答案的空间” | 为模型 output 预留的 tokens；如果 context 填满整个 window，模型就没有空间回答。 |

## Further Reading / 延伸阅读

- [Liu et al., 2023 -- "Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172) -- position-dependent attention 的决定性研究，展示模型难以处理 long contexts 中间的信息。
- [Anthropic's Contextual Retrieval blog post](https://www.anthropic.com/news/contextual-retrieval) -- Anthropic 如何做 context-aware chunk retrieval，把 retrieval failure 降低 49%。
- [Simon Willison's "Context Engineering"](https://simonwillison.net/2025/Jun/27/context-engineering/) -- 命名这门学科，并将它与 prompt engineering 区分开的 blog post。
- [LangChain documentation on RAG](https://python.langchain.com/docs/tutorials/rag/) -- 把 retrieval-augmented generation 作为 context engineering pattern 的实践实现。
- [Greg Kamradt's Needle in a Haystack test](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) -- 揭示所有主流模型都存在 position-dependent retrieval failures 的 benchmark。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) -- 为什么 context length 会驱动 memory 和 latency，以及 KV cache、MQA、GQA 如何改变预算计算。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) -- inference 的两个阶段，解释 long prompts 为什么让 TTFT 昂贵但 TPOT 便宜；这是 context-packing tradeoffs 的底层事实。
- [Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (EMNLP 2023)](https://arxiv.org/abs/2305.13245) -- grouped-query attention paper，在不损失质量的情况下把 production decoders 的 KV memory 降低 8×。
