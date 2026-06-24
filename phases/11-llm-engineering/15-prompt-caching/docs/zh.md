# Prompt Caching and Context Caching / Prompt Caching 与 Context Caching

> 你的 system prompt 有 4,000 tokens。RAG context 有 20,000 tokens。你每个 request 都把两者一起发出去，而且每次都付费。Prompt caching 让 provider 在自己那边保持这个 prefix 的 warm state，复用时只按正常价格的 10% 计费。用对之后，它可以把推理成本降低 50–90%，把 first-token latency 降低 40–85%。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 理解 prompt caching 如何复用 provider-side KV-cache，降低重复 prefix 的成本与 TTFT
- 比较 Anthropic `cache_control`、OpenAI automatic prefix detection 与 Gemini `CachedContent`
- 设计 cache-friendly prompt layout，把稳定内容放在 prefix，把动态内容放在后面
- 建立 hit rate、write/read cost、TTL 与 break-even 的生产监控方法

## The Problem / 问题

一个 coding agent 在每轮对话里都向 Claude 发送同一段 15,000-token system prompt。20 轮对话，按 $3/M input tokens 计算，光这段输入就要 $0.90，还没算用户真正的消息。每天 10,000 个 conversation，账单会变成 $9,000/day，而这些文本本身从未变化。

你不能缩短 prompt，否则质量会下降。你也不能不发送，因为模型每轮都需要它。唯一的办法，是别再为 provider 已经见过的 prefix 付全价。

这个办法就是 prompt caching。Anthropic 在 2024 年 8 月发布了它（2025 年又加入 1-hour extended-TTL variant），OpenAI 在当年晚些时候做了自动化，Google 随 Gemini 1.5 发布显式 context caching。现在三家都把它作为 frontier models 的一等能力。

## The Concept / 概念

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**The mechanic / 工作机制。** 当一个 request 的 prefix 与最近某个 request 的 prefix 匹配时，provider 会直接复用上次运行留下的 KV-cache，而不是重新编码这些 tokens。第一次写入时你支付小幅 write premium；之后每次命中都获得很大的 read discount。

**Three provider flavors in 2026 / 2026 年三种 provider 形态。**

| Provider | API style | Hit discount | Write premium | Default TTL | Min cacheable |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | Explicit `cache_control` markers on content blocks | 90% off input | 25% surcharge | 5 min (extendable to 1 hour) | 1,024 tokens (Sonnet/Opus), 2,048 (Haiku) |
| OpenAI | Automatic prefix detection | 50% off input | none | Up to 1 hour (best-effort) | 1,024 tokens |
| Google (Gemini) | Explicit `CachedContent` API | Storage-billed; read at ~25% of normal | Storage fee per token·hour | User-set (default 1 hour) | 4,096 tokens (Flash), 32,768 (Pro) |

**The invariant / 不变量。** 三家都只缓存 prefix。如果两个 request 之间有任意 token 不同，第一个不同 token 之后全部 miss。把*稳定*部分放在顶部，把*变化*部分放在底部。

### The cache-friendly layout / 适合缓存的布局

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

违反这个顺序，例如把 user message 放在 system prompt 上方，或把动态 retrieval 插在 few-shots 中间，cache 就永远不会命中。

### The break-even calculation / 盈亏平衡计算

Anthropic 的 25% write premium 意味着一个 cached block 至少要被读取两次才会净省钱。1 write + 1 read 的平均成本是每次 request 0.675x（省 32%）；1 write + 10 reads 的平均成本是 0.205x（省 80%）。经验规则：只要你预计某段内容在 TTL 内会复用至少 3 次，就值得缓存。

## Build It / 动手构建

### Step 1: Anthropic prompt caching with explicit markers / 第 1 步：使用显式 marker 的 Anthropic prompt caching

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` marker 告诉 Anthropic 把这个 block 存 5 分钟。窗口内复用会命中；过期后再次写入。

**Response usage fields / Response usage 字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

CI 中要检查两个字段：如果多个 request 之后 `cache_read_input_tokens` 仍然是 zero，说明你的 cache keys 正在漂移。

### Step 2: one-hour extended TTL / 第 2 步：一小时 extended TTL

对长时间运行的 batch job，5 分钟默认 TTL 可能在 job 间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1-hour TTL 的 write premium 是 2 倍（从 baseline +25% 变成 +50%），但只要一个 batch 对同一 prefix 复用超过 5 次，很快就能回本。

### Step 3: OpenAI automatic caching / 第 3 步：OpenAI 自动缓存

OpenAI 不需要你配置任何东西。任何超过 1,024 tokens 且与近期 request 匹配的 prefix，都会自动获得 50% discount。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同样要遵守 cache-friendly layout。还有两个点会破坏 OpenAI cache，但不一定破坏 Anthropic cache：改变 `user` 字段（它是 cache key component）以及重新排序 tools。

### Step 4: Gemini explicit context caching / 第 4 步：Gemini 显式 context caching

Gemini 把 cache 当作一个你可以创建并命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 按 token·hour 对 cache storage 收费，read 约按 normal input rate 的 25% 计费。当你要在多天内、跨很多 session 复用同一个巨大 prompt 或语料时，这种形态最合适。

### Step 5: measuring hit rate in production / 第 5 步：在生产中测量 hit rate

`code/main.py` 提供了一个模拟三家 provider 的 accountant，会追踪 write/read/miss count，并计算每 1K requests 的 blended cost。Deploy gate 应该包含目标 hit rate；大多数生产 Anthropic setup 在 warmup 后应该看到 >80% 的 read fraction。

## Pitfalls that still ship in 2026 / 2026 年仍会上线的坑

- **Dynamic timestamps at the top / 动态时间戳放在顶部。** `"Current time: 2026-04-22 15:30:02"` 如果位于 system prompt 顶部，每个 request 都 miss。把 timestamp 移到 cache breakpoint 下方。
- **Tool reordering / Tool 重排序。** 用稳定顺序 serialize tools；dict 在 deploy 之间 reshuffle 会打断所有命中。
- **Free-text near-duplicates / 自由文本近似但不相同。** "You are helpful." 和 "You are a helpful assistant." 只差一个字节也会 full miss。
- **Too-small blocks / block 太小。** Anthropic 要求 1,024-token floor（Haiku 是 2,048）。更小的 block 会静默不缓存。
- **Blind cost dashboards / 盲目的成本看板。** 把 “input tokens” 拆成 cached 与 uncached。否则一次流量下降会被误读成 cache win。

## Use It / 应用它

2026 年的 caching stack：

| Situation / 场景 | Pick / 选择 |
|-----------|------|
| Agent 有稳定的 10k+ system prompt，且有多轮对话 | Anthropic `cache_control` with 5-min TTL |
| Batch job 复用一个 prefix 超过 30 分钟 | Anthropic with `ttl: "1h"` |
| GPT-5 上的 serverless endpoint，不想维护自定义 infra | OpenAI automatic（只要让 prefix 稳定且足够长） |
| 跨多天复用巨大 code/doc corpus | Gemini explicit `CachedContent` |
| Cross-provider fallback | 保持跨 provider 的 cacheable prefix layout 一致，让任何 provider 都有机会命中 |

把它和 semantic caching（Phase 11 · 11）组合起来处理 user-message layer：prompt caching 处理*token-identical* 复用，semantic caching 处理*meaning-identical* 复用。

## Ship It / 交付它

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## Exercises / 练习

1. **Easy / 简单。** 取一个 10-turn conversation，里面包含 5,000-token system prompt，对 Claude 分别不使用和使用 `cache_control` 跑一遍。报告两种情况下的 input-token bill。
2. **Medium / 中等。** 写一个 test harness：给定 prompt template 和 request log，计算每家 provider（Anthropic 5m、Anthropic 1h、OpenAI automatic、Gemini explicit）的预期 hit rate 与 dollar savings。
3. **Hard / 困难。** 构建一个 layout optimizer：给定 prompt 和一组标记为 `stable=True/False` 的字段，在不丢信息的前提下重写 prompt，把单个 cache breakpoint 放到最大 cache-friendly position。用真实 Anthropic endpoint 验证。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| Prompt caching | “让长 prompt 变便宜” | 为匹配的 prefix 复用 provider-side KV-cache；重复 input tokens 获得 50–90% discount。 |
| `cache_control` | “Anthropic marker” | Content-block attribute，声明“到这里为止都可缓存”；形式是 `{"type": "ephemeral"}`。 |
| Cache write | “付 premium” | 第一次填充 cache 的 request；Anthropic 上约按 1.25x input rate 计费，OpenAI 免费。 |
| Cache read | “折扣” | 后续 request 匹配 prefix；Anthropic 按 10%，OpenAI 按 50%，Gemini 约按 25% 计费。 |
| TTL | “能活多久” | cache 保持 warm 的秒数；Anthropic 默认 5m（可扩到 1h），OpenAI best-effort up to 1h，Gemini 由用户设置。 |
| Extended TTL | “1-hour Anthropic cache” | `{"type": "ephemeral", "ttl": "1h"}`；write premium 是 2x，但 batch 复用时值得。 |
| Prefix match | “为什么我的 cache miss” | 只有从开头到 breakpoint 的每个 token 都 byte-identical，cache 才会 hit。 |
| Context caching (Gemini) | “显式的那个” | Google 的命名、storage-billed cache object；适合多天复用大 corpus。 |

## Further Reading / 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1h TTL、break-even tables。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — automatic prefix matching。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 与 storage pricing。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 首发文章，包含 latency numbers。
- Phase 11 · 05 (Context Engineering) — 如何切 prompt，cache 才能落地。
- Phase 11 · 11 (Caching and Cost) — 把 prompt caching 与 user message 上的 semantic cache 配合使用。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — prompt caching 暴露出来的 KV-cache memory model；解释为什么 cached prefix 重新读取比重新计算便宜约 10×。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill 是 prompt caching shortcut 的阶段；这篇解释 cache hit 为什么显著降低 TTFT 而不影响 TPOT。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — prompt caching 与 speculative decoding、Flash Attention、MQA/GQA 一样，都是改变 inference cost curve 的杠杆；读它可以理解另外三类杠杆。
