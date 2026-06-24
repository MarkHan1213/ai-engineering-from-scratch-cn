# Heritage of FIPA-ACL and Speech Acts / FIPA-ACL 与言语行为的传承

> 在 MCP 之前，在 A2A 之前，还有 FIPA-ACL。2000 年，IEEE Foundation for Intelligent Physical Agents 批准了一种 Agent 通信语言：二十个 performatives、两种内容语言，以及一组交互协议，例如 contract net、subscribe/notify、request-when。它后来淡出工业界，因为 ontology 成本对 Web 体系太重；但 LLM 复兴多 Agent 之后，大家正在用更宽松的方式重新实现同一批想法：JSON contract 代替 performative，自然语言代替 ontology。本课认真读 FIPA-ACL，是为了让你看清 2026 年的协议决策哪些是重新发明，哪些是真正的新东西，以及当前浪潮将重新遇到哪些 2000 年代已经解决过的问题。

**类型：** 学习
**语言：** Python（stdlib）
**前置知识：** 第 16 阶段 · 01（Why Multi-Agent）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释 speech-act theory、KQML 与 FIPA-ACL 如何塑造今天的 Agent 协议
- 读懂 FIPA-ACL envelope 中的 performative、content、ontology、conversation-id 等字段
- 把 MCP、A2A、ACP、ANP 的现代 JSON 形态映射回 FIPA 的消息意图和交互协议
- 判断现代协议放弃 formal semantics 和 ontology 后，会重新暴露哪些语义漂移风险

## The Problem / 问题

2026 年的 Agent 协议景观很拥挤：MCP 面向工具，A2A 面向 Agent，ACP 面向企业审计，ANP 面向去中心化信任，NLIP 面向自然语言内容，另外还有 CA-MCP 和几十个研究提案。每个规范都宣称自己是基础设施。

诚实地看，大多数规范都在重新发现一个二十年前就存在的决策树。Austin（1962）和 Searle（1969）的言语行为理论提出“话语也是行动”。KQML（1993）把它做成了软件 Agent 的线协议。FIPA-ACL（2000 年批准）给出了标准化版本：二十个 performatives、SL0/SL1 内容语言、contract-net 与 subscribe-notify 等交互协议。JADE 和 JACK 是当时的 Java 参考平台。这个方向在 2010 年左右衰落，因为 ontology 负担太重，而 Web 栈赢了。

当你看 MCP 的 `tools/call`、A2A 的 task lifecycle，或者 CA-MCP 的 shared context store 时，你看到的是一种更软、更 JSON-native 的 FIPA 决策重演。理解这段历史会告诉你两件事：哪些所谓“创新”其实是重新发明，以及新规范会重新踩到哪些旧失败模式。

## The Concept / 概念

### Speech acts, in one paragraph / 一段话讲清言语行为

Austin 注意到，有些句子不是在描述世界，而是在改变世界。“I promise.” “I request.” “I declare.” 他把这类话称作 performative utterances。Searle 将其形式化为五类：assertive、directive、commissive、expressive、declarative。KQML（Finin et al., 1993）把这套理论落到软件 Agent：一条消息由 performative（动作）和 content（动作作用的内容）组成。FIPA-ACL 补齐 KQML 的缺口，并围绕约二十个 performatives 做了标准化。

### The twenty FIPA performatives (partial list) / FIPA 的二十个 performatives（节选）

| Performative | Intent |
|---|---|
| `inform` | “我告诉你 P 为真” |
| `request` | “我请求你做 X” |
| `query-if` | “P 是否为真？” |
| `query-ref` | “X 的值是什么？” |
| `propose` | “我提议我们做 X” |
| `accept-proposal` | “我接受这个提议” |
| `reject-proposal` | “我拒绝这个提议” |
| `agree` | “我同意做 X” |
| `refuse` | “我拒绝做 X” |
| `confirm` | “我确认 P 为真” |
| `disconfirm` | “我否认 P” |
| `not-understood` | “你的消息无法解析” |
| `cfp` | “就 X 征集提案” |
| `subscribe` | “X 变化时通知我” |
| `cancel` | “取消正在进行的 X” |
| `failure` | “我尝试了 X，但失败了” |

完整列表在 `fipa00037.pdf`（FIPA ACL Message Structure）中。重点不是背下来，而是理解：这里每一个 primitive，最终都会被某个 LLM 协议重新加回来。

### Canonical FIPA-ACL message / 标准 FIPA-ACL 消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载协议 envelope；一个字段（`content`）承载 payload。其余字段正是你每次给 JSON 协议加重试、threading 和 ontology 时都会重新发明的东西。

### The two legacy platforms / 两个遗留平台

**JADE**（Java Agent DEvelopment framework，1999-2020s）是最常用的 FIPA 兼容 runtime。Agent 继承基类、交换 ACL 消息、运行在 container 中，并用 “behaviors” 协调。它的交互协议库内置 contract-net、subscribe-notify、request-when 和 propose-accept。

**JACK**（Agent Oriented Software，商业产品）强调在 FIPA 消息之上做 BDI（Belief-Desire-Intention）推理。形式化程度更高，采用面更窄。

随着 Web stack 吃掉多 Agent 用例，两者都式微了。MCP 和 A2A 是 2026 年的 runtime “containers”。

### Why FIPA faded / FIPA 为什么淡出

- **Ontology overhead.** FIPA 要求共享 ontology 才能解析 `content`。就 ontology 达成一致通常是多年标准化过程。Web 直接用了 HTTP + JSON。
- **Formal semantics nobody used.** SL（Semantic Language）给出了严格 truth conditions，但大多数生产系统用自由文本内容，并忽略形式语义。
- **Tooling lock-in.** JADE 只面向 Java；JACK 是商业产品。多语言团队绕开了它们。
- **The internet won the stack.** REST，之后 JSON-RPC，再之后 gRPC，替代了 ACL 的传输层。

### The LLM revival is FIPA-lite / LLM 复兴本质上是 FIPA-lite

比较一个 FIPA `request` 和一个 MCP `tools/call`：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

同样的 envelope，不同的语法。两者都包含：谁、给谁、意图、payload、correlation id。两者谈不上谁革了谁的命；它们只是在同一个设计空间里做了不同取舍。

Liu et al. 2025 survey（"A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP", arXiv:2505.02279）明确指出了这条谱系：MCP 对应 tool-use speech acts，A2A 对应 agent-peer speech acts，ACP 对应 audit-trail speech acts，ANP 对应 decentralized-identity 扩展。新规范是 ACL 的后代，只是换成 JSON 语法和更松的语义。

### The trade-off, stated plainly / 取舍说白了

**FIPA 给了你，而现代规范丢掉的东西：**

- Formal semantics - 可以证明 `inform` 意味着发送方相信内容为真。
- 一个规范化 performative 目录 - 不必反复争论“我们是否应该有 `cancel`？”。
- 数十年的交互协议模式 - contract-net、subscribe-notify、propose-accept，且已有已知正确性属性。

**现代规范给了你，而 FIPA 没做到的东西：**

- JSON-native payload，兼容所有现代工具。
- LLM 可以解释的自然语言内容，不需要手写 ontology。
- Web stack 传输：HTTP、SSE、WebSocket。
- 通过自描述文档做 capability discovery，例如 MCP `listTools`、A2A Agent Card。

更松的意图语义，换来更容易实现。这就是精确取舍。

### Interaction protocols worth porting / 值得迁移的交互协议

FIPA 提供了约 15 种交互协议。三种特别值得带到 LLM 多 Agent 系统里：

1. **Contract Net Protocol (CNP).** manager 发出 `cfp`（call for proposals）；bidder 返回 `propose`；manager 接受或拒绝。这是标准 task-market 模式（Phase 16 · 16 Negotiation）。
2. **Subscribe/Notify.** subscriber 发送 `subscribe`；publisher 在主题变化时发送 `inform`。这就是 2026 年几乎所有 event-bus 的原型。
3. **Request-When.** “当条件 Y 成立时做 X。”带前置条件的延迟动作。2026 年的类比是 durable workflow engine 里的 deferred tasks（Phase 16 · 22 Production Scaling）。

每一种都可以干净地映射到现代 message queue、HTTP + polling，或者 SSE streaming。

### What breaks when you drop the ontology / 放弃 ontology 后会坏在哪里

没有共享 ontology，Agent 就从自然语言内容里推断含义。2026 年已经记录下来的失败模式是 **semantic drift**：两个 Agent 用同一个词（`"customer"`）表示略有差别的概念，接收方 Agent 按错误解释行动，没有任何 schema validator 能抓到。FIPA 的 ontology 要求本会在 parse time 拒绝这条消息。

不走 full ontology 也可以做的缓解：

- `content` 上加 JSON Schema - 在线路层拒绝结构错误。
- Typed artifacts（A2A）- 拒绝错误 modality。
- Envelope 里显式写 performative - 即使 content 是自然语言，也让意图不含糊。

### The 2026 specs, mapped to speech-act heritage / 2026 规范与言语行为传承的映射

| Modern spec | FIPA analog | What it keeps | What it drops |
|---|---|---|---|
| MCP `tools/call` | `request` | 显式意图、correlation id | formal semantics、ontology |
| MCP `resources/read` | `query-ref` | 显式意图、correlation id | formal semantics |
| A2A Task lifecycle | contract-net + request-when | async lifecycle、状态转换 | formal completeness guarantees |
| A2A streaming events | subscribe/notify | async push | typed-predicate subscription |
| CA-MCP shared context | blackboard (Hayes-Roth 1985) | multi-writer shared memory | logical consistency model |
| NLIP | natural-language content | LLM-native | schema |

从上到下看，模式很清楚：保留结构 primitive，丢掉形式主义，让 LLM 去抹平歧义。

## Build It / 动手构建

`code/main.py` 实现了一个纯 stdlib 的 FIPA-ACL translator。它编码和解码标准 ACL envelope，并展示每一种 MCP / A2A 消息形态都能还原到同一组七个字段。demo 会：

- 把五条 MCP 风格和 A2A 风格消息编码为 FIPA-ACL。
- 再把 FIPA-ACL 解码回现代等价物。
- 用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 在一个 manager 和三个 bidder 之间跑玩具 Contract Net 协商。

运行：

```
python3 code/main.py
```

输出是并排 trace：每条现代消息同时展示 2026 JSON 形式和 FIPA-ACL 形式，然后展示 contract-net bid 的 round-trip。协议 primitive 在 round-trip 中保持不变，只有语法不同。

## Use It / 应用它

`outputs/skill-fipa-mapper.md` 是一个 skill，用来读取任意 Agent 协议规范并产出 FIPA-ACL 映射。在采纳新协议前用它回答：“这真的是新东西，还是 JSON 语法里的 `inform`？”

## Ship It / 交付它

不要把 FIPA-ACL 原样带回来。把它的 checklist 带回来：

- 每条消息的意图 primitive（performative）是什么？
- 请求-响应和取消是否有 correlation id？
- 是否有显式 content language（JSON-RPC、plain text、structured typed artifact）？
- 交互协议是一等公民，还是你正在从零重写 contract-net？
- 当两个 Agent 对 content 含义产生分歧（semantic drift）时会发生什么？

任何新协议进入生产前，都要把这五个问题写进文档。

## Exercises / 练习

1. 运行 `code/main.py`。观察 round-trip 编码。识别 `tools/call`、`resources/read` 和 A2A task creation 分别对应哪个 FIPA performative。
2. 给 contract-net demo 增加一个 `cancel` performative，让 manager 可以在竞标中途撤回任务。`cancel` 解决了哪个单靠 retry 解决不了的失败场景？
3. 阅读 FIPA ACL Message Structure（http://www.fipa.org/specs/fipa00037/）第 4.1-4.3 节。选择一个本课没有覆盖的 performative，并描述它的现代 JSON-RPC 类比。
4. 阅读 Liu et al., arXiv:2505.02279。分别列出 MCP、A2A、ACP、ANP 保留和丢弃的 FIPA performative families。
5. 为你自己系统中 `request` performative 的 `content` 字段设计一个最小 JSON-Schema。相比纯自然语言，这个 schema 给了你什么，又付出了什么成本？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Speech act | “会做事的话语” | Austin/Searle：把话语视为行动。ACL 的理论来源。 |
| FIPA | “那个旧 XML 东西” | IEEE Foundation for Intelligent Physical Agents。2000 年标准化 ACL。 |
| ACL | “Agent Communication Language” | FIPA 的 envelope 格式：performative + content + metadata。 |
| Performative | “动词” | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | “FIPA 的前身” | Knowledge Query and Manipulation Language（1993）。更简单、更窄。 |
| Ontology | “共享词表” | 对内容语言所谈概念的形式化定义。 |
| SL0 / SL1 | “FIPA 内容语言” | Semantic Language level 0 和 1，形式化内容语言族。 |
| Contract Net | “任务市场” | manager 发 cfp，bidder 提 proposal，manager 接受。标准交互协议。 |
| Interaction protocol | “消息模式” | 由 performative 构成、具备已知正确性的序列：request-when、subscribe-notify 等。 |

## Further Reading / 延伸阅读

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 连接现代规范与 FIPA 传承的 2025 核心综述
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000 年批准的 envelope 格式
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 完整 performative 目录
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — `request` / `query-ref` 的现代 tool-use 等价物
- [A2A specification](https://a2a-protocol.org/latest/specification/) — contract-net 和 subscribe-notify 的现代 agent-peer 等价物
