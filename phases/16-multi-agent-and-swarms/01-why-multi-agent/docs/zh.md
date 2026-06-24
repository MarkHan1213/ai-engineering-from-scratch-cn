# Why Multi-Agent? / 为什么需要多 Agent？

> 一个 Agent 会撞上天花板。真正聪明的做法往往不是把它做得更大，而是让更多 Agent 协作。

**类型：** 学习
**语言：** TypeScript
**前置知识：** 第 14 阶段（Agent Engineering）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 识别单 Agent 的上限：上下文溢出、混合职责、顺序瓶颈，并判断什么时候应该拆成多 Agent
- 比较常见编排模式：pipeline、parallel fan-out、supervisor、hierarchical，并为任务结构选择合适模式
- 设计一个职责边界、共享状态和通信契约都清楚的多 Agent 系统
- 分析多 Agent 的复杂度代价：延迟、成本、调试难度，以及它相对单 Agent 简洁性的取舍

## The Problem / 问题

你在 Phase 14 里已经搭过单 Agent。它能读文件、跑命令、调 API，也能根据结果继续推理。然后你把它丢到一个真实代码库里：200 个文件、三种语言、测试依赖基础设施，还要求它先研究外部 API 再写代码。

Agent 卡住了。不是因为 LLM 不够聪明，而是任务已经超过一个 Agent loop 能承受的范围。上下文窗口被文件内容塞满。40 次工具调用前读过的东西被淡忘。它同时想当研究员、工程师和 reviewer，结果三件事都做得平庸。

这就是单 Agent 天花板。凡是任务同时具备下面几类特征，你都会碰到它：

- **上下文超过一个窗口能容纳的量** - 读 50 个文件很容易冲破 200k tokens
- **不同阶段需要不同专长** - 研究文档和生成代码需要完全不同的 prompt
- **工作本来可以并行** - 三个文件为什么要顺序读，而不是同时读？

## The Concept / 概念

### The Single-Agent Ceiling / 单 Agent 天花板

单 Agent 是一个 loop、一个上下文窗口、一个 system prompt。可以这样想：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

会坏在三个地方：

1. **上下文饱和** - 工具结果不断堆积。到第 30 轮时，Agent 可能已经吃下 150k tokens 的文件内容、命令输出和先前推理，第 5 轮的关键细节会被挤掉。

2. **角色混乱** - 一个 system prompt 同时写着“你是研究员、工程师、reviewer 和测试员”，通常只会得到一个半研究、半编码、永远没真正 review 完的 Agent。

3. **顺序瓶颈** - Agent 先读文件 A，再读文件 B，再读文件 C。三次串行 LLM 调用，三次串行工具执行，没有并行度。

### The Multi-Agent Solution / 多 Agent 解法

把工作拆开。每个 Agent 只负责一件事，拥有自己的上下文窗口，以及为这件事定制的 system prompt：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个 Agent 都有：

- 聚焦的 system prompt，例如“你是代码 reviewer，你唯一的任务是找 bug”
- 自己的上下文窗口，不被其他 Agent 的工作污染
- 清晰的输入/输出契约，例如接收研究笔记，输出代码

### Real Systems That Do This / 真实系统里的做法

**Claude Code subagents** - Claude Code 用 `Task` 启动 subagent 时，会创建一个带作用域任务的子 Agent。父 Agent 保持上下文干净，子 Agent 专注做事，然后返回摘要。

**Devin** - 运行 planner agent、coder agent 和 browser agent。planner 拆步骤，coder 写代码，browser 研究文档。它们各自拥有独立上下文。

**Multi-agent coding teams (SWE-bench)** - SWE-bench 上表现更好的系统通常会用 researcher 读代码库、planner 设计修复、coder 实现修复。单 Agent 系统得分更低。

**ChatGPT Deep Research** - 并行启动多个搜索 Agent，每个探索一个角度，最后综合结果。

### The Spectrum / 光谱

Multi-agent 不是二元选择，而是一条光谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent** - 一个 loop，一个 prompt。适合简单任务。

**Subagents** - 父 Agent 为聚焦子任务启动子 Agent。父 Agent 维护计划，子 Agent 汇报结果。这正是 Claude Code 的做法。

**Pipeline** - Agent 顺序运行。Agent A 的输出成为 Agent B 的输入。适合分阶段工作流：research -> code -> review -> test。

**Team** - 多个 Agent 并行运行，通过共享 message bus 交流。每个 Agent 有一个角色，由 orchestrator 协调。适合同时需要多种技能的任务。

**Swarm** - 很多相同或近似相同的 Agent 共享状态，没有固定 orchestrator。Agent 从队列里领取工作。适合高吞吐并行任务。

### The Four Multi-Agent Patterns / 四种多 Agent 模式

#### Pattern 1: Pipeline / 模式 1：Pipeline

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个 Agent 转换数据并传给下一段。它最容易推理，但某一阶段失败会阻塞后续所有阶段。

#### Pattern 2: Fan-out / Fan-in / 模式 2：Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

把工作拆给并行 Agent，再合并结果。适合可以分解成独立子任务的问题。

#### Pattern 3: Orchestrator-Worker / 模式 3：Orchestrator-Worker

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

一个聪明的 orchestrator 决定做什么、委派给 worker、再综合结果。orchestrator 本身也是 Agent，只是它有启动 worker 的工具。

#### Pattern 4: Peer Swarm / 模式 4：Peer Swarm

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有中心 orchestrator。Agent 点对点通信，决策从交互中涌现。它更难调试，但可以扩展到很多 Agent。

### When NOT to Use Multi-Agent / 什么时候不要用多 Agent

多 Agent 会增加复杂度。Agent 之间每一次消息传递都是潜在失败点。调试也会从“读一个对话”变成“跨五个 Agent 追踪消息”。

**这些情况保持单 Agent：**

- 任务能放进一个上下文窗口，例如工作数据少于约 100k tokens
- 不需要为不同阶段使用不同 system prompt
- 顺序执行已经足够快
- 任务简单到拆分带来的开销大于收益

**复杂度成本：**

- 每个 Agent 边界都是一次有损压缩：Agent A 的完整上下文会被摘要成发给 Agent B 的消息
- 协调逻辑本身会出 bug：谁做什么、什么时候做、按什么顺序做
- 延迟会上升：N 个 Agent 至少意味着 N 次串行 LLM 调用，如果互相来回沟通会更多
- 成本会成倍增加：每个 Agent 独立消耗 tokens

经验规则：如果一个任务少于 20 次工具调用，并且能放进 100k tokens，就保持单 Agent。

```figure
swarm-messages
```

## Build It / 动手构建

### Step 1: The Overloaded Single Agent / 第 1 步：过载的单 Agent

下面是一个试图包办所有事的单 Agent。它有一个巨大的 system prompt，一个上下文窗口里同时放研究、代码和 review：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这个方案的问题：

- 上下文窗口每一阶段都会变大。到 review 时，它同时包含研究笔记、代码和前面的推理。
- system prompt 太泛，无法为每个阶段调优。
- 没有任何步骤能并行。

### Step 2: Specialist Agents / 第 2 步：专家 Agent

现在把它拆开。每个 Agent 只做一件事：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家都有聚焦 prompt。每个专家拿到的都是干净上下文，只包含自己需要的输入。

### Step 3: Coordinate Through Messages / 第 3 步：通过消息协调

用显式消息传递把专家串起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个 Agent 只接收发给自己的消息。没有上下文污染。researcher 读文档消耗的 50k tokens 不会进入 reviewer 的上下文。

### Step 4: Compare / 第 4 步：对比

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多 Agent 版本会使用更多总 tokens，因为有三个 Agent、三次独立 LLM 调用。但每个 Agent 的上下文都保持干净。每一阶段的质量提升，来自更专门的 system prompt。

## Use It / 应用它

本课产出一个可复用 prompt，用来判断什么时候应该切换到多 Agent。见 `outputs/prompt-multi-agent-decision.md`。

## Ship It / 交付它

在生产里引入多 Agent 前，先明确回答：

- 这个任务是否真的超过单 Agent 的上下文、角色或顺序执行上限？
- 每个 Agent 的输入/输出契约是什么？
- 哪些信息进入共享状态，哪些只在单个 Agent 的上下文里停留？
- 哪个组件负责协调、重试和失败归因？
- 额外 token 成本、延迟和调试成本是否被业务价值覆盖？

多 Agent 是扩展复杂任务的手段，不是默认架构。先证明单 Agent 不够，再引入更多 Agent。

## Exercises / 练习

1. 增加第四个专家：一个 "tester" Agent，从 coder 接收代码、从 reviewer 接收反馈，然后编写测试
2. 修改 pipeline，让 reviewer 可以把反馈发回 coder 进行一轮修订，最多 2 轮
3. 把顺序 pipeline 改成 fan-out：并行运行 researcher 和 "requirements analyzer" Agent，合并输出后再交给 coder

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Swarm | “AI Agent 的蜂群智能” | 一组共享状态、没有固定领导者的 peer Agent。行为从局部交互中涌现。 |
| Orchestrator | “老板 Agent” | 工具里包含启动和管理其他 Agent 能力的 Agent。它负责计划和委派，但未必亲自干活。 |
| Coordinator | “交通警察” | 非 Agent 组件，通常只是代码而不是 LLM，按规则在 Agent 之间路由消息。 |
| Consensus | “Agent 达成一致” | 多个 Agent 在继续执行前必须达成一致的协议。用于解决冲突输出。 |
| Emergent behavior | “Agent 自己想出来了” | 由 Agent 交互产生、但没有显式编程的系统级模式。可能有用，也可能有害。 |
| Fan-out / fan-in | “Agent 版 map-reduce” | 把任务拆给并行 Agent（fan-out），再合并结果（fan-in）。 |
| Message passing | “Agent 互相说话” | Agent 之间的通信机制：从一个 Agent 发给另一个 Agent 的结构化数据，用来替代共享上下文窗口。 |

## Further Reading / 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 多 Agent 模式综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft 的多 Agent 对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何用 Task 委派
- [CrewAI documentation](https://docs.crewai.com/) - 基于角色的多 Agent 框架
