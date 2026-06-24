# Multimodal Agents and Computer-Use (Capstone) / 多模态 Agent 与 Computer-Use（Capstone）

> 2026 年的前沿产品是 multimodal agent：读取 screenshots、点击按钮、导航 web UIs、填写表单，并端到端完成 workflows。SeeClick 和 CogAgent（2024）证明了 GUI-grounding primitive。Ferret-UI 加入移动端。ChartAgent 引入面向 charts 的 visual tool-use。VisualWebArena 和 AgentVista（2026）是 frontier 追逐的 benchmarks，就连 Gemini 3 Pro 和 Claude Opus 4.7 在 AgentVista hard tasks 上也只有约 30%。这个 capstone 汇总 Phase 12 的全部线索：perception（high-res VLM）、reasoning（带 tool use 的 LLM）、grounding（coordinate output）、long-horizon memory 和 evaluation。

**Type / 类型：** Capstone / 综合项目
**Languages / 语言：** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites / 前置知识：** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time / 时间：** 约 240 分钟

## Learning Objectives / 学习目标

- 设计 multimodal agent loop：perceive → reason → act → observe → repeat。
- 构建 GUI grounding output schema（click coordinates、type text、scroll、drag），让 VLM 以 JSON 输出。
- 比较 screenshot-only agents、accessibility-tree agents 和 hybrid agents。
- 在一个小 VisualWebArena slice 上建立 multimodal agent benchmark evaluation。

## The Problem / 问题

一个 booking-site workflow：“find me a flight to Tokyo for April 15, aisle seat under $800, book it.”

Multimodal agent 需要：

1. 截取 browser screenshot。
2. 把 screenshot + URL + goal 解析成 plan。
3. 输出 structured action：click（at x,y）、type “Tokyo”（at element E）、scroll down、select（radio button）。
4. 把 action 应用到 browser。
5. 观察新状态（下一张 screenshot）。
6. 重复直到任务完成。

每一步都是 multimodal VLM call。VLM output 必须是可解析 JSON。错误会逐步累积，因此 recovery 很重要。

## The Concept / 概念

### GUI grounding — the primitive / GUI grounding：核心原语

GUI grounding 是：给定 screenshot 和自然语言 instruction，输出要点击的 `(x, y)` coordinate（或其他 action）。

SeeClick（arXiv:2401.10935）是第一个大规模 open result：在 synthetic + real GUI data 上 fine-tune VLM，并把 coordinates 作为 plain text tokens 输出。有效。

CogAgent（arXiv:2312.08914）加入 1120x1120 high-resolution encoding，处理 dense UIs。Web navigation 约 84%。

Ferret-UI（arXiv:2404.05719）聚焦 mobile UIs，并整合 iOS accessibility data。

Output format 通常是 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 帮助 recovery：如果两张 screenshot 间坐标漂移，semantic hint 可以让系统 re-ground。

### Action schemas / 动作 schema

典型 action schema 有 6-10 种 action types：

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Agent 每步输出一个 action。Browser wrapper 执行后返回新状态。

### Screenshot-only vs accessibility-tree / Screenshot-only 与 accessibility-tree

两种输入模式：

- Screenshot-only：完整图像，没有结构信息。最通用，适用于任何 app。
- Accessibility tree：结构化 DOM / iOS accessibility info。Grounding 更可靠；前提是 tree 可用。
- Hybrid：两者都用，tree 负责可靠 ground atomic actions，screenshot 提供 semantic context。

生产 agents 在可能时使用 hybrid。Browser automation（Selenium + accessibility）总能拿到 tree；desktop apps 有时不行。

### Long-horizon memory / 长程记忆

20-step workflow 产生 20 张 screenshots。VLM context 很快填满。三种压缩策略：

- Summary-chain：每 5 步总结一次发生了什么，丢旧 screenshots。
- Skip-frame：保留第一张、最后一张和每第 3 张 screenshot。
- Tool-recorded log：执行 actions，保留文字 log；不重新查看旧 screenshots。

Claude computer-use API 使用 log pattern。更简单、更可靠。

### Visual tool use / 视觉工具调用

ChartAgent（arXiv:2510.04514）为 chart understanding 引入 visual tool use：crop、zoom、OCR、调用外部 detection。Agent 可以输出 “crop to region (100, 200, 300, 400) then call OCR” 作为 tool call。Tool 返回 text；VLM 继续推理。

这个模式可泛化：set-of-mark prompting、region annotation 和 external detection tools 都能放进同一个 “output a tool call, receive a structured response” schema。

### The 2026 benchmarks / 2026 benchmarks

- ScreenSpot-Pro。约 1k web screenshots 上的 GUI grounding。Open SOTA Qwen2.5-VL-72B 约 85%。Frontier 约 90%。
- VisualWebArena。端到端 web tasks（shop、forum、classifieds）。Open SOTA 约 20%。Gemini 3 Pro 约 27%。
- AgentVista（arXiv:2602.23166）。2026 年最难 benchmark。12 个 domain 的真实 workflows。Frontier models 27-40%；open models 10-20%。
- WebArena / WebShop。更老的 benchmarks；frontier 已趋饱和。

### Why it's still hard / 为什么仍然难

Agent performance bottlenecks：

1. Fine-scale visual grounding。“Click the small X” 在 mobile resolution 下经常失败。
2. Long-horizon planning。10 个 action 后，agent 会偏离 goal。
3. Error recovery。点击失败（错 button）后，检测 + 恢复的数据很少。
4. Cross-page context。跨 tabs 或长表单时丢状态。

研究方向：memory architectures、explicit replanning、multimodal verification（用 screenshot 验证 action success）。

### The capstone build-it / Capstone 构建任务

Capstone task：构建一个 computer-use agent：

1. 读取 booking-site mock page 的 HTML + screenshot。
2. 规划 multi-step sequence：search → select → fill form → submit。
3. 输出匹配 action schema 的 JSON actions。
4. 在固定 10-task slice 上评估。

Lesson 提供 scaffold code，可扩展成真实 browser。

## Build It / 动手构建

本 capstone 构建完整 agent loop skeleton：定义 action schema，读取 mock browser state，调用 agent 决策，执行 action，观察新状态，并用 mini-benchmark 计算 end-to-end success。重点不是单步 grounding，而是多步错误累积与 recovery。

## Use It / 应用它

`code/main.py` 是 capstone scaffold：

- Action schema JSON definition（10 actions）。
- Mock browser state as dict。
- Agent loop skeleton：receive state、emit action、apply、loop。
- 10-task mini-benchmark（synthetic pages），测量 end-to-end success rate。
- Action failure 时的 error-recovery hook。

## Ship It / 交付它

本课产出 `outputs/skill-multimodal-agent-designer.md`。给定 computer-use product（domain、action set、evaluation target），它会设计完整 agent loop、memory strategy、grounding mode 和 expected benchmark score。

## Exercises / 练习

1. 扩展 action schema，加入 `screenshot_region` tool（crop + zoom）。哪些任务会受益？

2. 阅读 AgentVista（arXiv:2602.23166）。描述最难 task category，以及为什么 frontier models 仍会失败。

3. Long-horizon memory compression：设计一个 summary-chain，live 保留 ≤4 screenshots，log 数量不限。

4. 构建 error-recovery hook：action failure（button not found）后 agent 下一步做什么？

5. 在 10 个 web tasks 上比较 screenshot-only Claude 4.7 与 hybrid screenshot + accessibility-tree Qwen2.5-VL。各自在哪类任务胜出？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GUI grounding | “Click coordinates” | 模型在 screenshot 上为 instruction target 输出 `(x,y)` |
| Action schema | “Tool definitions” | 有效 actions（click、type、scroll、drag）的 JSON 描述 |
| Accessibility tree | “Structured DOM” | 来自 browser/iOS APIs 的 machine-readable UI hierarchy |
| Hybrid agent | “Screenshot + tree” | 同时使用 image 和 structured info；比单独使用任一更可靠 |
| Visual tool use | “Zoom/crop/detect” | Agent 在 plan 中调用外部 vision tools（OCR、detection） |
| Summary-chain | “Memory compression” | 定期用文本 summaries 替代长 screenshot history |
| VisualWebArena | “E2E web bench” | 2024 年 end-to-end web tasks benchmark |
| AgentVista | “2026 hard bench” | 12-domain realistic workflows；即使 Gemini 3 Pro 也约 30% |

## Further Reading / 延伸阅读

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
