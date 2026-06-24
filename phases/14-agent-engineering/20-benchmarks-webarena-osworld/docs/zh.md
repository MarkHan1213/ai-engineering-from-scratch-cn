# Benchmarks: WebArena and OSWorld / Benchmarks：WebArena 与 OSWorld

> WebArena 在四个 self-hosted apps 上测试 web-agent capability。OSWorld 在 Ubuntu、Windows、macOS 上测试 desktop-agent capability。发布时（2023-2024），两者都显示最佳 Agent 和人类之间还有巨大差距。差距正在缩小；failure modes 没有变。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib)
**Prerequisites / 前置知识：** Phase 14 · 19 (SWE-bench, GAIA)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 描述 WebArena 的四个 self-hosted apps，以及为什么 execution-based evaluation 重要。
- 解释 OSWorld 为什么使用真实 OS screenshots，而不是 accessibility APIs。
- 说出 OSWorld 两个主要 failure modes：GUI grounding 和 operational knowledge。
- 总结 OSWorld-G 和 OSWorld-Human 在 base benchmark 上增加了什么。

## The Problem / 问题

Generalist agents 可以调用工具。它们能不能驱动浏览器跨 20 次点击完成购物 checkout？能不能只用键盘鼠标配置 Linux 机器？WebArena 和 OSWorld 回答的就是这些问题。

## The Concept / 概念

### WebArena (Zhou et al., ICLR 2024) / WebArena

- 812 个 long-horizon tasks，覆盖四个 self-hosted web apps：购物站点、论坛、GitLab-like dev tool、business CMS。
- 另有 utilities：map、calculator、scratchpad。
- 通过 gym APIs 做 execution-based evaluation：订单是否下单、issue 是否关闭、CMS 页面是否更新？
- 发布时：最佳 GPT-4 Agent 成功率 14.41%，人类 78.24%。

self-hosted framing 很重要：目标 apps 被固定和可复现，benchmark 不会因线上服务变化而 flaky。

### Extensions / 扩展

- **VisualWebArena**：visually grounded tasks，成功依赖图像解释（screenshots 是一等 observations）。
- **TheAgentCompany**（2024 年 12 月）：增加 terminal + coding，更像真实 remote-work environment。

### OSWorld (Xie et al., NeurIPS 2024) / OSWorld

- 369 个真实 computer tasks，跨 Ubuntu、Windows、macOS。
- 对真实应用进行自由形式 keyboard 和 mouse control。
- 以 1920x1080 screenshots 作为 observation。
- 发布时：最佳模型 12.24%，人类 72.36%。

### Primary failure modes / 主要失败模式

1. **GUI grounding。** Pixel -> element mapping。模型很难在 1920x1080 中可靠定位 UI elements。
2. **Operational knowledge。** 哪个菜单有设置，哪个快捷键，哪个 preference pane。这是人类多年积累的知识长尾。

### Follow-ups / 后续工作

- **OSWorld-G**：564-sample grounding suite + Jedi training set。把 grounding 从 planning 中拆开，让你能单独测量。
- **OSWorld-Human**：人工整理的 gold action trajectories。显示顶级 agents 的步数比必要步数多 1.4-2.7 倍（trajectory-efficiency gap）。

### Why this matters / 为什么重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（Lesson 21）都训练在类似 WebArena 和 OSWorld 的 workload 上。这些 benchmarks 是目标；生产模型是被交付出来的答案。

### Where benchmarking goes wrong / benchmark 常见误用

- **Screenshot-only evals。** OSWorld 是 screenshot-driven；如果一个 Agent 用 DOM 或 accessibility APIs 跑 OSWorld，就绕过了 grounding challenge。
- **Ignoring trajectory length。** 只看 success-rate 会漏掉 OSWorld-Human 暴露的 1.4-2.7x step inefficiency。
- **Stale self-hosted apps。** WebArena apps pin 了具体版本；不重新 curate 就升级 app，会破坏可比性。

## Build It / 动手构建

`code/main.py` 实现一个 toy web-agent harness：

- 一个最小 “shopping app” state machine：list_items、add_to_cart、checkout。
- 3 个 tasks 的 gold trajectories。
- 一个脚本化 Agent 尝试完成每个 task。
- Execution-based evaluator（state check）和 trajectory-efficiency metric（steps vs gold）。

运行：

```
python3 code/main.py
```

输出：per-task success rate 和 trajectory efficiency，对齐 OSWorld-Human 的方法。

## Use It / 应用它

- 在 internal cluster 上 self-host **WebArena Verified**，做 continuous evaluation。
- 在 VM fleet 中跑 **OSWorld**，评测 desktop agents。
- **Computer-use agents**（Lesson 21）：Claude、OpenAI CUA、Gemini 都训练在这类 workload 上。
- **你的产品 flows**：为前 20 个任务捕获 gold trajectories，每周让 Agent 跑一次。

## Ship It / 交付它

`outputs/skill-web-desktop-harness.md` 会生成 web / desktop agent harness，带 execution-based eval 和 trajectory efficiency metric。

## Exercises / 练习

1. 给 toy harness 增加第二个 app（论坛）。写 3 个 tasks 和 gold trajectories。
2. 增加 per-task trajectory-efficiency reporting。你的 toy Agent 是 gold 的 1x、2x 还是 3x？
3. 实现一个 “distractor” tool，也就是 gold trajectory 从不使用的工具。脚本化 Agent 会被诱惑吗？
4. 阅读 OSWorld-G。你会如何在自己的 evals 中区分 grounding failures 和 planning failures？
5. 阅读 WebArena apps README。升级其中一个 pinned app version 会破坏什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| WebArena | “Web agent benchmark” | 4 个 self-hosted apps 上的 812 个 tasks；gym-style evaluation |
| VisualWebArena | “Visual WebArena” | visually grounded WebArena；screenshots 是 observations |
| OSWorld | “Desktop agent benchmark” | Ubuntu / Windows / macOS 真实任务 369 个 |
| GUI grounding | “Pixel-to-element mapping” | 模型在 1920x1080 中定位 UI elements |
| Operational knowledge | “OS know-how” | 哪个菜单、哪个快捷键、哪个 preference pane |
| OSWorld-G | “Grounding suite” | 564 个 grounding-only samples + training set |
| OSWorld-Human | “Gold trajectories” | 人工 expert action sequences，用于测效率 |
| Trajectory efficiency | “Steps over gold” | Agent step count 除以 human minimum |

## Further Reading / 延伸阅读

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — four-app web benchmark
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — cross-OS desktop benchmark
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的 benchmark-shaped capability
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena numbers
