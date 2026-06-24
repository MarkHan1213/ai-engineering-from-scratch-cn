# Scalable Oversight and Weak-to-Strong Generalization / 可扩展监督与弱到强泛化

> Burns et al.（OpenAI Superalignment，“Weak-to-Strong Generalization”，2023）提出了 superalignment problem 的一个 proxy：用较弱模型生成的 labels 去 fine-tune 强模型。如果强模型能从 imperfect weak supervision 中正确泛化，那么当前 human-scale alignment methods 或许能扩展到 superhuman systems。Scalable oversight 与 W2SG 是互补关系。Scalable oversight（debate、recursive reward modeling、task decomposition）提升 overseer 的有效能力，让它跟得上被监督的模型。W2SG 确保强模型能从 overseer 提供的不完美 supervision 中正确泛化。Debate Helps W2SG（arXiv:2501.13124，2025 年 1 月）把二者结合起来。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, W2SG gap simulator)
**Prerequisites / 前置知识：** Phase 18 · 01 (instruction-following), Phase 18 · 10 (AI Control), Phase 09 (RL foundations)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 定义 scalable oversight 与 weak-to-strong generalization，并解释二者如何互补。
- 描述 Burns et al. 2023 的实验设置：用 GPT-2 生成的 labels fine-tune GPT-4。
- 解释 performance gap recovered（PGR）metric 及其测量对象。
- 说出三种主要 scalable-oversight mechanisms（debate、recursive reward modeling、task decomposition）以及各自一个优势。

## The Problem / 问题

Phase 18 到目前为止的 alignment techniques 都假设 overseer 能评估模型 behaviour。当模型超越人类时，overseer 变成 weak link。Superalignment 问题是：较弱 supervisor 能否可靠地产生更强且 aligned 的模型？

Burns et al. 把这个问题降成可操作的 empirical setup：用弱监督训练强模型，测量强模型 capability 有多少穿过 weak supervision 保留下来。这不是 superalignment 的解法，而是一个可衡量 progress 的 proxy。

## The Concept / 概念

### W2SG: the Burns et al. setup / W2SG：Burns et al. 设置

- Weak model：GPT-2 class。
- Strong model：GPT-4 class。
- Target：任务上 strong GPT-4 ceiling。

流程：
1. 获取 weak model 在任务上的 zero-shot predictions。
2. 在 weak-labeled data 上 fine-tune strong model。
3. 测量 strong fine-tuned model 的 accuracy。

Baselines：weak model 自身 accuracy，以及 strong model 在 gold labels 上的 supervised ceiling。

Gap metric：Performance Gap Recovered（PGR）= (fine-tuned - weak) / (ceiling - weak)。PGR 为 1.0 表示 weak supervision 完全弥合 gap；PGR 为 0 表示 weak supervision 没有帮助。

### Burns et al. empirical findings / Burns et al. 经验发现

在 NLP、chess-puzzle 和 reward-modeling tasks 上，PGR 一直为正：强模型能越过 weak supervisor 的错误继续泛化。Task-specific PGR 从约 20% 到约 80% 不等。强模型会从 imperfect labels 中“理解” intended task，再用 pre-trained priors 补全。

Burns et al. 提到的限制：
- 这里的 weak-to-strong gap 是 capability，不是 alignment。Weak model 是 aligned 的；真正 superhuman model 可能也由 human 对齐，但 capability gap 更大。
- Strong model 的 generalization 可能更多来自 task priors，而不是恢复 ground-truth intent。
- Eliciting strong model's latent knowledge 才是真问题；PGR 只测量一个具体 operationalization。

### Scalable oversight: three mechanisms / 可扩展监督的三种机制

- **Debate (Irving et al. 2018)。** U 的两个 instances 为相反立场辩论；weak judge 选择。假设是：找到有说服力的 true answer 比找到有说服力的 false answer 更容易。2024-2025 的近期工作（Khan et al., Michael et al.）显示 debate 有时有帮助、有时没有，取决于 task structure。
- **Recursive Reward Modeling (Leike et al. 2018)。** U 帮助人类训练 U+1 的 reward model。Overseer 的有效能力随模型增长。
- **Task Decomposition (Christiano, Shlegeris, Amodei 2018)。** 把 hard task 递归分解成人类可检查的 sub-tasks。假设任务可 decomposable。

每种机制都依赖关于 task structure 或 intermediate components alignment 的某种假设。

### Why scalable oversight and W2SG are complementary / 为什么二者互补

Scalable oversight 提高 overseer 的有效 signal quality。W2SG 则从 overseer 能提供的任何 imperfect signal 中弥合 gap。

Lang et al. — Debate Helps Weak-to-Strong Generalization（arXiv:2501.13124）把二者结合：debate protocol 提供更好的 weak labels，strong model 在这些 labels 上训练。论文报告 NLP tasks 上 PGR gains。

### The organizational drama / 组织层面的变化

OpenAI Superalignment team 在 Jan Leike 离开去 Anthropic 后于 2024 年 5 月解散。Agenda（scalable oversight、W2SG、automated alignment research）在 Anthropic 与学术实验室继续推进，包括 MATS（Lesson 28）、Redwood（Lesson 10）、Apollo（Lesson 8）、METR（Lesson 28）。组织结构变了，研究问题没有变。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lessons 6-10 描述 threat 和在 U 不可信假设下的 defensive paradigm。Lesson 11 是 offensive paradigm：让 overseer 足够强，能验证 U 的 alignment。Lessons 12-16 随后转向 adversarial evaluation 的实践 tooling。

## Build It / 动手构建

本课构建一个 W2SG gap simulator：用带结构性错误的 weak labeler 监督 strong model，计算 PGR，并比较 weak-alone、strong-on-gold 与 strong-on-weak。

## Use It / 应用它

`code/main.py` 在 synthetic task 上模拟 W2SG fine-tune。Weak labeler 有 70% accuracy 且错误有结构；strong model 在 gold labels 上有 95% ceiling。你会在 weak labels 上 fine-tune strong model，测量 PGR，并与 strong-on-gold 和 weak-alone 比较。

## Ship It / 交付它

本课产出 `outputs/skill-w2sg-pgr.md`。给定 oversight setup description，它会识别 weak supervisor、strong model、supervision quality，并计算（或请求）PGR。它会标记 claim 是 “weak can supervise strong” 还是 “weak + oversight mechanism can supervise strong”。

## Exercises / 练习

1. 运行 `code/main.py`。报告 weak_accuracy = 0.60、0.70、0.80 时的 PGR。解释 PGR curve 的形状。

2. 修改 weak labeler，使其有 structured error（例如总是在某个 input class 上出错）。PGR 增加、减少还是不变？解释原因。

3. 阅读 Burns et al. 2023 Section 4.3（NLP tasks）。复现 “confidence auxiliary loss” 直觉：当 strong model 比 weak labels 更 confident，谁会赢？

4. 为软件工程任务设计一个结合 debate 与 task decomposition 的 scalable-oversight protocol。说出每个组件的一个 failure mode，并解释组合如何处理或无法处理这些 failure modes。

5. 阐明什么证据会 falsify “weak-to-strong generalization is a viable path to superalignment” 这个 claim。具体说明你需要看到什么 empirical signature。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Scalable oversight | “making the overseer stronger” | 提高 overseer 评估更强模型能力的机制 |
| W2SG | “weak supervises strong” | 在 weak labels 上 fine-tune strong model，并测量 recovered capability |
| PGR | “performance gap recovered” | (fine-tuned - weak) / (ceiling - weak)；1.0 = fully closed，0 = no help |
| Debate | “two U instances argue” | weak judge 在两个 U defenders 之间选择的 scalable oversight mechanism |
| RRM | “recursive reward modeling” | U 帮助训练 U+1 的 reward model；overseer capability 跟随 U |
| Task decomposition | “sub-tasks the human checks” | 把 hard task 递归拆成人类可验证 sub-tasks |
| Superalignment | “aligning superhuman AI” | 关注对齐人类不能直接评估的模型的研究 agenda |

## Further Reading / 延伸阅读

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — W2SG paper。
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — debate mechanism。
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — recursive reward modeling。
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — 2024 empirical debate study。
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — 2025 debate + W2SG 组合。
