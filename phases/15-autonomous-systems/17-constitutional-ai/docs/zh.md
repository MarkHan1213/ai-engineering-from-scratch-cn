# Constitutional AI and Rule Overrides / Constitutional AI 与规则覆盖

> Anthropic 在 2026 年 1 月 22 日发布的 Claude Constitution 长达 79 页，采用 CC0。它从 rule-based alignment 转向 reason-based alignment，并建立四层优先级： (1) safety and supporting human oversight，(2) ethics，(3) Anthropic guidelines，(4) helpfulness。Behaviours 被拆成 hardcoded prohibitions（bioweapons uplift、CSAM），operators 和 users 都不能覆盖；以及 soft-coded defaults，operators 可以在定义边界内调整。2022 年原始工作（Bai et al.）通过 constitution 驱动的 self-critique 和 RLAIF 训练 harmlessness。诚实 caveat 是：reason-based alignment 依赖模型把原则泛化到未预见情境。Anthropic 自己 2023 年的 participatory experiment 显示 public-sourced 和 corporate principles 约 50% 不一致；2026 版本没有纳入这些 findings。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, four-tier priority resolver)
**Prerequisites / 前置知识：** Phase 15 · 06 (Automated alignment research), Phase 15 · 10 (Permission modes)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 rule-based alignment 与 reason-based alignment 的能力和失败模式。
- 描述 Claude Constitution 的四层 priority hierarchy。
- 解释 hardcoded prohibitions 与 soft-coded defaults 在 override 权限上的差异。
- 识别 principle ambiguity、principle conflict 和 principle drift。
- 审计一个部署中的 constitutional layer 是否真的按 priority order 解析冲突。

## The Problem / 问题

部署中的 Agent 会看到设计者从未见过的输入。没有任何 rule list 长到覆盖所有情形，也没有任何 rule list 短到能在推理压力下快速适用。实际问题是：如何把 Agent 对齐到能同时承受长尾案例和快速推理的原则？

Rule-based alignment（RBA）：列出每一种不允许的事。检查快、易审计、无法保持最新，并且经常对未预见但相近的类比过度拒绝。Reason-based alignment（2026 Claude Constitution）：编码原则，让模型 reasoning。它能扩展到未见案例，但更难审计；失败模式从 miss-the-rule 变成 principle-misapplication。

2026 Constitution 明确采取中间位置。Hardcoded prohibitions——错误性不依赖上下文的事项（bioweapons uplift、CSAM）——是 RBA：永远不允许，不受 operator 或 user instruction 影响。其他部分在四层 hierarchy 内 reason-based：safety and supporting human oversight 第一；ethics 第二；Anthropic-declared guidelines 第三；helpfulness 最后。Operators 可以在 soft-coded zone 内调整 defaults，但不能触碰 hardcoded prohibitions。

## The Concept / 概念

### The four-tier priority hierarchy / 四层优先级

1. **Safety and supporting human oversight / 安全与支持人类监督。** 最高优先级。模型优先避免削弱人类和 Anthropic 监督、纠正 AI 的能力。这不是简单的“谨慎”，而是明确“不做让 human oversight 更难的事”。
2. **Ethics / 伦理。** 诚实、避免伤害人、不欺骗、不操纵。当它与 Anthropic guidelines 冲突时，ethics 优先。
3. **Anthropic guidelines / Anthropic 指南。** Anthropic 认为重要的 operational norms：product scope、interaction patterns、何时使用什么 tools。
4. **Helpfulness / 有用性。** 最低层。在更高优先级之内尽可能有用。

当 tiers 冲突时，高层获胜。这和 Unix priorities 或 network QoS 是同一形状——framing 目标是产生可预测 resolution，而不是在任何单一轴上追求最优行为。

### Hardcoded prohibitions vs soft-coded defaults / Hardcoded prohibitions 与 soft-coded defaults

**Hardcoded：**
- Bioweapons / CBRN uplift
- CSAM
- Attacks on critical infrastructure
- 用户直接询问时，不能欺骗用户关于模型身份的信息

Operator 不能覆盖这些。User 也不能覆盖这些。能在 model-weights level 强制的地方，通过 RLHF / Constitutional AI training 实现；不能的地方，在 inference layer 实现。

**Soft-coded defaults（operator-adjustable）：**
- Response length defaults
- Topical scope（模型可以拒绝 operator deployment 范围外的话题）
- Style（formal vs casual）
- Tool-use patterns

Operator adjustments 发生在 declared bound 内。Operator 不能通过重命名 hardcoded prohibitions 来移除它们。

### The 2022 CAI training / 2022 年 CAI 训练

原始 Constitutional AI（Bai et al., 2022）这样训练 harmlessness：

1. 对一组 prompts 生成 responses。
2. 让模型根据 constitution（显式原则）critique 每个 response。
3. 基于 critique 修订 response。
4. 在修订 pairs 上做 RLAIF（reinforcement learning from AI feedback）。

结果是：模型能用原则性解释拒绝 harmful requests，而不是 blanket refusals。2026 Constitution 使用了这一训练的后代，并额外围绕显式 tier hierarchy 做 post-training。

### What reason-based alignment catches and misses / Reason-based alignment 能抓住什么、漏掉什么

**能抓住：**
- 由 allowed primitives 组合出的、原则清晰适用的未预见案例。
- 与禁止事项相近的新颖请求。
- 依赖“你没说 X 不允许”的 social-engineering attacks。

**会漏掉：**
- 利用原则歧义的攻击（“用户要求了，所以 helpfulness 应该同意”）。
- 两个原则在未预见方式下冲突、且 tier order 本身不清晰的情境。
- 训练周期中 principle interpretation 的慢漂移（reinterpretation）。

### The 2023 participatory experiment / 2023 年参与式实验

Anthropic 在 2023 年做过一个实验，比较 corporate-authored constitution 与 public input（约 1,000 名美国受访者）生成的 constitution。两版原则约 50% 一致。在分歧处，public-sourced version 在某些问题上更严格（political-content handling），在另一些问题上更宽松（AI identity self-disclosure）。2026 Constitution 没有纳入 public-sourced findings。这是该方法中有记录的张力。

### Why hardcoded prohibitions are necessary / 为什么 hardcoded prohibitions 必要

Reason-based alignment 单独无法覆盖长尾。如果攻击者能让模型接受某个前提（例如“我们是持证 bioweapons research lab”），就经常能绕过依赖 case reasoning 的原则。Hardcoded prohibitions 不会被 premise framing 弯曲。它们是 Lesson 14 中 “hard constitutional limit” 在 alignment layer 的体现。

### Where the Constitution sits in the stack / Constitution 在系统栈中的位置

Constitution 不是 Lesson 14 的 kill switch。它位于 model layer：模型权重被训练成偏好什么。Kill switches 和 canary tokens 位于 runtime layer：runtime 允许什么。两者都需要。模型权重过于宽松导致 runtime 执行了所有错误动作，这是 runtime problem。Runtime 过度限制导致模型拒绝所有正确动作，也是 runtime problem。不同层覆盖不同类别。

## Build It / 动手构建

本课实现一个四层 priority resolver。它会接收 proposed action 和各层 principle evaluations，并按照 hierarchy 返回 allow、refuse 或 modified action。你会看到 hardcoded prohibition 为什么不能被 helpfulness 覆盖。

## Use It / 应用它

`code/main.py` 实现一个 minimal four-tier priority resolver。Resolver 接收 proposed action 和一组 principle-evaluations（safety、ethics、guidelines、helpfulness），并返回 action、refusal 或 modified action。Driver 运行一组小 case：clear allow、clear disallow、hardcoded prohibition、跨 tiers 的 ambiguous case。

## Ship It / 交付它

`outputs/skill-constitution-review.md` 审计一个 deployment 的 constitutional layer：哪些是 hardcoded，哪些是 soft-coded，operator 可以在哪里调整，以及 four-tier hierarchy 是否真的是 resolution order。

## Exercises / 练习

1. 运行 `code/main.py`。确认即使 helpfulness 很高，hardcoded prohibition 也会触发。把 resolver 改成让 helpfulness 高于 ethics，观察 failure mode。

2. 阅读公开的 79 页 CC0 Claude Constitution。找一个你认为 under-specified 的 principle。写两段说明具体歧义，并提出更严格 formulation。

3. 为 customer-support agent 设计一组 soft-coded defaults。Operator 调整什么？不能触碰什么？说明每条边界。

4. 阅读 Bai et al. 2022 CAI paper。描述一个 Constitutional AI 的 critique-and-revise loop 可能比 blanket rule 产生更差结果的案例，并指出类别。

5. Anthropic 2023 participatory experiment 发现 public 与 corporate principles 约 50% 不一致。选择一个会影响生产部署的类别（例如 political neutrality）。提出一种设计，让 operators 表达自己的 values，同时 hardcoded prohibitions 保持不可放松。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Constitutional AI | “Anthropic 的 alignment method” | 针对 written constitution 做 self-critique + RLAIF |
| Reason-based alignment | “Principles, not rules” | 模型基于原则 reasoning 来处理未见案例 |
| Hardcoded prohibition | “Never do X” | Operator 和 user 都不能覆盖的 rule-based prohibition |
| Soft-coded default | “Operator-adjustable” | 在 declared bound 内可由 operator 控制的行为 |
| Four-tier hierarchy | “Priority order” | safety > ethics > guidelines > helpfulness |
| RLAIF | “AI feedback RL” | Reward 来自模型生成 critique 的 RL |
| Participatory constitution | “Public-sourced principles” | Anthropic 2023 实验；与 corporate principles 约 50% 分歧 |
| Principle drift | “Interpretation slip” | 模型读取固定 principle text 的方式慢慢变化 |

## Further Reading / 延伸阅读

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 79 页 CC0 document。
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — 2022 原始工作。
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — participatory experiment。
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — Constitution 在 RSP stack 中的位置。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Constitution 在 long-horizon deployments 中的作用。
