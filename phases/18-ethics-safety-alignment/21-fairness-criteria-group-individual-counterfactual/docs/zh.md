# Fairness Criteria — Group, Individual, Counterfactual / 公平性准则：群体、个体与反事实

> 公平性文献由三大类组织。Group fairness：demographic parity、equalized odds、conditional use accuracy equality，在 protected groups 的平均层面让 rates 相等。Individual fairness（Dwork et al. 2012）：相似个体获得相似决策；decision map 满足 Lipschitz condition。Counterfactual fairness（Kusner et al. 2017）：如果 counterfactually 改变 sensitive attributes，决策不变，则该决策对个体公平。2024 理论结果（NeurIPS 2024）：CF 与 accuracy 之间存在 inherent trade-off；一个 model-agnostic method 可以把 optimal-but-unfair predictor 转成 CF predictor，并有 bounded accuracy loss。Backtracking counterfactuals（arXiv:2401.13935，2024 年 1 月）：新的 paradigm，避免要求对 legally protected attributes 做 interventions。Philosophical reconciliation（ICLR Blogposts 2024）：在 causal graphs 下，满足某些 group fairness measures 会 entail counterfactual fairness。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, three-criteria comparison)
**Prerequisites / 前置知识：** Phase 18 · 20 (bias), Phase 02 (classical ML)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 说出三种 group-fairness criteria（demographic parity、equalized odds、conditional use accuracy equality）以及一个 impossibility result。
- 用 Dwork et al. 2012 的 Lipschitz formulation 描述 individual fairness。
- 描述 counterfactual fairness 及其对 causal graph 的依赖。
- 解释 backtracking counterfactuals，以及为什么它们绕开了 protected-attribute intervention problem。

## The Problem / 问题

Lesson 20 讨论如何测量 bias。Lesson 21 讨论 measurement 应服务哪个 fairness standard。三大家族给出结构上不同的标准：一个模型可以 group-fair 但 individual-unfair，也可以 counterfactually fair 但 group-unfair。选择标准是 policy decision；没有标准是 universally optimal。

## The Concept / 概念

### Group fairness / 群体公平

- **Demographic parity。** P(Y=1 | A=a) = P(Y=1 | A=a') for all groups。Acceptance rates 相等。
- **Equalized odds。** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a')。各组 TPR 与 FPR 相等。
- **Conditional use accuracy equality。** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a')。各组 predictive value 相等。

Impossibility（Chouldechova, Kleinberg-Mullainathan-Raghavan 2017）：在 unequal base rates 下，这三者不能同时满足。

### Individual fairness / 个体公平

Dwork et al. 2012。给定 task-specific similarity metric d，如果对某个 Lipschitz constant L 有 |f(x) - f(x')| <= L * d(x, x')，则 decision map f 关于 d 是 individually fair。相似个体获得相似决策。

这要求定义 d。这是 policy question，不是纯统计问题。

### Counterfactual fairness / 反事实公平

Kusner et al. 2017。给定 population 的 causal model，如果对个体 i 的 sensitive attributes 做 counterfactual alteration 后，决策不变，则该决策对 i 是 counterfactually fair。

这要求 causal DAG。DAG 是 modeling choice。Counterfactual fairness 的正当性只和 DAG 一样强。

### The CF-vs-accuracy trade-off / CF 与 accuracy 的权衡

NeurIPS 2024 理论结果：counterfactual fairness 与 predictive accuracy 之间存在 inherent trade-off。一个 model-agnostic method 可以把 optimal-but-unfair predictor 转为 CF predictor，accuracy cost 有界。Accuracy cost 取决于 optimal unfair predictor 中 sensitive-attribute coefficient 的大小。

### Backtracking counterfactuals / 回溯式反事实

arXiv:2401.13935（2024 年 1 月）。传统 counterfactuals 要求对 sensitive attribute 做 intervention：“如果这个人是另一种 gender，决策会变吗？” 在法律上，这有问题：classification law 中不能对 protected attributes 做 intervention。

Backtracking counterfactuals 反转方向：不是干预 attribute，而是问这个个体的实际 features 要怎样组合，才会产生 counterfactual outcome。这绕开了法律 objection。

### Philosophical reconciliation / 哲学调和

ICLR Blogposts 2024。在有 causal graph 的情况下，满足某些 group-fairness measures 会 entail counterfactual fairness。三大家族不是正交的；它们是同一个 underlying causal structure 的不同侧面。

这不能消除 impossibility theorems（unequal base rates 仍阻止 simultaneous group fairness）。但它说明 “group” 与 “individual / counterfactual” 的表面对立，部分来自没有显式说明 causal model。

### Where this fits in Phase 18 / 在 Phase 18 中的位置

Lesson 20 是 bias measurement。Lesson 21 是 fairness definition。Lesson 22 是 privacy（differential privacy）。Lesson 23 是 watermarking。这些是 allocation-adjacent lessons，与 deception-adjacent Lessons 7-11 互补。

## Build It / 动手构建

本课构建一个带 sensitive attribute 与 unequal base rates 的 toy binary-classification dataset。你会计算三种 group metrics，再做 demographic-parity re-weighting，观察其他 metrics 和 accuracy 的代价。

## Use It / 应用它

`code/main.py` 构建一个带 sensitive attribute 和 unequal base rates 的 toy binary-classification dataset。你会在 simple classifier 上计算 demographic parity、equalized odds 和 conditional use accuracy equality。观察三种 metrics 彼此 disagree。应用 demographic parity 的 re-weighting，再观察它对另外两个 metrics 的成本。

## Ship It / 交付它

本课产出 `outputs/skill-fairness-criterion.md`。给定 fairness claim 或 policy，它会识别正在 claim 的 criterion，判断在 claimed unequal base rates 下模型是否还能满足其他 criteria，以及该 claim 依赖什么 causal DAG。

## Exercises / 练习

1. 运行 `code/main.py`。报告 default data 上的三种 group metrics。应用针对 demographic parity 的 re-weighting 后重新报告。

2. 使用 non-sensitive features 上的 L2 实现 Dwork et al. 2012 individual-fairness metric。报告在 L=1 时多少 pairs 违反 Lipschitz。

3. 阅读 Kusner et al. 2017。为 resume scoring 构造一个简单 two-feature causal DAG，并识别它隐含的 counterfactual-fairness condition。

4. 2024 backtracking-counterfactuals paper 避免对 protected attributes 做 intervention。描述一个这对 legal compliance 很重要的场景。

5. ICLR 2024 reconciliation 认为 group 与 counterfactual fairness 是同一结构的不同面。选择 `code/main.py` 中三种 criteria 的任意两种，说出让它们等价所需的 causal assumption。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Demographic parity | “equal rates” | P(Y=1 | A=a) 在 groups 间相等 |
| Equalized odds | “equal TPR/FPR” | 各组 true-positive 与 false-positive rates 相等 |
| Conditional use accuracy | “equal PPV/NPV” | 各组 predictive values 相等 |
| Individual fairness | “Lipschitz condition” | 相似个体获得相似决策 |
| Counterfactual fairness | “causal alteration invariance” | counterfactual attribute alteration 后决策不变 |
| Backtracking counterfactual | “explain via actuals” | 从 outcome 向后推理 counterfactual，而不是从 attribute 向前干预 |
| Impossibility theorem | “the three conflict” | Chouldechova / KMR 2017：unequal base rates 下 group criteria 互斥 |

## Further Reading / 延伸阅读

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) — individual fairness。
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) — counterfactual fairness。
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) — impossibility。
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) — protected-attribute interventions 的新范式。
