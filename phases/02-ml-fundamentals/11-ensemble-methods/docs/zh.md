# Ensemble Methods / 集成方法

> 一组 weak learners，只要组合得当，就会变成 strong learner。这不是比喻，而是定理。

**Type / 类型：** Build / 构建
**Language / 语言：** Python
**Prerequisites / 前置知识：** Phase 2, Lesson 10 (Bias-Variance Tradeoff)
**Time / 时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 从零实现 AdaBoost 和 gradient boosting，并解释 boosting 如何顺序降低 bias
- 构建 bagging ensemble，并演示平均去相关模型如何在不增加 bias 的情况下降低 variance
- 从各自针对的 error component 角度比较 bagging、boosting 和 stacking
- 评估 ensemble diversity，并解释为什么更多独立 weak learners 的 majority voting accuracy 会提升

## The Problem / 问题

单棵 decision tree 训练快、易解释，但容易 overfit。单个 linear model 会在复杂边界上 underfit。你可以花几天设计完美模型架构。或者，你可以把一堆不完美模型组合起来，得到比任何单个模型都更好的结果。

Ensemble methods 做的正是这件事。它们是 Kaggle tabular data 比赛中最可靠的获胜技术，支撑着许多生产 ML 系统，也非常直观地展示了 bias-variance tradeoff。Bagging 降低 variance。Boosting 降低 bias。Stacking 学会在不同输入上信任哪个模型。

## The Concept / 概念

### Why Ensembles Work / 为什么 ensembles 有效

假设你有 N 个独立 classifiers，每个 accuracy 为 p > 0.5。Majority vote 的 accuracy 是：

```
P(majority correct) = sum over k > N/2 of C(N,k) * p^k * (1-p)^(N-k)
```

如果 21 个 classifiers 各自只有 60% accuracy，majority vote accuracy 约为 74%。如果有 101 个，会升到 84%。当模型犯不同错误时，错误会相互抵消。

关键要求是 **diversity**。如果所有模型犯同样错误，组合它们没有帮助。Ensembles 通过这些方式产生 diverse models：

- 不同 training subsets（bagging）
- 不同 feature subsets（random forests）
- 顺序纠错（boosting）
- 不同 model families（stacking）

### Bagging (Bootstrap Aggregating) / Bagging（Bootstrap aggregating）

Bagging 通过让每个模型在不同 bootstrap sample 上训练来制造多样性。

```mermaid
flowchart TD
    D[Training Data] --> B1[Bootstrap Sample 1]
    D --> B2[Bootstrap Sample 2]
    D --> B3[Bootstrap Sample 3]
    D --> BN[Bootstrap Sample N]

    B1 --> M1[Model 1]
    B2 --> M2[Model 2]
    B3 --> M3[Model 3]
    BN --> MN[Model N]

    M1 --> V[Average or Majority Vote]
    M2 --> V
    M3 --> V
    MN --> V

    V --> P[Final Prediction]
```

Bootstrap sample 是从原始数据中有放回抽样得到的样本，大小与原始数据相同。每个 bootstrap 中约有 63.2% 的 unique samples 会出现，剩下 36.8%（out-of-bag samples）提供了免费 validation set。

Bagging 会降低 variance，而几乎不增加 bias。每棵树都会 overfit 自己的 bootstrap sample，但不同树 overfit 的噪声不同，平均后噪声会被抵消。

**Random Forests** 是带额外机制的 bagging：每次 split 时，只考虑随机 feature subset。这会进一步增加树之间的 diversity。候选 features 的典型数量是 classification 中的 `sqrt(n_features)`，以及 regression 中的 `n_features / 3`。

### Boosting (Sequential Error Correction) / Boosting（顺序纠错）

Boosting 顺序训练模型。每个新模型都关注之前模型预测错的样本。

```mermaid
flowchart LR
    D[Data with weights] --> M1[Model 1]
    M1 --> E1[Find errors]
    E1 --> W1[Increase weights on errors]
    W1 --> M2[Model 2]
    M2 --> E2[Find errors]
    E2 --> W2[Increase weights on errors]
    W2 --> M3[Model 3]
    M3 --> F[Weighted sum of all models]
```

Boosting 会降低 bias。每个新模型都修正当前 ensemble 的 systematic errors。最终 prediction 是所有模型的 weighted sum，表现更好的模型得到更高权重。

权衡是：boosting 如果轮数太多会 overfit，因为它会不断拟合更难的样本，其中一些可能只是 noise。

### AdaBoost / AdaBoost

AdaBoost（Adaptive Boosting）是第一个实用 boosting 算法。它可以配合任何 base learner，通常是 decision stumps（depth-1 trees）。

算法：

```
1. Initialize sample weights: w_i = 1/N for all i

2. For t = 1 to T:
   a. Train weak learner h_t on weighted data
   b. Compute weighted error:
      err_t = sum(w_i * I(h_t(x_i) != y_i)) / sum(w_i)
   c. Compute model weight:
      alpha_t = 0.5 * ln((1 - err_t) / err_t)
   d. Update sample weights:
      w_i = w_i * exp(-alpha_t * y_i * h_t(x_i))
   e. Normalize weights to sum to 1

3. Final prediction: H(x) = sign(sum(alpha_t * h_t(x)))
```

误差更低的模型得到更高 alpha。被误分类的样本权重会升高，让下一个模型重点关注它们。

### Gradient Boosting / 梯度提升

Gradient boosting 把 boosting 泛化到任意 loss functions。它不是重加权样本，而是让每个新模型拟合当前 ensemble 的 residuals（loss 的 negative gradient）。

```
1. Initialize: F_0(x) = argmin_c sum(L(y_i, c))

2. For t = 1 to T:
   a. Compute pseudo-residuals:
      r_i = -dL(y_i, F_{t-1}(x_i)) / dF_{t-1}(x_i)
   b. Fit a tree h_t to the residuals r_i
   c. Find optimal step size:
      gamma_t = argmin_gamma sum(L(y_i, F_{t-1}(x_i) + gamma * h_t(x_i)))
   d. Update:
      F_t(x) = F_{t-1}(x) + learning_rate * gamma_t * h_t(x)

3. Final prediction: F_T(x)
```

对 squared error loss，pseudo-residuals 就是实际 residuals：`r_i = y_i - F_{t-1}(x_i)`。每棵树实际上都在拟合前一个 ensemble 的错误。

Learning rate（shrinkage）控制每棵树贡献多少。更小 learning rates 需要更多 trees，但泛化更好。典型值：0.01 到 0.3。

### XGBoost: Why It Dominates Tabular Data / XGBoost：为什么它主导 tabular data

XGBoost（eXtreme Gradient Boosting）是在 gradient boosting 上加入工程优化，使其快速、准确、抗 overfitting：

- **Regularized objective：** 对 leaf weights 加 L1 和 L2 penalties，防止单棵树过度自信
- **Second-order approximation：** 同时使用 loss 的一阶和二阶导数，做出更好的 split decisions
- **Sparsity-aware splits：** 原生处理 missing values，在每次 split 中学习 missing data 的最佳方向
- **Column subsampling：** 像 random forests 一样在每次 split 采样 features 增加 diversity
- **Weighted quantile sketch：** 在 distributed data 上高效寻找连续 features 的 split points
- **Cache-aware block structure：** 针对 CPU cache lines 优化 memory layout

对 tabular data，XGBoost（及其后继 LightGBM）一直稳定胜过 neural networks。这短期不会改变。如果数据是行列形式的表，先试 gradient boosting。

### Stacking (Meta-Learning) / Stacking（元学习）

Stacking 把多个 base models 的 predictions 当作 meta-learner 的 features。

```mermaid
flowchart TD
    D[Training Data] --> M1[Model 1: Random Forest]
    D --> M2[Model 2: SVM]
    D --> M3[Model 3: Logistic Regression]

    M1 --> P1[Predictions 1]
    M2 --> P2[Predictions 2]
    M3 --> P3[Predictions 3]

    P1 --> META[Meta-Learner]
    P2 --> META
    P3 --> META

    META --> F[Final Prediction]
```

Meta-learner 会学习在什么输入上该信任哪个 base model。如果 random forest 在某些区域更好，SVM 在另一些区域更好，meta-learner 会学会相应路由。

为避免 data leakage，base model predictions 必须通过 training set 上的 cross-validation 生成。绝不能在同一份数据上既训练 base models，又生成 meta-features。

### Voting / 投票

最简单的 ensemble。直接组合 predictions。

- **Hard voting：** 对 class labels 做 majority vote。
- **Soft voting：** 平均 predicted probabilities，选择平均概率最高的类别。通常更好，因为它利用了 confidence 信息。

## Build It / 动手构建

### Step 1: Decision Stump (Base Learner) / 第 1 步：Decision stump（base learner）

`code/ensembles.py` 从零实现所有内容。我们从 decision stump 开始：只有一个 split 的 tree。

```python
class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        best_error = float("inf")

        for f in range(n_features):
            thresholds = np.unique(X[:, f])
            for thresh in thresholds:
                for polarity in [1, -1]:
                    pred = np.ones(n_samples)
                    pred[polarity * X[:, f] < polarity * thresh] = -1
                    error = np.sum(weights[pred != y])
                    if error < best_error:
                        best_error = error
                        self.feature_idx = f
                        self.threshold = thresh
                        self.polarity = polarity

    def predict(self, X):
        n = X.shape[0]
        pred = np.ones(n)
        idx = self.polarity * X[:, self.feature_idx] < self.polarity * self.threshold
        pred[idx] = -1
        return pred
```

### Step 2: AdaBoost from Scratch / 第 2 步：从零实现 AdaBoost

```python
class AdaBoostScratch:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []

    def fit(self, X, y):
        n = X.shape[0]
        weights = np.full(n, 1 / n)

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y, weights)
            pred = stump.predict(X)

            err = np.sum(weights[pred != y])
            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)
            weights *= np.exp(-alpha * y * pred)
            weights /= weights.sum()

            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        total = sum(a * s.predict(X) for a, s in zip(self.alphas, self.stumps))
        return np.sign(total)
```

### Step 3: Gradient Boosting from Scratch / 第 3 步：从零实现 Gradient Boosting

```python
class GradientBoostingScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        current_pred = np.full(len(y), self.initial_pred)

        for _ in range(self.n_estimators):
            residuals = y - current_pred
            tree = SimpleRegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            update = tree.predict(X)
            current_pred += self.lr * update
            self.trees.append(tree)

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred
```

### Step 4: Compare against sklearn / 第 4 步：与 sklearn 比较

代码会验证 from-scratch implementations 与 sklearn 的 `AdaBoostClassifier` 和 `GradientBoostingClassifier` 产生相近 accuracy，并把所有方法并排比较。

## Use It / 应用它

### When to Use Each Method / 何时使用每种方法

| Method / 方法 | Reduces / 降低 | Best for / 适合 | Watch out for / 注意 |
|--------|---------|----------|---------------|
| Bagging / Random Forest | Variance | 噪声数据、许多 features | 不能解决 bias |
| AdaBoost | Bias | 干净数据、简单 base learners | 对 outliers 和 noise 敏感 |
| Gradient Boosting | Bias | Tabular data、比赛 | 训练慢，缺少 tuning 时容易 overfit |
| XGBoost / LightGBM | Both | 生产 tabular ML | Hyperparameters 很多 |
| Stacking | Both | 追求最后 1-2% accuracy | 复杂，meta-learner 有 overfitting 风险 |
| Voting | Variance | 快速组合 diverse models | 只有模型足够 diverse 才有帮助 |

### The Production Stack for Tabular Data / Tabular data 的生产组合

对多数 tabular prediction problems，尝试顺序如下：

1. 使用默认参数的 **LightGBM or XGBoost**
2. 调 n_estimators、learning_rate、max_depth、min_child_weight
3. 如果需要最后 0.5%，用 3-5 个 diverse models 构建 stacking ensemble
4. 全程使用 cross-validation

尽管相关研究不断出现，neural networks 在 tabular data 上几乎总是输给 gradient boosting。TabNet、NODE 等架构偶尔能接近，但很少击败调好的 XGBoost。

## Ship It / 交付它

本课会产出 `outputs/prompt-ensemble-selector.md`：一个帮助你根据数据集选择合适 ensemble method 的 prompt。描述你的数据（规模、feature types、noise level、class balance）和要解决的问题，它会走一遍 decision checklist，推荐方法，给出起始 hyperparameters，并提示该方法的常见错误。还会产出 `outputs/skill-ensemble-builder.md`，其中包含完整选择指南。

## Exercises / 练习

1. 修改 AdaBoost 实现，记录每一轮后的 training accuracy。绘制 accuracy vs number of estimators。它什么时候收敛？

2. 通过给 regression tree 添加 random feature subsampling，从零实现 random forest。训练 100 棵树，`max_features=sqrt(n_features)`，并平均 predictions。与单棵树比较 variance reduction。

3. 在 gradient boosting 实现中添加 early stopping：每轮后跟踪 validation loss，如果连续 10 轮没有改善就停止。它实际需要多少棵树？

4. 构建一个 stacking ensemble，包含三个 base models（logistic regression、decision tree、k-nearest neighbors）和一个 logistic regression meta-learner。用 5-fold cross-validation 生成 meta-features。与每个 base model 单独比较。

5. 在同一数据集上用默认参数运行 XGBoost。比较它与 from-scratch gradient boosting 的 accuracy。记录两者耗时。速度差距有多大？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Bagging | “Train on random subsets” | Bootstrap aggregating：在 bootstrap samples 上训练模型，平均 predictions 来降低 variance |
| Boosting | “Focus on hard examples” | 顺序训练模型，每个模型修正当前 ensemble 的错误，从而降低 bias |
| AdaBoost | “Reweight the data” | 通过 sample weight updates 实现 boosting；misclassified points 在下一轮权重更高 |
| Gradient boosting | “Fit the residuals” | 通过拟合 loss function 的 negative gradient 来 boosting |
| XGBoost | “The Kaggle weapon” | 带 regularization、second-order optimization 和系统级加速技巧的 gradient boosting |
| Stacking | “Models on top of models” | 把 base models 的 predictions 作为 meta-learner 的输入 features |
| Random forest | “Many randomized trees” | 对 decision trees 做 bagging，并在每次 split 加入 random feature subsampling 来增加 diversity |
| Ensemble diversity | “Make different mistakes” | Ensemble 要提升效果，模型错误必须不高度相关 |
| Out-of-bag error | “Free validation” | Bootstrap draw 中未出现的样本（约 36.8%）可作为无需 holdout 的 validation set |

## Further Reading / 延伸阅读

- [Schapire & Freund: Boosting: Foundations and Algorithms](https://mitpress.mit.edu/9780262526036/) -- AdaBoost 创始人写的书
- [Friedman: Greedy Function Approximation: A Gradient Boosting Machine (2001)](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) -- gradient boosting 原始论文
- [Chen & Guestrin: XGBoost (2016)](https://arxiv.org/abs/1603.02754) -- XGBoost 论文
- [Wolpert: Stacked Generalization (1992)](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231) -- stacking 原始论文
- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html) -- 实用参考
