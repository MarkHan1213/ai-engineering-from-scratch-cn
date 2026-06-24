# Logistic Regression / 逻辑回归

> Logistic regression 把一条直线压成 S 曲线，用概率回答是/否问题。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 2 Lesson 1-2 (What Is ML, Linear Regression)
**Time / 时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 使用 sigmoid function 和 binary cross-entropy loss 从零实现 logistic regression
- 计算并解释 binary classification 中的 precision、recall、F1 score 和 confusion matrix
- 解释为什么 MSE 不适合 classification，以及为什么 binary cross-entropy 会产生 convex cost surface
- 构建 multi-class classification 的 softmax regression 模型，并评估 threshold tuning 的权衡

## The Problem / 问题

你想根据肿瘤大小预测它是 malignant 还是 benign。你尝试 linear regression，它输出 0.3、1.7、-0.5 这样的数字。这些数字是什么意思？1.7 是“非常恶性”吗？-0.5 是“非常良性”吗？Linear regression 输出的是无界数值。Classification 需要 0 到 1 之间的有界概率，以及清晰的决策：是或否。

Logistic regression 解决这个问题。它使用同样的线性组合（wx + b），再通过 sigmoid function，把任意数字压缩到 (0, 1) 范围。输出就是概率。你设定一个 threshold（通常是 0.5），然后做决策。

这是实践中最常用的算法之一。尽管名字里有 regression，logistic regression 是 classification 算法，不是 regression 算法。这个名字来自它使用的 logistic（sigmoid）function。

## The Concept / 概念

### Why Linear Regression Fails for Classification / 为什么 linear regression 不适合分类

想象你根据学习时长预测 pass/fail（1/0）。Linear regression 会拟合一条穿过数据的线：

```
hours:  1   2   3   4   5   6   7   8   9   10
actual: 0   0   0   0   1   1   1   1   1   1
```

线性拟合可能会在第 1 小时给出 -0.2，在第 10 小时给出 1.3。这些值不是概率。它们会低于 0，也会高于 1。更糟的是，一个离群点（有人学习 50 小时）会把整条线拉偏，改变所有人的预测。

Classification 需要一个函数满足：
- 输出值在 0 到 1 之间（概率）
- 能形成清晰的转换（decision boundary）
- 不会被远离边界的 outliers 过度扭曲

### The Sigmoid Function / Sigmoid 函数

Sigmoid function 正好做到这一点：

```
sigmoid(z) = 1 / (1 + e^(-z))
```

性质：
- z 很大且为正时，sigmoid(z) 接近 1
- z 很大且为负时，sigmoid(z) 接近 0
- z = 0 时，sigmoid(z) = 0.5
- 输出永远在 0 和 1 之间
- 函数处处平滑且可微

它的导数形式很方便：sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))。这让 gradient 计算很高效。

### Logistic Regression = Linear Model + Sigmoid / Logistic regression = 线性模型 + sigmoid

模型先计算 z = wx + b（和 linear regression 一样），再应用 sigmoid：

```mermaid
flowchart LR
    X[Input features x] --> L["Linear: z = wx + b"]
    L --> S["Sigmoid: p = 1/(1+e^-z)"]
    S --> D{"p >= 0.5?"}
    D -->|Yes| P[Predict 1]
    D -->|No| N[Predict 0]
```

输出 p 被解释为 P(y=1 | x)，即输入属于 class 1 的概率。Decision boundary 位于 wx + b = 0 的地方，此时 sigmoid 输出正好是 0.5。

### Binary Cross-Entropy Loss / 二元交叉熵损失

Logistic regression 不能使用 MSE。带 sigmoid 的 MSE 会产生非凸 cost surface，并带有多个 local minima。应该使用 binary cross-entropy（log loss）：

```
Loss = -(1/n) * sum(y * log(p) + (1-y) * log(1-p))
```

为什么它有效：
- 当 y=1 且 p 接近 1：log(1) = 0，所以 loss 接近 0（正确，低 cost）
- 当 y=1 且 p 接近 0：log(0) 接近负无穷，所以 loss 很大（错误，高 cost）
- 当 y=0 且 p 接近 0：log(1) = 0，所以 loss 接近 0（正确，低 cost）
- 当 y=0 且 p 接近 1：log(0) 接近负无穷，所以 loss 很大（错误，高 cost）

对 logistic regression 来说，这个 loss function 是 convex 的，因此保证只有一个 global minimum。

### Gradient Descent for Logistic Regression / Logistic regression 的梯度下降

Sigmoid 加 binary cross-entropy 的 gradients 形式很简洁：

```
dL/dw = (1/n) * sum((p - y) * x)
dL/db = (1/n) * sum(p - y)
```

它们看起来和 linear regression 的 gradients 一样。区别是 p = sigmoid(wx + b)，而不是 p = wx + b。Sigmoid 引入了非线性，但 gradient update rule 保持不变。

```mermaid
flowchart TD
    A[Initialize w=0, b=0] --> B[Forward pass: z = wx+b, p = sigmoid z]
    B --> C[Compute loss: binary cross-entropy]
    C --> D["Compute gradients: dw = (1/n) * sum((p-y)*x)"]
    D --> E[Update: w = w - lr*dw, b = b - lr*db]
    E --> F{Converged?}
    F -->|No| B
    F -->|Yes| G[Model trained]
```

### The Decision Boundary / 决策边界

对于二维输入（两个 features），decision boundary 是满足下面方程的直线：

```
w1*x1 + w2*x2 + b = 0
```

一侧的点被分类为 1，另一侧被分类为 0。Logistic regression 总会产生线性 decision boundary。如果你需要弯曲边界，要么添加 polynomial features，要么使用 nonlinear model。

### Multi-Class Classification with Softmax / 用 softmax 做多分类

Binary logistic regression 处理两个类别。对于 k 个类别，使用 softmax function：

```
softmax(z_i) = e^(z_i) / sum(e^(z_j) for all j)
```

每个类别都有自己的 weight vector。模型为每个类别计算一个 score z_i，然后 softmax 把 scores 转成加和为 1 的 probabilities。预测类别就是概率最高的类别。

Loss function 变为 categorical cross-entropy：

```
Loss = -(1/n) * sum(sum(y_k * log(p_k)))
```

其中 y_k 对真实类别为 1，对其他类别为 0（one-hot encoding）。

### Evaluation Metrics / 评估指标

只看 accuracy 不够。如果数据集中 95% 是 negative，5% 是 positive，一个永远预测 negative 的模型也有 95% accuracy，但完全没用。

**Confusion Matrix / 混淆矩阵**：

| | Predicted Positive / 预测为正 | Predicted Negative / 预测为负 |
|---|---|---|
| Actually Positive / 实际为正 | True Positive (TP) | False Negative (FN) |
| Actually Negative / 实际为负 | False Positive (FP) | True Negative (TN) |

**Precision / 精确率**：所有预测为 positive 的样本中，有多少真的 positive？
```
Precision = TP / (TP + FP)
```

**Recall / 召回率**（Sensitivity）：所有实际 positive 的样本中，我们抓住了多少？
```
Recall = TP / (TP + FN)
```

**F1 Score / F1 分数**：precision 和 recall 的 harmonic mean，用来平衡两者。
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

优先级选择：
- **Precision**：false positives 代价高时（垃圾邮件过滤，不希望拦截正常邮件）
- **Recall**：false negatives 代价高时（癌症筛查，不希望漏掉肿瘤）
- **F1**：需要一个平衡指标时

```figure
logistic-sigmoid
```

## Build It / 动手构建

### Step 1: Sigmoid function and data generation / 第 1 步：Sigmoid function 与数据生成

```python
import random
import math

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))


random.seed(42)
N = 200
X = []
y = []

for _ in range(N // 2):
    X.append([random.gauss(2, 1), random.gauss(2, 1)])
    y.append(0)

for _ in range(N // 2):
    X.append([random.gauss(5, 1), random.gauss(5, 1)])
    y.append(1)

combined = list(zip(X, y))
random.shuffle(combined)
X, y = zip(*combined)
X = list(X)
y = list(y)

print(f"Generated {N} samples (2 classes, 2 features)")
print(f"Class 0 center: (2, 2), Class 1 center: (5, 5)")
print(f"First 5 samples:")
for i in range(5):
    print(f"  Features: [{X[i][0]:.2f}, {X[i][1]:.2f}], Label: {y[i]}")
```

### Step 2: Logistic regression from scratch / 第 2 步：从零实现 logistic regression

```python
class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.loss_history = []

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def compute_loss(self, X, y):
        n = len(y)
        total = 0.0
        for i in range(n):
            p = self.predict_proba(X[i])
            p = max(1e-15, min(1 - 1e-15, p))
            total += y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p)
        return -total / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n):
                p = self.predict_proba(X[i])
                error = p - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n)
            self.bias -= self.lr * (db / n)
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | w: [{self.weights[0]:.3f}, {self.weights[1]:.3f}] | b: {self.bias:.3f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


split = int(0.8 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("\n=== Training Logistic Regression ===")
model = LogisticRegression(n_features=2, learning_rate=0.1)
model.fit(X_train, y_train, epochs=1000, print_every=200)

print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.4f}")
print(f"Test accuracy:  {model.accuracy(X_test, y_test):.4f}")
print(f"Weights: [{model.weights[0]:.4f}, {model.weights[1]:.4f}]")
print(f"Bias: {model.bias:.4f}")
```

### Step 3: Confusion matrix and metrics from scratch / 第 3 步：从零实现 confusion matrix 和 metrics

```python
class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        self.tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        self.fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        self.fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_confusion_matrix(self):
        print(f"\n  Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Pos   Neg")
        print(f"  Actual Pos     {self.tp:4d}  {self.fn:4d}")
        print(f"  Actual Neg     {self.fp:4d}  {self.tn:4d}")

    def print_report(self):
        self.print_confusion_matrix()
        print(f"\n  Accuracy:  {self.accuracy():.4f}")
        print(f"  Precision: {self.precision():.4f}")
        print(f"  Recall:    {self.recall():.4f}")
        print(f"  F1 Score:  {self.f1():.4f}")


y_pred_test = [model.predict(x) for x in X_test]
print("\n=== Classification Report (Test Set) ===")
metrics = ClassificationMetrics(y_test, y_pred_test)
metrics.print_report()
```

### Step 4: Decision boundary analysis / 第 4 步：决策边界分析

```python
print("\n=== Decision Boundary ===")
w1, w2 = model.weights
b = model.bias
print(f"Decision boundary: {w1:.4f}*x1 + {w2:.4f}*x2 + {b:.4f} = 0")
if abs(w2) > 1e-10:
    print(f"Solved for x2:     x2 = {-w1/w2:.4f}*x1 + {-b/w2:.4f}")

print("\nSample predictions near the boundary:")
test_points = [
    [3.0, 3.0],
    [3.5, 3.5],
    [4.0, 4.0],
    [2.5, 2.5],
    [5.0, 5.0],
]
for point in test_points:
    prob = model.predict_proba(point)
    pred = model.predict(point)
    print(f"  [{point[0]}, {point[1]}] -> prob={prob:.4f}, class={pred}")
```

### Step 5: Multi-class with softmax / 第 5 步：用 softmax 做多分类

```python
class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.biases = [0.0] * n_classes

    def softmax(self, scores):
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [e / total for e in exp_scores]

    def predict_proba(self, x):
        scores = [
            sum(self.weights[k][j] * x[j] for j in range(self.n_features)) + self.biases[k]
            for k in range(self.n_classes)
        ]
        return self.softmax(scores)

    def predict(self, x):
        probs = self.predict_proba(x)
        return probs.index(max(probs))

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        for epoch in range(epochs):
            grad_w = [[0.0] * self.n_features for _ in range(self.n_classes)]
            grad_b = [0.0] * self.n_classes
            total_loss = 0.0
            for i in range(n):
                probs = self.predict_proba(X[i])
                for k in range(self.n_classes):
                    target = 1.0 if y[i] == k else 0.0
                    error = probs[k] - target
                    for j in range(self.n_features):
                        grad_w[k][j] += error * X[i][j]
                    grad_b[k] += error
                true_prob = max(probs[y[i]], 1e-15)
                total_loss -= math.log(true_prob)
            for k in range(self.n_classes):
                for j in range(self.n_features):
                    self.weights[k][j] -= self.lr * (grad_w[k][j] / n)
                self.biases[k] -= self.lr * (grad_b[k] / n)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Loss: {total_loss / n:.4f}")
        return self

    def accuracy(self, X, y):
        correct = sum(1 for i in range(len(y)) if self.predict(X[i]) == y[i])
        return correct / len(y)


random.seed(42)
X_3class = []
y_3class = []

centers = [(1, 1), (5, 1), (3, 5)]
for label, (cx, cy) in enumerate(centers):
    for _ in range(50):
        X_3class.append([random.gauss(cx, 0.8), random.gauss(cy, 0.8)])
        y_3class.append(label)

combined = list(zip(X_3class, y_3class))
random.shuffle(combined)
X_3class, y_3class = zip(*combined)
X_3class = list(X_3class)
y_3class = list(y_3class)

split_3 = int(0.8 * len(X_3class))
X_train_3 = X_3class[:split_3]
y_train_3 = y_3class[:split_3]
X_test_3 = X_3class[split_3:]
y_test_3 = y_3class[split_3:]

print("\n=== Multi-class Softmax Regression (3 classes) ===")
softmax_model = SoftmaxRegression(n_features=2, n_classes=3, learning_rate=0.1)
softmax_model.fit(X_train_3, y_train_3, epochs=1000, print_every=200)
print(f"\nTrain accuracy: {softmax_model.accuracy(X_train_3, y_train_3):.4f}")
print(f"Test accuracy:  {softmax_model.accuracy(X_test_3, y_test_3):.4f}")

print("\nSample predictions:")
for i in range(5):
    probs = softmax_model.predict_proba(X_test_3[i])
    pred = softmax_model.predict(X_test_3[i])
    print(f"  True: {y_test_3[i]}, Predicted: {pred}, Probs: [{', '.join(f'{p:.3f}' for p in probs)}]")
```

### Step 6: Threshold tuning / 第 6 步：阈值调优

```python
print("\n=== Threshold Tuning ===")
print("Default threshold: 0.5. Adjusting the threshold trades precision for recall.\n")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)

for t in thresholds:
    y_pred_t = [1 if model.predict_proba(x) >= t else 0 for x in X_test]
    m = ClassificationMetrics(y_test, y_pred_t)
    print(f"{t:>10.1f} {m.accuracy():>10.4f} {m.precision():>10.4f} {m.recall():>10.4f} {m.f1():>10.4f}")
```

## Use It / 应用它

下面用 scikit-learn 完成同样的事。

```python
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X_0 = np.random.randn(100, 2) + [2, 2]
X_1 = np.random.randn(100, 2) + [5, 5]
X_sk = np.vstack([X_0, X_1])
y_sk = np.array([0] * 100 + [1] * 100)

X_tr, X_te, y_tr, y_te = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

lr = SklearnLR()
lr.fit(X_tr_sc, y_tr)
y_pred = lr.predict(X_te_sc)

print("=== Scikit-learn Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall:    {recall_score(y_te, y_pred):.4f}")
print(f"F1:        {f1_score(y_te, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_te, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_te, y_pred)}")
```

你的 from-scratch 实现会产生相同的 decision boundary 和 metrics。Scikit-learn 还加入了 solver options（liblinear、lbfgs、saga）、自动 regularization、multi-class strategies（one-vs-rest、multinomial）和数值稳定性优化。

## Ship It / 交付它

本课会产出：
- `code/logistic_regression.py` - 从零实现的 logistic regression 与 metrics

## Exercises / 练习

1. 生成一个不是 linearly separable 的数据集（例如两个同心圆）。训练 logistic regression 并观察它的失败。然后添加 polynomial features（x1^2、x2^2、x1*x2）重新训练，展示 accuracy 的提升。
2. 为 3-class softmax model 实现 multi-class confusion matrix。计算每个类别的 precision 和 recall。哪个类别最难分类？
3. 从零构建 ROC curve。对 0 到 1 的 100 个 threshold values，计算 true positive rate 和 false positive rate。用 trapezoidal rule 计算 AUC（area under the curve）。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Logistic regression | “Regression for classification” | 一个线性模型后接 sigmoid function，输出类别概率 |
| Sigmoid function | “The S-curve” | 函数 1/(1+e^(-z))，把任意实数映射到 (0, 1) |
| Binary cross-entropy | “Log loss” | 损失函数 -[y*log(p) + (1-y)*log(1-p)]，会严厉惩罚自信但错误的预测 |
| Decision boundary | “The dividing line” | 模型输出概率等于 0.5 的表面，用来分隔预测类别 |
| Softmax | “Multi-class sigmoid” | 把 scores 向量转换成总和为 1 的 probabilities 的函数 |
| Precision | “How many selected are relevant” | TP / (TP + FP)，positive predictions 中真正 positive 的比例 |
| Recall | “How many relevant are selected” | TP / (TP + FN)，actual positives 中被模型正确识别的比例 |
| F1 score | “Balanced accuracy” | Precision 和 recall 的 harmonic mean：2*P*R / (P+R) |
| Confusion matrix | “The error breakdown” | 展示每个类别对的 TP、TN、FP、FN 数量的表 |
| Threshold | “The cutoff” | 高于该 probability value 时模型预测 class 1（默认 0.5，可调） |
| One-hot encoding | “Binary columns for categories” | 用一个只有第 k 位为 1、其他位置为 0 的向量表示 class k |
| Categorical cross-entropy | “Multi-class log loss” | 使用 one-hot encoded labels 的 k 类 binary cross-entropy 扩展 |
