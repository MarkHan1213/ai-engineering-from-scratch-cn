# Probability and Distributions / 概率与分布

> 概率是 AI 表达不确定性的语言。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 1, Lessons 01-04
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 从零为 Bernoulli、categorical、Poisson、uniform 和 normal distributions 实现 PMFs 与 PDFs
- 计算 expected value、variance，并用 Central Limit Theorem 解释为什么 Gaussians 如此常见
- 用数值稳定技巧（subtract max logit）构建 softmax 和 log-softmax functions
- 从 logits 计算 cross-entropy loss，并把它连接到 negative log-likelihood

## The Problem / 问题

一个 classifier 输出 `[0.03, 0.91, 0.06]`。一个 language model 会从 50,000 个候选词里选择下一个词。一个 diffusion model 通过从学到的 distributions 中采样来生成图像。这些都是概率在发挥作用。

模型做出的每个预测都是 probability distribution。每个 loss function 都在衡量预测分布与真实分布有多远。每个训练步骤都在调整参数，让一个分布更像另一个分布。没有概率，你读不懂 ML 论文，调不动模型，也无法理解为什么 training loss 会变成 NaN。

## The Concept / 概念

### Events, Sample Spaces, and Probability / 事件、样本空间与概率

Sample space S 是所有可能结果的集合。Event 是 sample space 的一个子集。Probability 会把 events 映射到 0 到 1 之间的数。

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

三个公理定义了全部概率论：
1. 对任意 event A，P(A) >= 0
2. P(S) = 1，也就是总会发生某个结果
3. 当 A 和 B 不可能同时发生时，P(A or B) = P(A) + P(B)

其他所有内容，包括 Bayes' theorem、expectations 和 distributions，都从这三条规则推出。

### Conditional Probability and Independence / 条件概率与独立性

P(A|B) 表示在 B 已经发生的条件下 A 发生的概率。

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

如果知道一个事件并不会提供另一个事件的信息，这两个事件就是独立的：

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

抛硬币是独立的。不放回抽牌不是。

### Probability Mass Functions vs Probability Density Functions / 概率质量函数与概率密度函数

离散随机变量有 probability mass function（PMF）。每个结果都有一个可以直接读出的具体概率。

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续随机变量有 probability density function（PDF）。单个点处的 density 不是概率。概率来自对某个区间上的 density 积分。

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

这个区别在 ML 中很重要。Classification outputs 是 PMFs（离散选择）。VAE latent spaces 使用 PDFs（连续变量）。

### Common Distributions / 常见分布

**Bernoulli：** 一次试验，两个结果。用于建模 binary classification。

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical：** 一次试验，k 个结果。用于建模 multi-class classification（softmax output）。

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform：** 所有结果等可能。常用于 random initialization。

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian)：** 钟形曲线。由 mean（mu）和 variance（sigma^2）参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson：** 固定时间区间内稀有事件的计数。用于建模事件率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Expected Value and Variance / 期望与方差

Expected value 是按概率加权的平均结果。

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Variance 衡量围绕 mean 的分散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在 ML 中，expected value 会以 loss function 的形式出现，也就是数据分布上的平均 loss。Variance 告诉你模型稳定性。Gradients 的 variance 很高，意味着训练噪声大。

### Joint and Marginal Distributions / 联合分布与边缘分布

Joint distribution P(X, Y) 会一起描述两个随机变量。

Joint PMF 示例（X = weather，Y = umbrella）：

| | Y=0 (no umbrella) | Y=1 (umbrella) | Marginal P(X) |
|---|---|---|---|
| X=0 (sun) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (rain) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

Marginal distribution 会把另一个变量求和消去：

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

上表中的行合计和列合计就是 marginals。

### Why the Normal Distribution Shows Up Everywhere / 为什么正态分布到处出现

Central Limit Theorem：许多 independent random variables 的和（或平均值）会收敛到 normal distribution，不管原始分布是什么。

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

这就是为什么：
- 测量误差近似 normal，因为它由许多小的独立来源叠加而成
- 神经网络的 weight initializations 使用 normal distributions
- SGD 中的 gradient noise 近似 normal，因为它是许多 sample gradients 的和
- 在给定 mean 和 variance 时，normal distribution 是 maximum entropy distribution

### Log Probabilities / 对数概率

原始概率会造成数值问题。把很多很小的概率相乘，很快就会下溢为零。

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Log probabilities 可以解决这个问题。乘法会变成加法。

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

规则：
- log(a * b) = log(a) + log(b)
- log probabilities 永远 <= 0，因为 0 < P <= 1
- 越负表示越不可能
- Cross-entropy loss 是正确类别的 negative log probability

### Softmax as a Probability Distribution / Softmax 作为概率分布

神经网络输出 raw scores（logits）。Softmax 会把它们转换成合法 probability distribution。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Softmax trick：在 exponentiating 前减去最大 logit，防止 overflow。

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax 会把 softmax 和 log 组合起来，以获得数值稳定性。PyTorch 的 cross-entropy loss 内部就用这种做法。

### Sampling / 采样

Sampling 指的是从一个分布中抽取随机值。在 ML 中：
- Dropout 会随机采样要置零的 neurons
- Data augmentation 会采样随机变换
- Language models 会从预测分布中采样下一个 token
- Diffusion models 会采样噪声，再逐步去噪

从任意 distributions 采样需要 inverse transform sampling、rejection sampling 或 reparameterization trick（VAEs 中使用）这样的技术。

```figure
gaussian-pdf
```

## Build It / 动手构建

### Step 1: Probability basics / 第 1 步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Step 2: PMF and PDF from scratch / 第 2 步：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Step 3: Expected value and variance / 第 3 步：期望与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Step 4: Sampling from distributions / 第 4 步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Step 5: Softmax and log probabilities / 第 5 步：Softmax 与 log probabilities

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Step 6: Central Limit Theorem demonstration / 第 6 步：演示 Central Limit Theorem

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Step 7: Visualization / 第 7 步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

包含所有可视化的完整实现位于 `code/probability.py`。

## Use It / 应用它

用 NumPy 和 SciPy，上面的所有操作都可以写成一行：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已经从零构建了这些东西。现在你知道 library calls 背后在做什么。

## Ship It / 交付它

本课交付一套概率工具箱心智模型：用 PMF/PDF 描述分布，用 expected value 和 variance 描述总体行为，用 stable softmax/log-softmax 把 logits 转成可训练的概率目标。

## Exercises / 练习

1. 为 exponential distribution 实现 inverse transform sampling。采样 10,000 个值，并把 histogram 与真实 PDF 对比验证。

2. 为两个 loaded dice 构建 joint distribution table。计算 marginal distributions，并检查两个骰子是否 independent。

3. 当正确类别是 index 3 时，计算一个 5-class classifier 对 logits `[2.0, 0.5, -1.0, 3.0, 0.1]` 的 cross-entropy loss。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证答案。

4. 写一个函数，输入 log probabilities 列表，返回最可能的序列、总 log probability，以及等价 raw probability。用一句 50 个词的句子测试，每个词的 probability 都是 0.01。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Sample space | “所有可能性” | 实验所有可能结果组成的集合 S |
| PMF | “概率函数” | 给出每个离散结果精确概率的函数，总和为 1 |
| PDF | “概率曲线” | 连续变量的密度函数。对某个区间积分才能得到概率 |
| Conditional probability | “给定某事后的概率” | P(A\|B) = P(A and B) / P(B)。Bayesian thinking 和 Bayes' theorem 的基础 |
| Independence | “互不影响” | P(A and B) = P(A) * P(B)。知道一个事件，不会告诉你另一个事件的信息 |
| Expected value | “平均值” | 所有结果按概率加权求和。Loss function 就是一个 expected value |
| Variance | “有多分散” | 相对 mean 的期望平方偏差。High variance = noisy, unstable estimates |
| Normal distribution | “钟形曲线” | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。由于 CLT，它无处不在 |
| Central Limit Theorem | “平均值会变成正态分布” | 许多 independent samples 的 mean 会收敛到 normal distribution，不管原始来源是什么 |
| Joint distribution | “两个变量一起看” | P(X, Y) 描述 X 与 Y 所有结果组合的概率 |
| Marginal distribution | “把另一个变量求和消去” | P(X) = sum_y P(X, Y)。从 joint 中恢复某个变量的分布 |
| Log probability | “概率的 log” | log P(x)。把乘法变成加法，避免长序列中的 numerical underflow |
| Softmax | “把分数变成概率” | softmax(z_i) = exp(z_i) / sum(exp(z_j))。把 real-valued logits 映射为合法 probability distribution |
| Cross-entropy | “Loss function” | -sum(p_true * log(p_predicted))。衡量两个 distributions 有多不同。越低越好 |
| Logits | “模型原始输出” | Softmax 之前的 unnormalized scores。名字来自 logistic function |
| Sampling | “抽随机值” | 按某个 probability distribution 生成值。模型就是这样生成输出的 |

## Further Reading / 延伸阅读

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 用可视化证明解释为什么平均值会趋近正态
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 覆盖本课及更多内容的简明参考
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为什么数值稳定重要，以及如何做到
