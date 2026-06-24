# The Perceptron / 感知机

> 感知机是神经网络的原子。把它拆开，你会看到 weights、一个 bias，以及一次决策。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 1 (Linear Algebra Intuition)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 用 Python 从零实现一个 perceptron，包括 weight update rule 和 step activation function
- 解释为什么单个 perceptron 只能解决 linearly separable problem，并演示 XOR 失败案例
- 通过组合 OR、NAND 和 AND gate 构造 multi-layer perceptron 来解决 XOR
- 训练一个带 sigmoid activation 和 backpropagation 的两层网络，让它自动学会 XOR

## The Problem / 问题

你已经理解 vectors 和 dot products，也知道 matrix 会把 inputs 变换成 outputs。但机器到底如何*学会*该使用哪一种变换？

Perceptron 给出了答案。它是最简单的学习机器：取一些 inputs，乘以 weights，加上 bias，然后做一次二分类决策。接着再调整。就这么多。历史上所有 neural network 都是在把这个想法一层层叠起来。

理解 perceptron，就是理解代码里的“学习”到底是什么意思：不断调整数字，直到 output 匹配现实。

## The Concept / 概念

### One Neuron, One Decision / 一个神经元，一次决策

Perceptron 接收 n 个 inputs，把每个 input 乘以对应 weight，求和后加上 bias，再把结果送进 activation function。

```mermaid
graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["bias"] --> sum
    sum --> step["step(z)"]
    step --> out["output (0 or 1)"]
```

Step function 非常直接：如果 weighted sum plus bias >= 0，就输出 1。否则输出 0。

```
step(z) = 1  if z >= 0
           0  if z < 0
```

这是一个 linear classifier。Weights 和 bias 定义了一条线（在更高维里是 hyperplane），把 input space 分成两个区域。

### The Decision Boundary / 决策边界

对于两个 inputs，perceptron 会在 2D 空间里画出一条线：

```
  x2
  ┤
  │  Class 1        /
  │    (0)          /
  │                /
  │               / w1·x1 + w2·x2 + b = 0
  │              /
  │             /     Class 2
  │            /        (1)
  ┼───────────/──────────── x1
```

线的一侧输出 0，另一侧输出 1。训练就是移动这条线，直到它能正确分开类别。

### The Learning Rule / 学习规则

Perceptron learning rule 很简单：

```
For each training example (x, y_true):
    y_pred = predict(x)
    error = y_true - y_pred

    For each weight:
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error
```

如果预测正确，error = 0，什么都不变。如果它预测 0 但真实值应该是 1，weights 会增加。如果它预测 1 但真实值应该是 0，weights 会减小。Learning rate 控制每次调整的幅度。

### The XOR Problem / XOR 问题

问题会在这里暴露出来。看这些 logic gates：

```
AND gate:           OR gate:            XOR gate:
x1  x2  out         x1  x2  out         x1  x2  out
0   0   0           0   0   0           0   0   0
0   1   0           0   1   1           0   1   1
1   0   0           1   0   1           1   0   1
1   1   1           1   1   1           1   1   0
```

AND 和 OR 都是 linearly separable：你可以画一条线把 0 和 1 分开。XOR 不是。没有任何一条直线能把 [0,1] 和 [1,0] 同 [0,0] 和 [1,1] 分开。

```
AND (separable):        XOR (not separable):

  x2                      x2
  1 ┤  0     1            1 ┤  1     0
    │     /                 │
  0 ┤  0 / 0              0 ┤  0     1
    ┼──/──────── x1         ┼──────────── x1
       line works!          no single line works!
```

这是一个根本限制。单个 perceptron 只能解决 linearly separable problem。Minsky 和 Papert 在 1969 年证明了这一点，它几乎让 neural network 研究停滞了十年。

修复方式：把 perceptrons 堆叠成 layers。Multi-layer perceptron 可以把两个线性决策组合成一个非线性决策，从而解决 XOR。

```figure
perceptron-boundary
```

## Build It / 动手构建

### Step 1: The Perceptron class / 第 1 步：Perceptron class

```python
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")
```

### Step 2: Train on logic gates / 第 2 步：在 logic gates 上训练

```python
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")
```

### Step 3: Watch XOR fail / 第 3 步：观察 XOR 失败

```python
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
```

它永远不会收敛。这就是单个 perceptron 无法学习 XOR 的硬证据。

### Step 4: Solve XOR with two layers / 第 4 步：用两层解决 XOR

技巧是：XOR = (x1 OR x2) AND NOT (x1 AND x2)。组合三个 perceptrons：

```mermaid
graph LR
    x1["x1"] --> OR["OR neuron"]
    x1 --> NAND["NAND neuron"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND neuron"]
    NAND --> AND
    AND --> out["output"]
```

```python
def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")
```

四种情况全部正确。把 perceptrons 堆叠成 layers，可以创造出单个 perceptron 无法产生的 decision boundaries。

### Step 5: Train a Two-Layer Network / 第 5 步：训练一个两层网络

第 4 步是手动写死 weights。它能解决 XOR，但真实问题里你事先不知道正确的 weights。修复方式是：把 step function 换成 sigmoid，并通过 backpropagation 自动学习 weights。

```python
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
```

```python
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")
```

相比第 4 步，这里有两个关键差异。第一，sigmoid 取代了 step function，它是平滑的，因此 gradients 存在。第二，`train` method 会把 error 从 output layer 反向传播到 hidden layer，并按每个 weight 对 error 的贡献比例调整它。这就是 20 行代码里的 backpropagation。

这也是通往 Lesson 03 的桥。`d_output` 和 `hidden_deltas` 背后的数学，就是把 chain rule 应用于 network graph。下一课会正式推导它。

## Use It / 应用它

你刚刚从零构建的东西，用一个 import 就能得到：

```python
from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])
```

五行代码。你的 30 行 `Perceptron` class 做的是同一件事。sklearn 版本增加了 convergence checks、multiple loss functions 和 sparse input support，但核心循环完全一致：weighted sum、step function、在 error 上更新 weight。

真正的差异出现在规模上。生产网络会改变这些部分：

- Step function 会变成 sigmoid、ReLU 或其他 smooth activations
- Weights 会通过 backpropagation 自动学习（Lesson 03）
- Layers 会变深：3 层、10 层、100+ 层
- 同一个原则仍然成立：每一层都从上一层的 outputs 中创造新的 features

单个 perceptron 只能画直线。把它们堆叠起来，你就能画出任意形状。

## Ship It / 交付它

本课产出：
- `outputs/skill-perceptron.md` - 一个说明何时需要 single-layer 与 multi-layer architectures 的 skill

## Exercises / 练习

1. 在 NAND gate 上训练一个 perceptron（NAND 是 universal gate，任何 logic circuit 都可以由 NAND 构建）。验证它的 weights 和 bias 能形成有效的 decision boundary。
2. 修改 Perceptron class，让它在每个 epoch 跟踪 decision boundary（w1*x1 + w2*x2 + b = 0）。打印它在 AND gate 训练过程中如何移动。
3. 构建一个 3-input perceptron：只有 3 个 inputs 中至少 2 个为 1 时才输出 1（majority vote function）。它是 linearly separable 吗？为什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Perceptron | “一个假神经元” | 一个 linear classifier：对 inputs 和 weights 做 dot product，加上 bias，再通过 step function |
| Weight | “某个 input 有多重要” | 一个 multiplier，用来缩放每个 input 对决策的贡献 |
| Bias | “阈值” | 一个会平移 decision boundary 的常数，让 perceptron 即使在零输入时也能触发 |
| Activation function | “压缩数值的东西” | 加权求和后应用的函数；perceptron 使用 step function，现代网络常用 sigmoid/ReLU |
| Linearly separable | “能在它们中间画一条线” | 一个可以用单个 hyperplane 完美分开类别的数据集 |
| XOR problem | “Perceptrons 做不了的那个问题” | 证明 single-layer networks 无法学习 non-linearly-separable functions 的例子 |
| Decision boundary | “分类器切换的位置” | 划分 input space 中两个类别的 hyperplane w*x + b = 0 |
| Multi-layer perceptron | “真正的神经网络” | 按 layers 堆叠的 perceptrons，每一层的 output 会作为下一层的 input |

## Further Reading / 延伸阅读

- Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958) -- 开创这一方向的原始论文
- Minsky & Papert, "Perceptrons" (1969) -- 证明 XOR 无法由 single-layer networks 解决、并让 perceptron 研究沉寂十年的那本书
- Michael Nielsen, "Neural Networks and Deep Learning", Chapter 1 (http://neuralnetworksanddeeplearning.com/) -- 免费在线资源，对 perceptrons 如何组合成 networks 的可视化解释非常清楚
