# Chain Rule & Automatic Differentiation / 链式法则与自动微分

> 链式法则是每个会学习的神经网络背后的引擎。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 1, Lesson 04 (Derivatives & Gradients)
**时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 构建一个最小 autograd engine（Value class），记录操作并通过 reverse-mode autodiff 计算 gradients
- 使用 topological sort，在 computation graph 上实现 forward pass 和 backward pass
- 只用从零实现的 autograd engine，构建并训练一个解决 XOR 的 multi-layer perceptron
- 用 numerical finite differences 做 gradient checking，验证 autodiff 正确性

## The Problem / 问题

你已经能计算简单函数的导数。但神经网络不是简单函数。它由成百上千个函数组合而成：matrix multiply、add bias、apply activation、再次 matrix multiply、softmax、cross-entropy loss。输出是一个函数的函数的函数。

要训练网络，你需要知道 loss 对每一个权重的 gradient。对数百万参数手算是不可能的。用 numerical finite differences 计算又太慢。

链式法则给出数学基础。Automatic differentiation 给出算法。二者结合后，你就能以接近一次 forward pass 的时间复杂度，穿过任意函数组合计算精确 gradients。

PyTorch、TensorFlow 和 JAX 都是这样工作的。你会从零构建一个微型版本。

## The Concept / 概念

### The Chain Rule / 链式法则

如果 `y = f(g(x))`，那么 `y` 对 `x` 的导数是：

```
dy/dx = dy/dg * dg/dx = f'(g(x)) * g'(x)
```

沿着函数链把导数相乘。每一环都会贡献自己的 local derivative。

例子：`y = sin(x^2)`

```
g(x) = x^2       g'(x) = 2x
f(g) = sin(g)     f'(g) = cos(g)

dy/dx = cos(x^2) * 2x
```

更深的组合也只是把链拉长：

```
y = f(g(h(x)))

dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```

神经网络中的每一层，都是这条链上的一环。

### Computational Graphs / 计算图

Computational graph 会把 chain rule 可视化。每个操作都是一个节点。数据沿图向前流动，gradients 沿图向后流动。

**Forward pass（计算值）：**

```mermaid
graph TD
    x1["x1 = 2"] --> mul["* (multiply)"]
    x2["x2 = 3"] --> mul
    mul -->|"a = 6"| add["+ (add)"]
    b["b = 1"] --> add
    add -->|"c = 7"| relu["relu"]
    relu -->|"y = 7"| y["output y"]
```

**Backward pass（计算 gradients）：**

```mermaid
graph TD
    dy["dy/dy = 1"] -->|"relu'(c)=1 since c>0"| dc["dy/dc = 1"]
    dc -->|"dc/da = 1"| da["dy/da = 1"]
    dc -->|"dc/db = 1"| db["dy/db = 1"]
    da -->|"da/dx1 = x2 = 3"| dx1["dy/dx1 = 3"]
    da -->|"da/dx2 = x1 = 2"| dx2["dy/dx2 = 2"]
```

Backward pass 会在每个节点应用 chain rule，把 gradients 从输出传播回输入。

### Forward Mode vs Reverse Mode / 前向模式与反向模式

穿过一张图应用 chain rule 有两种方式。

**Forward mode** 从输入开始，把 derivatives 向前推。它先计算 `dx/dx = 1`，再穿过每个操作传播。适合输入少、输出多的场景。

```
Forward mode: seed dx/dx = 1, propagate forward

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```

**Reverse mode** 从输出开始，把 gradients 向后拉。它先计算 `dy/dy = 1`，再按反向穿过每个操作。适合输入多、输出少的场景。

```
Reverse mode: seed dy/dy = 1, propagate backward

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```

神经网络有数百万个输入（权重），但通常只有一个输出（loss）。Reverse mode 可以在一次 backward pass 中计算所有 gradients。这就是 backpropagation 使用 reverse mode 的原因。

| Mode | Seed | Direction | Best when |
|------|------|-----------|-----------|
| Forward | `dx_i/dx_i = 1` | Input to output | Few inputs, many outputs |
| Reverse | `dy/dy = 1` | Output to input | Many inputs, few outputs (neural nets) |

### Dual Numbers for Forward Mode / 用 dual numbers 实现 forward mode

Forward mode 可以用 dual numbers 优雅实现。一个 dual number 形如 `a + b*epsilon`，其中 `epsilon^2 = 0`。

```
Dual number: (value, derivative)

(2, 1) means: value is 2, derivative w.r.t. x is 1

Arithmetic rules:
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```

把输入变量的 derivative 设为 1，derivative 就会自动穿过每个操作传播。

### Building an Autograd Engine / 构建一个 Autograd Engine

一个 autograd engine 需要三件事：

1. **Value wrapping。** 把每个数字包进对象，存储它的值和 gradient。
2. **Graph recording。** 每个操作记录它的输入和 local gradient function。
3. **Backward pass。** 对图做 topological sort，然后反向遍历，在每个节点应用 chain rule。

这正是 PyTorch 的 `autograd` 在做的事情。`torch.Tensor` class 包装值，在 `requires_grad=True` 时记录操作，并在调用 `.backward()` 时计算 gradients。

### How PyTorch Autograd Works Under the Hood / PyTorch Autograd 底层如何工作

当你写 PyTorch 代码时：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```

PyTorch 在内部会：

1. 为 `x` 创建一个 `requires_grad=True` 的 `Tensor` 节点
2. 每个操作（`**`、`*`、`+`）都会创建新节点，并记录 backward function
3. `y.backward()` 触发穿过记录图的 reverse-mode autodiff
4. 每个节点的 `grad_fn` 计算 local gradients，并把它们传给父节点
5. Gradients 通过加法累积到 `.grad` 属性中，而不是覆盖

这张图是动态的（define-by-run）。每次 forward pass 都会构建一张新图。这就是为什么 PyTorch 支持在模型内部写 control flow（if/else、loops）。

```figure
chain-rule
```

## Build It / 动手构建

### Step 1: The Value class / 第 1 步：Value class

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

每个 `Value` 都会存储数值数据、gradient（初始为零）、backward function，以及生成它的 child nodes 指针。

### Step 2: Arithmetic operations with gradient tracking / 第 2 步：带 gradient tracking 的算术运算

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
```

每个操作都会创建一个 closure，知道如何计算 local gradients，并乘以上游 gradient（`out.grad`）。这里的 `+=` 处理的是同一个 value 被多个操作使用的情况。

### Step 3: The backward pass / 第 3 步：Backward pass

```python
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

Topological sort 确保每个节点的 gradient 已经完全计算好，再传播到它的 children。种子 gradient 是 1.0（dy/dy = 1）。

### Step 4: More operations for a complete engine / 第 4 步：补齐完整 engine 需要的更多操作

基础 Value class 只处理加法、乘法和 relu。真实 autograd engine 需要更多操作。下面这些操作足以构建神经网络：

```python
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
```

**每个操作为什么重要：**

| Operation | Backward rule | Used in |
|-----------|--------------|---------|
| `__sub__` | Reuses add + neg | Loss computation (pred - target) |
| `__pow__` | n * x^(n-1) | Polynomial activations, MSE (error^2) |
| `__truediv__` | Reuses mul + pow(-1) | Normalization, learning rate scaling |
| `exp` | exp(x) * upstream | Softmax, log-likelihood |
| `log` | (1/x) * upstream | Cross-entropy loss, log probabilities |
| `tanh` | (1 - tanh^2) * upstream | Classic activation function |

聪明之处在于：`__sub__` 和 `__truediv__` 是基于已有操作定义的。因为 chain rule 会穿过底层 add/mul/pow 操作组合起来，它们会自动得到正确 gradients。

### Step 5: Mini MLP from scratch / 第 5 步：从零构建 Mini MLP

有了完整 Value class，你就可以构建神经网络。不用 PyTorch，不用 NumPy，只用 Values 和 chain rule。

```python
import random

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

一个 `Neuron` 计算 `tanh(w1*x1 + w2*x2 + ... + b)`。一个 `Layer` 是一组 neurons。一个 `MLP` 会堆叠多个 layers。每个 weight 都是 `Value`，所以调用 `loss.backward()` 会把 gradients 传播到每个参数。

**在 XOR 上训练：**

```python
random.seed(42)
model = MLP([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR pattern (using -1/1 for tanh)

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss = {loss.data:.4f}")

print("\nPredictions after training:")
for x, y in zip(xs, ys):
    print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")
```

这就是 micrograd。纯 Python 写出的完整神经网络训练循环，带 automatic differentiation。每个商业深度学习框架都在大规模上做同一件事。

### Step 6: Gradient checking / 第 6 步：Gradient checking

你怎么知道自己的 autodiff 是正确的？把它和 numerical derivatives 对比。这就是 gradient checking。

```python
def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff
```

在一个复杂表达式上测试它：

```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"Autodiff:  {ad:.8f}")
print(f"Numerical: {num:.8f}")
print(f"Difference: {diff:.2e}")
# Difference should be < 1e-5
```

实现新操作时，gradient checking 是必需的。如果你的 backward pass 有 bug，numerical check 会抓到它。每个严肃的 deep learning implementation 都会在开发期间运行 gradient checks。

**什么时候使用 gradient checking：**

| Situation | Do gradient check? |
|-----------|-------------------|
| 给 autograd 添加新操作 | 是，始终要做 |
| 调试不收敛的 training loop | 是，先检查 gradients |
| 生产训练 | 不做，太慢（每个参数需要 2 次 forward passes） |
| Autograd code 的 unit tests | 是，把它自动化 |

### Step 7: Verify against manual calculation / 第 7 步：对照手算验证

```python
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)    # b = 7.0
y = b.relu()          # y = 7.0

y.backward()

print(f"y = {y.data}")          # 7.0
print(f"dy/dx1 = {x1.grad}")   # 3.0 (= x2)
print(f"dy/dx2 = {x2.grad}")   # 2.0 (= x1)
```

手算检查：`y = relu(x1*x2 + 1)`。因为 `x1*x2 + 1 = 7 > 0`，relu 是 identity。
`dy/dx1 = x2 = 3`。`dy/dx2 = x1 = 2`。这个 engine 的结果一致。

## Use It / 应用它

### Verify against PyTorch / 对照 PyTorch 验证

```python
import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0
```

Gradients 相同。你的 engine 计算结果与 PyTorch 相同，因为数学相同：通过 chain rule 做 reverse-mode autodiff。

### A more complex expression / 一个更复杂的表达式

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = (a * b + c).relu()  # relu(2*(-3) + 10) = relu(4) = 4

f.backward()
print(f"df/da = {a.grad}")  # -3.0 (= b)
print(f"df/db = {b.grad}")  #  2.0 (= a)
print(f"df/dc = {c.grad}")  #  1.0
```

## Ship It / 交付它

本课产出：
- `outputs/skill-autodiff.md`：一个用于构建和调试 autograd systems 的 skill
- `code/autodiff.py`：一个可以继续扩展的最小 autograd engine

这里构建的 Value class 是 Phase 3 神经网络训练循环的基础。

## Exercises / 练习

1. 给 Value class 添加 `__pow__`，让它可以计算 `x ** n`。验证 `x=2` 时 `d/dx(x^3)` 等于 `12.0`。

2. 添加 `tanh` activation function。验证 `tanh'(0) = 1`，并且 `tanh'(2) = 0.0707`（近似）。

3. 为单个 neuron 构建 computation graph：`y = relu(w1*x1 + w2*x2 + b)`。计算全部五个 gradients，并与 PyTorch 对照验证。

4. 使用 dual numbers 实现 forward-mode autodiff。创建一个 `Dual` class，并验证它给出的 derivatives 与你的 reverse-mode engine 一致。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Chain rule | “把导数乘起来” | 复合函数的导数等于每个函数在正确位置的 local derivative 之积 |
| Computational graph | “网络图” | 一个 directed acyclic graph，其中节点是操作，边承载数值（forward）或 gradients（backward） |
| Forward mode | “把 derivatives 往前推” | 从输入到输出传播 derivatives 的 autodiff。每个输入变量需要一次 pass。 |
| Reverse mode | “Backpropagation” | 从输出到输入传播 gradients 的 autodiff。每个输出变量需要一次 pass。 |
| Autograd | “自动 gradients” | 一个记录 value 上的操作、构建图，并通过 chain rule 计算精确 gradients 的系统 |
| Dual numbers | “值加导数” | 形如 a + b*epsilon（epsilon^2 = 0）的数，可以在算术运算中携带 derivative 信息 |
| Topological sort | “依赖顺序” | 对图节点排序，让每个节点都在它的依赖之后出现。正确传播 gradients 需要它。 |
| Gradient accumulation | “相加，不是替换” | 当一个 value 流入多个操作时，它的 gradient 是所有 incoming gradient contributions 的总和 |
| Dynamic graph | “Define by run” | 每次 forward pass 都重建的 computation graph，允许在模型中使用 Python control flow（PyTorch 风格） |
| Gradient checking | “Numerical verification” | 把 autodiff gradients 与 numerical finite-difference gradients 对比来验证正确性。调试时必不可少。 |
| MLP | “Multi-layer perceptron” | 一个有一层或多层 hidden layers 的 neural network。每个 neuron 计算加权和加 bias，再应用 activation function。 |
| Neuron | “Weighted sum + activation” | 基本单元：output = activation(w1*x1 + w2*x2 + ... + b)。Weights 和 bias 是 learnable parameters。 |

## Further Reading / 延伸阅读

- [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) -- 神经网络中 chain rule 的可视化解释
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) -- 真实系统如何工作
- [Baydin et al., Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) -- 综合参考
