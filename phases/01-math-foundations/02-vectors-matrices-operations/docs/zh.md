# Vectors, Matrices & Operations / 向量、矩阵与运算

> 每个神经网络，本质上都是矩阵乘法，只是外面多了几层步骤。

**类型：** 构建
**语言：** Python, Julia
**前置要求：** Phase 1, Lesson 01 (Linear Algebra Intuition)
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 构建一个 Matrix class，支持逐元素运算、矩阵乘法、转置、行列式和逆矩阵
- 区分逐元素乘法和矩阵乘法，并解释它们各自在什么场景下适用
- 只使用从零实现的 Matrix class，实现一个 dense neural network layer（`relu(W @ x + b)`）
- 解释 broadcasting rules，以及神经网络框架中 bias addition 是如何工作的

## The Problem / 问题

你想构建一个神经网络。读代码时看到这一行：

```
output = activation(weights @ input + bias)
```

这里的 `@` 是矩阵乘法。`weights` 是矩阵。`input` 是向量。如果你不知道这些运算在做什么，这一行就是魔法；如果你懂，它就是一个层的整个 forward pass，只用了三步。

模型处理的每张图像都是像素值矩阵。每个 word embedding 都是向量。每个神经网络的每一层都是矩阵变换。你需要像理解变量一样熟悉矩阵运算，否则无法真正构建 AI 系统。

本课会从零建立这种熟练度。

## The Concept / 概念

### Vectors: ordered lists of numbers / 向量：有序数字列表

向量是一串有方向和长度的数字。在 AI 中，向量表示数据点、特征或参数。

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

二维向量 `[3, 4]` 指向平面上的坐标 (3, 4)。它的长度是 5，也就是 3-4-5 三角形。

### Matrices: grids of numbers / 矩阵：数字网格

矩阵是二维网格，有行和列。一个 m x n 矩阵有 m 行、n 列。

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

在神经网络中，权重矩阵会把输入向量变换成输出向量。一个有 784 个输入和 128 个输出的层，会使用一个 128x784 的权重矩阵。

### Why shapes matter / 为什么 shape 很重要

矩阵乘法有一条严格规则：`(m x n) @ (n x p) = (m x p)`。内部维度必须匹配。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

如果你在 PyTorch 里遇到 shape mismatch error，原因通常就在这里。

### The operations map / 运算地图

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | 逐元素组合 | 给输出加 bias |
| Scalar multiply | 缩放每个元素 | Learning rate * gradients |
| Matrix multiply | 变换向量 | Layer forward pass |
| Transpose | 交换行和列 | Backpropagation |
| Determinant | 单个数值摘要 | 检查可逆性 |
| Inverse | 撤销一个变换 | 求解线性方程组 |
| Identity | 什么都不做的矩阵 | 初始化、residual connections |

### Element-wise vs matrix multiplication / 逐元素乘法与矩阵乘法

初学者最常被这个区别绊倒。

逐元素乘法：对应位置相乘。两个矩阵必须形状相同。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：第一矩阵的行与第二矩阵的列做点积。内部维度必须匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

这是不同的运算，得到不同结果，也遵守不同规则。

### Broadcasting / 广播

当你把 bias vector 加到输出矩阵上时，二者 shape 并不一致。Broadcasting 会把较小的数组“拉伸”到合适形状。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

现代框架都会自动做这件事。理解 broadcasting，能避免你在 shape 看起来不匹配但代码却能运行时感到困惑。

```figure
vector-projection
```

## Build It / 动手构建

### Step 1: Vector class / 第 1 步：Vector class

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Step 2: Matrix class with core operations / 第 2 步：带核心运算的 Matrix class

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Step 3: See it work / 第 3 步：运行看看

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Step 4: Connect to neural networks / 第 4 步：连接到神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是一个 dense layer：`output = relu(W @ x + b)`。每个神经网络里的每个 dense layer 都在做这件事。

## Use It / 应用它

NumPy 能用更少的代码完成上面所有事情，而且速度快几个数量级。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` operator 会调用 `__matmul__`。NumPy 用 C 和 Fortran 写的优化 BLAS routines 来实现它。数学相同，速度快 100 倍。

NumPy 中的 broadcasting：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 会自动把 1D bias 广播到两行。这就是每个神经网络框架中 bias addition 的工作方式。

## Ship It / 交付它

本课产出一个 prompt，用几何直觉讲解矩阵运算。见 `outputs/prompt-matrix-operations.md`。

这里构建的 Matrix class 是我们在 Phase 3, Lesson 10 中构建 mini neural network framework 的基础。

## Exercises / 练习

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()`，确认得到 identity matrix。用三个不同的 2x2 矩阵试一试。determinant 为零时会发生什么？

2. **实现 3x3 逆矩阵。** 扩展 Matrix class，用 adjugate method 计算 3x3 矩阵的逆。用 NumPy 的 `np.linalg.inv` 对照测试。

3. **构建两层网络。** 只使用你的 Matrix class，不用 NumPy，创建一个两层神经网络：input (3) -> hidden (4) -> output (2)。初始化随机权重，运行一次 forward pass，并验证所有 shape 正确。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Vector | “一个箭头” | 有序数字列表。在 AI 中：高维空间里的一个点。 |
| Matrix | “一张数字表” | 一个线性变换。它把向量从一个空间映射到另一个空间。 |
| Matrix multiply | “把数字乘起来而已” | 第一个矩阵的每一行与第二个矩阵的每一列做点积。顺序很重要。 |
| Transpose | “翻过来” | 交换行和列。把 m x n 矩阵变成 n x m。对 backpropagation 很关键。 |
| Determinant | “矩阵算出来的某个数” | 衡量矩阵会把面积（2D）或体积（3D）缩放多少。为零表示这个变换压扁了一个维度。 |
| Inverse | “撤销矩阵” | 反转原变换的矩阵。只有 determinant 不为零时才存在。 |
| Identity matrix | “很无聊的矩阵” | 矩阵里的乘以 1。用于 residual connections（ResNets）。 |
| Broadcasting | “神奇地修 shape” | 通过沿缺失维度重复，把较小数组扩展到匹配较大数组。 |
| Element-wise | “普通乘法” | 对应位置相乘。两个数组必须形状相同，或可以 broadcast。 |

## Further Reading / 延伸阅读

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 这里覆盖的每种运算都有直观可视化
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的精确规则
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 面向 ML 的简明线性代数参考
