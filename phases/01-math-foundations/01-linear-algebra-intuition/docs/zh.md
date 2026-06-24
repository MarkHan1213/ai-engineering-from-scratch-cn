# Linear Algebra Intuition / 线性代数直觉

> 每个 AI 模型，本质上都是矩阵运算，只是披上了更复杂的外衣。

**类型：** 学习
**语言：** Python, Julia
**前置要求：** Phase 0
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 从零用 Python 实现向量和矩阵运算，包括加法、点积和矩阵乘法
- 从几何角度解释点积、投影和 Gram-Schmidt 过程到底在做什么
- 用行约简判断一组向量的线性无关性、秩和基
- 把线性代数概念连接到 AI 应用：embedding、注意力分数和 LoRA

## The Problem / 问题

打开任意一篇 ML 论文。第一页里通常就会出现向量、矩阵、点积和变换。如果没有线性代数直觉，这些只是符号；有了直觉，你就能看见神经网络真正做的事：在空间中移动点。

你不需要先成为数学家。你需要看懂这些运算的几何意义，然后亲手把它们写成代码。

## The Concept / 概念

### Vectors Are Points (and Directions) / 向量是点，也是方向

向量就是一串数字。但这些数字不是随便摆在一起的，它们是在空间里的坐标。

**2D vector [3, 2] / 二维向量 [3, 2]：**

| x | y | Point |
|---|---|-------|
| 3 | 2 | 这个向量从原点 (0,0) 指向平面上的 (3, 2) |

这个向量的长度是 sqrt(3^2 + 2^2) = sqrt(13)，方向朝右上方。

在 AI 里，向量几乎表示一切：
- 一个词 -> 768 个数字组成的向量，也就是它在 embedding 空间中的“含义”
- 一张图像 -> 数百万个像素值组成的向量
- 一个用户 -> 一组偏好向量

### Matrices Are Transformations / 矩阵是变换

矩阵会把一个向量变成另一个向量。它可以旋转、缩放、拉伸，也可以做投影。

```mermaid
graph LR
    subgraph Before
        A["Point A"]
        B["Point B"]
    end
    subgraph Matrix["Matrix Multiplication"]
        M["M (transformation)"]
    end
    subgraph After
        A2["Point A'"]
        B2["Point B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

在 AI 中，矩阵就是模型本身：
- 神经网络权重 -> 把输入转换成输出的矩阵
- 注意力分数 -> 决定该关注什么的矩阵
- Embedding -> 把词映射为向量的矩阵

### The Dot Product Measures Similarity / 点积衡量相似度

两个向量的点积告诉你它们有多相似。

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

Same direction:      a · b > 0  (similar)
Perpendicular:       a · b = 0  (unrelated)
Opposite direction:  a · b < 0  (dissimilar)
```

搜索引擎、推荐系统和 RAG 的核心就是这个思路：找到点积很高的向量。

### Linear Independence / 线性无关

如果一组向量里没有任何一个向量能由其他向量组合出来，那么这组向量就是线性无关的。如果 v1、v2、v3 线性无关，它们张成一个 3D 空间；如果其中一个可以由另外几个组合出来，它们最多只张成一个平面。

这对 AI 很重要：你的特征矩阵最好有线性无关的列。如果两个特征完全相关，也就是线性相关，模型就无法区分它们各自的作用。这会在回归中造成多重共线性：权重矩阵变得不稳定，输入发生很小变化，输出却可能剧烈摆动。

**具体例子：**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

v1 和 v2 线性无关，因为它们都不是对方的标量倍数，也不能由对方组合出来。但 v3 = 2*v1 + v2，所以 {v1, v2, v3} 是线性相关的。这三个向量都位于 xy 平面。不管怎么组合它们，你都到不了 [0, 0, 1]。你有三个向量，却只有两个自由维度。

放到数据集中：如果 feature_3 = 2*feature_1 + feature_2，加入 feature_3 不会给模型带来任何新信息。更糟的是，它会让正规方程变成奇异的，也就是权重没有唯一解。

### Basis and Rank / 基与秩

基是一组最小的线性无关向量，它们能张成整个空间。基向量的数量就是空间维度。

3D 空间的标准基是 {[1,0,0], [0,1,0], [0,0,1]}。但 3D 中任意三个线性无关向量都可以构成合法的基。选择基，本质上就是选择坐标系。

矩阵的秩 = 线性无关列的数量 = 线性无关行的数量。如果 rank < min(rows, cols)，这个矩阵就是秩亏的。这意味着：
- 方程组有无穷多解，或者没有解
- 变换过程中信息丢失了
- 矩阵不可逆

| Situation | Rank | What it means for ML |
|-----------|------|---------------------|
| 满秩（rank = min(m, n)） | 最大可能值 | 存在唯一的最小二乘解。模型条件良好。 |
| 秩亏（rank < min(m, n)） | 低于最大值 | 特征存在冗余。有无穷多组权重解。需要正则化。 |
| 秩 1 | 1 | 每一列都是同一个向量的缩放版。所有数据都落在一条线上。 |
| 近似秩亏（奇异值很小） | 数值上偏低 | 矩阵病态。极小的输入噪声会导致很大的输出变化。使用 SVD 截断或 ridge regression。 |

### Projection / 投影

把向量 **a** 投影到向量 **b** 上，会得到 **a** 沿 **b** 方向的分量：

```
proj_b(a) = (a dot b / b dot b) * b
```

残差 (a - proj_b(a)) 与 b 垂直。这种正交分解是最小二乘拟合的基础。

投影在 ML 中无处不在：
- 线性回归最小化观测值到列空间的距离，它的解本身就是一个投影
- PCA 把数据投影到最大方差方向上
- Transformer 中的注意力会计算 query 到 key 的投影

```mermaid
graph LR
    subgraph Projection["Projection of a onto b"]
        direction TB
        O["Origin"] --> |"b (direction)"| B["b"]
        O --> |"a (original)"| A["a"]
        O --> |"proj_b(a)"| P["projection"]
        A -.-> |"residual (perpendicular)"| P
    end
```

**例子：** a = [3, 4], b = [1, 0]

proj_b(a) = (3*1 + 4*0) / (1*1 + 0*0) * [1, 0] = 3 * [1, 0] = [3, 0]

这个投影丢掉了 y 分量。这就是最简单形式的降维：扔掉你不关心的方向。

### Gram-Schmidt Process / Gram-Schmidt 过程

Gram-Schmidt 会把任意一组线性无关向量转换成一组标准正交基。标准正交的意思是：每个向量长度为 1，并且任意两个向量互相垂直。

算法如下：
1. 取第一个向量，并归一化
2. 取第二个向量，减去它在第一个向量上的投影，再归一化
3. 取第三个向量，减去它在所有前面向量上的投影，再归一化
4. 对剩余向量重复这个过程

```
Input:  v1, v2, v3, ... (linearly independent)

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

Output: u1, u2, u3, ... (orthonormal basis)
```

这就是 QR 分解内部的工作方式。Q 是标准正交基，R 捕获投影系数。QR 分解会用于：
- 求解线性方程组，比 Gaussian elimination 更稳定
- 计算特征值，也就是 QR algorithm
- 最小二乘回归，这是标准数值方法

```figure
eigen-directions
```

## Build It / 动手构建

### Step 1: Vectors from scratch (Python) / 第 1 步：从零实现向量（Python）

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### Step 2: Matrices from scratch (Python) / 第 2 步：从零实现矩阵（Python）

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### Step 3: Why this matters for AI / 第 3 步：为什么这对 AI 重要

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")
```

### Step 4: Julia version / 第 4 步：Julia 版本

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia supports unicode operators
println("|a| = ", √(a ⋅ a))
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# Matrix-vector multiplication
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("This is a neural network layer.")
```

### Step 5: Linear independence and projection from scratch (Python) / 第 5 步：从零实现线性无关和投影（Python）

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## Use It / 应用它

现在用 NumPy 做同样的事，这也是你在实际项目中更常使用的写法：

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### Rank, Projection, and QR with NumPy / 用 NumPy 计算秩、投影和 QR

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch -- Tensors Are Vectors with Autodiff / PyTorch：Tensor 是带自动微分的向量

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

点积对 x 的梯度就是 y。PyTorch 会自动算出这一点。神经网络里的每个操作都建立在类似的运算之上，包括矩阵乘法、点积和投影；autodiff 会沿着这些操作追踪梯度。

你刚刚从零实现了 NumPy 一行代码完成的事。现在你知道底层发生了什么。

## Ship It / 交付它

本课产出：
- `outputs/prompt-linear-algebra-tutor.md`：一个 prompt，用来让 AI assistant 通过几何直觉讲解线性代数

## Connections / 知识连接

本课中的每个概念都连接到现代 AI 的具体部分：

| Concept | Where it shows up |
|---------|------------------|
| 点积 | Transformer 的注意力分数，RAG 中的 cosine similarity |
| 矩阵乘法 | 每一层神经网络，每一个线性变换 |
| 线性无关 | 特征选择，避免多重共线性 |
| 秩 | 判断方程组是否可解，LoRA（low-rank adaptation） |
| 投影 | 线性回归（投影到列空间）、PCA |
| Gram-Schmidt / QR | 数值求解器、特征值计算 |
| 标准正交基 | 稳定的数值计算、whitening transforms |

LoRA 值得单独说明。它通过把权重更新分解为低秩矩阵来微调大语言模型。与其更新一个 4096x4096 的权重矩阵，也就是 16M 参数，LoRA 只更新两个矩阵：4096x16 和 16x4096，也就是 131K 参数。rank-16 约束意味着 LoRA 假设权重更新位于完整 4096 维空间中的一个 16 维子空间里。这就是线性代数在真实系统中发挥作用。

## Exercises / 练习

1. 实现 `Vector.angle_between(other)`，返回两个向量之间的角度，单位为 degree
2. 创建一个 2D 缩放矩阵，让 x 坐标翻倍、y 坐标变成三倍，然后把它应用到向量 [1, 1]
3. 给定 5 个随机“类词向量”（维度 50），用 cosine similarity 找出最相似的两个
4. 验证 Gram-Schmidt 的输出确实是标准正交的：检查任意一对向量点积为 0，并且每个向量长度为 1
5. 创建一个 rank 为 2 的 3x3 矩阵。用 `rank()` 方法验证它。然后解释这些列向量张成了什么几何对象。
6. 把向量 [1, 2, 3] 投影到 [1, 1, 1] 上。这个结果在几何上代表什么？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Vector | “一个箭头” | 表示 n 维空间中某个点或方向的一串数字 |
| Matrix | “一张数字表” | 把向量从一个空间映射到另一个空间的变换 |
| Dot product | “相乘再求和” | 衡量两个向量对齐程度的量，也是 similarity search 的核心 |
| Embedding | “某种 AI 魔法” | 表示某个对象含义的向量，例如词、图像或用户 |
| Linear independence | “它们不重叠” | 这组向量中没有任何一个能由其他向量组合出来 |
| Rank | “有多少维” | 矩阵中线性无关列（或行）的数量 |
| Projection | “影子” | 一个向量沿另一个向量方向的分量 |
| Basis | “坐标轴” | 一组最小的无关向量，能够张成整个空间 |
| Orthonormal | “互相垂直的单位向量” | 向量彼此垂直，并且每个长度都是 1 |
