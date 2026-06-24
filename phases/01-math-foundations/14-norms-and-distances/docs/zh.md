# Norms and Distances / 范数与距离

> 距离函数定义了什么叫“相似”。选错了，后面的所有东西都会错。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 从零实现 L1、L2、cosine、Mahalanobis、Jaccard 和 edit distance functions
- 为给定 ML task 选择合适的 distance metric，并解释为什么其他选择会失败
- 把 L1 和 L2 norms 连接到 LASSO 与 Ridge regularization，以及它们的几何 constraint regions
- 演示同一个 dataset 在不同 metrics 下如何产生不同 nearest neighbors

## The Problem / 问题

你有两个向量。也许它们是 word embeddings，也许是 user profiles，也许是 pixel arrays。你需要知道：它们有多近？

答案完全取决于你选择哪种 distance function。两个 data points 在一种 metric 下可能是 nearest neighbors，在另一种 metric 下却相距很远。你的 KNN classifier、recommendation engine、vector database、clustering algorithm、loss function，全都依赖这个选择。选错了，模型就会优化错误的东西。

不存在 universally best distance。L2 适合 spatial data。Cosine similarity 主导 NLP。Jaccard 处理 sets。Edit distance 处理 strings。Mahalanobis 会考虑 correlations。Wasserstein 移动 probability mass。每一种距离都编码了一种关于“相似”含义的假设。

本课会从零构建主要 distance functions，说明什么时候该用哪一个，并演示同一份数据在不同 metric 下如何产生完全不同的 nearest neighbors。

## The Concept / 概念

### Norms: measuring vector magnitude / 范数：衡量向量大小

Norm 衡量一个向量的“大小”。任意两个向量之间的 distance function 都可以写成它们差值的 norm：d(a, b) = ||a - b||。所以理解 norms，就是理解 distances。

### L1 Norm (Manhattan distance) / L1 范数（Manhattan distance）

L1 norm 对所有 components 的绝对值求和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它叫 Manhattan distance，因为它衡量的是你在城市网格中只能沿坐标轴走时的距离。不能走对角线。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

什么时候用 L1：
- High-dimensional sparse data，例如 text features、one-hot encodings
- 希望对 outliers 更 robust，一个巨大差异不会主导结果
- Feature selection problems，因为 L1 regularization 会促进 sparsity

与 L1 regularization（Lasso）的连接：在 loss function 中添加 ||w||_1，会惩罚 weight 绝对值之和。这会把小 weights 推到精确零，从而自动 feature selection。L1 penalty 在 weight space 中产生 diamond-shaped constraint regions，菱形的角位于坐标轴上，也就是某些 weights 为零的位置。

与 loss functions 的连接：Mean Absolute Error（MAE）是 predictions 与 targets 之间的平均 L1 distance。它对所有 errors 线性惩罚，因此相比 MSE 对 outliers 更 robust。

### L2 Norm (Euclidean distance) / L2 范数（Euclidean distance）

L2 norm 是直线距离。它是 components 平方和的平方根。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这就是几何课上学过的距离。n 维中的 Pythagoras。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

什么时候用 L2：
- Low-to-medium dimensional continuous data
- Feature scales 彼此可比
- Physical distances，例如 spatial data、sensor readings
- Pixel 级别的 image similarity

与 L2 regularization（Ridge）的连接：在 loss function 中添加 ||w||_2^2，会惩罚大 weights。与 L1 不同，它不会把 weights 推到零，而是把所有 weights 按比例向零收缩。L2 penalty 产生 circular constraint regions，因此没有落在坐标轴上的尖角。Weights 会变小，但很少精确为零。

与 loss functions 的连接：Mean Squared Error（MSE）是 L2 distances squared 的平均。平方会让大 errors 比小 errors 受到更重惩罚。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp Norms: the general family / Lp 范数：通用族

L1 和 L2 是 Lp norm 的特殊情况：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同 p 值会产生不同形状的 “unit balls”，也就是所有距离 origin 为 1 的点集合：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity Norm (Chebyshev distance) / L-infinity 范数（Chebyshev distance）

当 p 趋近 infinity，Lp norm 收敛到最大 absolute component。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离由差异最大的单一维度决定，其他维度被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

什么时候用 L-infinity：
- 任何单一维度中的 worst-case deviation 很重要
- Game boards，例如 chess 中 king 的移动就是 L-infinity：任意方向走一步成本都是 1
- Manufacturing tolerances，每个 dimension 都必须在 spec 内

### Cosine Similarity and Cosine Distance / Cosine similarity 与 cosine distance

Cosine similarity 衡量两个向量之间的角度，忽略它们的 magnitudes。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

它范围从 -1（相反方向）到 +1（相同方向）。垂直向量的 cosine similarity 是 0。

Cosine distance 把它转换成距离：cosine_distance = 1 - cosine_similarity。范围从 0（方向相同）到 2（方向相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么 cosine 主导 NLP 和 embeddings：在文本中，document length 不应该影响 similarity。一篇关于 cats 的文档，即使比另一篇关于 cats 的文档长两倍，也仍然应该“相似”。Cosine similarity 忽略 magnitude（长度），只关心 direction。两个 word distribution 相同但长度不同的 documents，会指向同一方向，并得到 cosine similarity 1.0。

什么时候用 cosine similarity：
- Text similarity，例如 TF-IDF vectors、word embeddings、sentence embeddings
- 任何 magnitude 是噪声、direction 是信号的领域
- Recommendation systems（user preference vectors）
- Embedding search（vector databases 几乎总是用 cosine 或 dot product）

### Dot Product Similarity vs Cosine Similarity / Dot product similarity 与 cosine similarity

两个向量的 dot product 是：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Cosine similarity 是除以两个 magnitudes 后的 dot product。当两个向量已经 unit-normalized（magnitude = 1）时，dot product 和 cosine similarity 完全相同。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

它们的差别：dot product 包含 magnitude 信息。Magnitude 更大的向量会得到更高 dot product score。这在某些 retrieval systems 中很重要，因为你可能希望“热门” items 排得更高。Magnitude 会成为隐式 quality 或 importance signal。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实践中：
- 当你只想要纯 direction similarity 时，使用 cosine similarity
- 当 magnitudes 携带有意义信息时，使用 dot product
- 许多 vector databases（Pinecone、Weaviate、Qdrant）允许你选择
- 如果 embeddings 已经 L2-normalized，这个选择没有区别

### Mahalanobis Distance / Mahalanobis 距离

Euclidean distance 把所有 dimensions 同等对待。但如果 features 相关，或 scale 不同，L2 会给出误导结果。

Mahalanobis distance 会考虑数据的 covariance structure。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的 covariance matrix。

直觉上：Mahalanobis distance 会先对数据 decorrelate 并 normalize，也就是 whitening，然后在转换后的空间里计算 L2 distance。如果 S 是 identity matrix（uncorrelated、unit variance features），Mahalanobis distance 就退化为 Euclidean distance。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

什么时候用 Mahalanobis distance：
- Outlier detection，Mahalanobis distance 离 mean 很远的点就是 outliers
- Features 有不同 scales 和 correlations 时的 classification
- 有足够数据估计可靠 covariance matrix 时
- Manufacturing 中的 quality control，例如 multivariate process monitoring

### Jaccard Similarity (for sets) / Jaccard similarity（用于 sets）

Jaccard similarity 衡量两个 sets 的重叠。

```
J(A, B) = |A intersect B| / |A union B|
```

范围从 0（无重叠）到 1（完全相同）。Jaccard distance = 1 - Jaccard similarity。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

什么时候用 Jaccard：
- 比较 tags、categories 或 features 的 sets
- 基于 word presence 而不是 frequency 的 document similarity
- Near-duplicate detection（MinHash 是 Jaccard 的近似）
- 比较 binary feature vectors（presence/absence data）
- 评估 segmentation models（Intersection over Union = Jaccard）

### Edit Distance (Levenshtein Distance) / 编辑距离（Levenshtein distance）

Edit distance 计算把一个 string 变成另一个 string 所需的最少 single-character operations。操作包括 insert、delete 或 substitute。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

它用 dynamic programming 计算。填一个矩阵，其中 entry (i, j) 是 string A 的前 i 个字符与 string B 的前 j 个字符之间的 edit distance。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

什么时候用 edit distance：
- Spell checking and correction
- DNA sequence alignment（带 weighted operations）
- Fuzzy string matching
- Messy text data 的 deduplication

### KL Divergence (not a distance, but used like one) / KL 散度（不是距离，但常被当距离用）

KL divergence 衡量一个 probability distribution 与另一个有多不同。Lesson 09 已覆盖它，但它也属于这次讨论，因为很多人把它当成“距离”使用，尽管它不是。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL divergence **不对称**。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足 distance metric 的基本要求。它也不满足 triangle inequality。它是 divergence，不是 distance。

Forward KL（D_KL(P || Q)）是 "mean-seeking"：Q 会尝试覆盖 P 的所有 modes。
Reverse KL（D_KL(Q || P)）是 "mode-seeking"：Q 会聚焦 P 的单个 mode。

你会在这些地方看到 KL divergence：
- VAEs（ELBO 中的 KL term 会把 latent distribution 推向 prior）
- Knowledge distillation（student 尝试匹配 teacher distribution）
- RLHF（KL penalty 让 fine-tuned model 靠近 base model）
- Policy gradient methods（约束 policy updates）

### Wasserstein Distance (Earth Mover's Distance) / Wasserstein 距离（Earth Mover's Distance）

Wasserstein distance 衡量把一个 probability distribution 变成另一个所需的最小“工作量”。可以这样想：如果一个分布是一堆土，另一个分布是坑，你需要搬多少土、搬多远？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对 1D distributions，它会简化为 cumulative distribution functions 绝对差的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么 Wasserstein 重要：
- 它是真正的 metric，对称且满足 triangle inequality
- 即使 distributions 不重叠，它也能提供 gradients，而 KL divergence 会变成 infinity
- 这个性质让它成为 Wasserstein GANs（WGANs）的核心，解决了原始 GANs 的 training instability

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

什么时候用 Wasserstein：
- GAN training（WGAN、WGAN-GP）
- 比较可能不重叠的 distributions
- Optimal transport problems
- Image retrieval（比较 color histograms）

### Why Different Tasks Need Different Distances / 为什么不同任务需要不同距离

| Task | Best distance | Why |
|------|--------------|-----|
| Text similarity | Cosine | Magnitude 是噪声，direction 是含义 |
| Image pixel comparison | L2 | Spatial relationships 重要，features scale 可比 |
| Sparse high-dim features | L1 | Robust，不会放大稀有的大差异 |
| Set overlap（tags、categories） | Jaccard | 数据天然是 set-valued，不是 vectorial |
| String matching | Edit distance | Operations 对应人类编辑直觉 |
| Outlier detection | Mahalanobis | 考虑 feature correlations 和 scales |
| Comparing distributions | KL divergence | 衡量使用 Q 而不是 P 丢失的信息 |
| GAN training | Wasserstein | 即使 distributions 不重叠也能提供 gradients |
| Embeddings（vector DB） | Cosine or dot product | Embeddings 被训练为用 direction 编码 meaning |
| Recommendation | Dot product | Magnitude 可以编码 popularity 或 confidence |
| DNA sequences | Weighted edit distance | 不同 nucleotide pair 的 substitution costs 不同 |
| Manufacturing QC | L-infinity | 任意 dimension 的 worst-case deviation 都重要 |

### Connection to Loss Functions / 与 Loss Functions 的连接

Loss functions 是应用在 predictions 和 targets 之间的 distance functions。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### Connection to Regularization / 与 Regularization 的连接

Regularization 会给 loss function 添加 weight 的 norm penalty。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 产生 sparsity，而 L2 不会：想象 2D weight space 中的 constraint region。L1 是菱形，L2 是圆。Loss function 的 contours（ellipses）最可能在菱形的角上接触约束区域，而角位于某个 weight 为零的轴上。它们接触圆时通常在平滑点上，两个 weights 都非零。

### Nearest Neighbor Search / 最近邻搜索

每个 distance function 都隐含了一个 nearest neighbor search problem：给定 query point，在 dataset 中找到最近的点。

对 n 个点、d 维的 dataset，exact nearest neighbor search 每次 query 是 O(n * d)。大数据集上这太慢。

Approximate Nearest Neighbor（ANN）algorithms 用少量 accuracy 损失换取巨大速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（Hierarchical Navigable Small World）是现代 vector databases 中的主流算法。它构建一张多层图，每个节点连接到自己的 approximate nearest neighbors。搜索从顶层开始（稀疏、长跳跃），逐层下降到底层（密集、短跳跃）。

```figure
norm-unit-balls
```

## Build It / 动手构建

### Step 1: All norm and distance functions / 第 1 步：所有 norm 和 distance functions

完整实现见 `code/distances.py`。每个 function 都只用基础 Python math 从零构建。

### Step 2: Same data, different distances, different neighbors / 第 2 步：同一份数据，不同距离，不同邻居

`distances.py` 中的 demo 会创建一个 dataset，选择一个 query point，并展示 nearest neighbor 如何随 distance metric 改变。L1 下“最近”的点，在 L2 或 cosine 下可能不是最近的。

### Step 3: Embedding similarity search / 第 3 步：Embedding similarity search

代码包含一个 mock embedding similarity search，用 cosine similarity 与 L2 distance 找出与 query 最相似的“documents”，展示 rankings 可能不同。

## Use It / 应用它

最常见的实践用途：在 vector database 中寻找相似 items。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后搜索 vector database 时，底层做的就是这件事。Embedding model 把文本映射到 vectors。Vector database 在 query vector 与每个 stored vector 之间计算 cosine similarity（或 dot product），并用 ANN algorithms 避免逐个检查所有向量。

## Ship It / 交付它

本课交付一个距离选择框架：根据数据形态、任务目标和下游算法，判断该用 L1、L2、cosine、Mahalanobis、Jaccard、edit distance、KL divergence 还是 Wasserstein。

## Exercises / 练习

1. 计算 (1, 2, 3) 和 (4, 0, 6) 之间的 L1、L2 和 L-infinity distances。验证任意一对点都满足 L-inf <= L2 <= L1。证明为什么这个顺序总是成立。

2. 创建两个 vectors，使它们的 cosine similarity 很高（> 0.9），但 L2 distance 很大（> 10）。从几何上解释发生了什么。再创建两个 vectors，使 cosine similarity 很低（< 0.3），但 L2 distance 很小（< 0.5）。

3. 实现一个函数，输入 dataset 和 query point，分别返回 L1、L2、cosine 和 Mahalanobis distance 下的 nearest neighbor。找到一个 dataset，让四种 metric 对哪个点最近产生分歧。

4. 用 CDF method 手算 [0.5, 0.5, 0, 0] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein distance。然后计算 [0.25, 0.25, 0.25, 0.25] 和 [0, 0, 0.5, 0.5] 之间的距离。哪个更大？为什么？

5. 实现 MinHash 来近似 Jaccard similarity。生成 100 个 random sets，计算所有 pairs 的 exact Jaccard，并用 50、100、200 个 hash functions 的 MinHash approximation 对比。绘制 approximation error。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Norm | “向量大小” | 一个把 vector 映射到非负标量的 function，满足 triangle inequality、absolute homogeneity，且只有 zero vector 的 norm 为零 |
| L1 norm | “Manhattan distance” | Components 绝对值之和。在 optimization 中产生 sparsity。对 outliers robust |
| L2 norm | “Euclidean distance” | Components 平方和的平方根。Euclidean space 中的直线距离 |
| Lp norm | “Generalized norm” | Components 绝对值 p 次方之和的 p 次根。L1 和 L2 是特殊情况 |
| L-infinity norm | “Max norm” 或 “Chebyshev distance” | 最大 absolute component value。Lp 在 p 趋近 infinity 时的极限 |
| Cosine similarity | “向量之间的角度” | 由两个 magnitudes 归一化后的 dot product。范围 -1 到 +1。忽略 vector length |
| Cosine distance | “1 minus cosine similarity” | 把 cosine similarity 转换为 distance。范围 0 到 2 |
| Dot product | “Unnormalized cosine” | Component-wise products 之和。等于 cosine similarity 乘以两个 magnitudes |
| Mahalanobis distance | “考虑 correlation 的距离” | 在用 data covariance matrix whitening（decorrelated and normalized）后的空间中计算 L2 distance |
| Jaccard similarity | “Set overlap” | Intersection size 除以 union size。用于 sets，不是 vectors |
| Edit distance | “Levenshtein distance” | 把一个 string 转换成另一个所需的最少 insertions、deletions 和 substitutions |
| KL divergence | “分布之间的距离” | 不是真正 distance（不对称）。衡量使用 Q 编码 P 时额外浪费的 bits |
| Wasserstein distance | “Earth mover's distance” | 把 mass 从一个 distribution 运到另一个的最小工作量。是真正 metric |
| Approximate nearest neighbor | “ANN search” | 用 HNSW、LSH、IVF 等算法，以远快于 exact search 的速度寻找近似最近点 |
| HNSW | “Vector DB algorithm” | Hierarchical Navigable Small World graph。用于快速 approximate nearest neighbor search 的多层图 |
| L1 regularization | “Lasso” | 把 weights 的 L1 norm 加到 loss 中。驱动 weights 变成零（sparsity） |
| L2 regularization | “Ridge” 或 “weight decay” | 把 weights 的 squared L2 norm 加到 loss 中。让 weights 向零收缩，但不产生 sparsity |
| Elastic Net | “L1 + L2” | 结合 L1 和 L2 regularization。比单独使用任一方法更适合处理 correlated feature groups |

## Further Reading / 延伸阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 的 billion-scale ANN search library
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 把 Earth Mover's distance 引入 GANs 的论文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 基础 ANN algorithm
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，让 cosine similarity 成为 embeddings 默认选择
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中 distance metrics 和 neighbor algorithms 的实践指南
