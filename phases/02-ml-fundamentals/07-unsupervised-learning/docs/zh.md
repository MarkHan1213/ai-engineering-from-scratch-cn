# Unsupervised Learning / 无监督学习

> 没有标签，没有老师。算法自己发现结构。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 1 (Norms & Distances, Probability & Distributions), Phase 2 Lessons 1-6
**Time / 时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 从零实现 K-Means、DBSCAN 和 Gaussian Mixture Models，并比较它们的 clustering 行为
- 使用 silhouette score 和 elbow method 评估 cluster quality，并选择合适的 K
- 解释 DBSCAN 什么时候优于 K-Means，并识别哪些算法能处理非球形 clusters 和 outliers
- 构建 anomaly detection pipeline，用 clustering methods 标记偏离正常模式的点

## The Problem / 问题

到目前为止，每节 ML 课都假设你有 labeled data：“这是输入，这是正确输出。”但真实世界中，标签很贵。医院有数百万条病历，却没人手动给每条记录标 disease category。电商网站有数百万用户会话，却没人手工标注 customer segments。安全团队有网络日志，但没人把每个 anomaly 都标出来。

Unsupervised learning 在没有明确告诉它要找什么的情况下发现模式。它把相似的数据点分组，发现隐藏结构，并浮现异常。如果 supervised learning 像是拿着答案册读课本，那么 unsupervised learning 就是在原始数据里盯到模式自己显现。

代价是：没有 labels，你无法直接衡量“对”或“错”。你需要不同工具来判断算法发现的结构是否有意义。

## The Concept / 概念

### Clustering: Grouping Similar Things Together / Clustering：把相似事物分到一起

Clustering 把每个数据点分配到一个 group（cluster），让同一 group 内的点比不同 group 的点更相似。核心问题永远是：“similar” 到底是什么意思？

```mermaid
flowchart LR
    A[Raw Data] --> B{Choose Method}
    B --> C[K-Means]
    B --> D[DBSCAN]
    B --> E[Hierarchical]
    B --> F[GMM]
    C --> G[Flat, spherical clusters]
    D --> H[Arbitrary shapes, noise detection]
    E --> I[Tree of nested clusters]
    F --> J[Soft assignments, elliptical clusters]
```

### K-Means: The Workhorse / K-Means：最常用的基线方法

K-Means 把数据划分成刚好 K 个 clusters。每个 cluster 有一个 centroid（质心），每个点都属于最近的 centroid。

Lloyd's algorithm：

1. 随机选择 K 个点作为初始 centroids
2. 把每个数据点分配给最近的 centroid
3. 把每个 centroid 重新计算为其分配点的均值
4. 重复步骤 2-3，直到 assignments 不再变化

Objective function（inertia）衡量每个点到其所属 centroid 的总平方距离。K-Means 会最小化它，但只能找到 local minimum。不同初始化可能得到不同结果。

### Choosing K / 选择 K

两种标准方法：

**Elbow method / 肘部法：** 对 K = 1, 2, 3, ..., n 运行 K-Means。绘制 inertia vs K。寻找“肘部”：继续增加 clusters 不再显著降低 inertia 的位置。

**Silhouette score / 轮廓系数：** 对每个点，衡量它与自己 cluster 的相似度（a）和与最近其他 cluster 的相似度（b）。Silhouette coefficient 为 (b - a) / max(a, b)，范围从 -1（分错 cluster）到 +1（cluster 很好）。对所有点取平均就是全局分数。

### DBSCAN: Density-Based Clustering / DBSCAN：基于密度的聚类

K-Means 假设 clusters 是球形的，并要求你提前选择 K。DBSCAN 两个假设都不需要。它把 clusters 视为由稠密区域组成、由稀疏区域隔开的结构。

两个参数：
- **eps**：neighborhood 的半径
- **min_samples**：形成稠密区域所需的最少点数

三类点：
- **Core point**：eps 距离内至少有 min_samples 个点
- **Border point**：位于某个 core point 的 eps 距离内，但自身不是 core point
- **Noise point**：既不是 core，也不是 border。这些是 outliers。

DBSCAN 会把 eps 距离内互相连通的 core points 归入同一 cluster。Border points 加入附近 core point 的 cluster。Noise points 不属于任何 cluster。

优点：能发现任意形状 clusters，自动确定 clusters 数量，识别 outliers。缺点：面对不同密度的 clusters 时表现困难。

### Hierarchical Clustering / 层次聚类

构建一棵嵌套 clusters 的树（dendrogram）。

Agglomerative（自底向上）：
1. 每个点一开始都是自己的 cluster
2. 合并最近的两个 clusters
3. 重复直到只剩一个 cluster
4. 在期望层级切断 dendrogram，得到 K 个 clusters

Clusters 之间的“距离”可以这样衡量：
- **Single linkage**：两个 clusters 中任意两点的最小距离
- **Complete linkage**：任意两点的最大距离
- **Average linkage**：所有点对距离的平均值
- **Ward's method**：使 cluster 内总 variance 增加最小的合并方式

### Gaussian Mixture Models (GMM) / 高斯混合模型

K-Means 给出 hard assignments：每个点只属于一个 cluster。GMM 给出 soft assignments：每个点都有属于每个 cluster 的概率。

GMM 假设数据由 K 个 Gaussian distributions 混合生成，每个 Gaussian 都有自己的 mean 和 covariance。Expectation-Maximization (EM) algorithm 在两步之间交替：

- **E-step**：计算每个点属于每个 Gaussian 的概率
- **M-step**：更新每个 Gaussian 的 mean、covariance 和 mixing weight，使数据 likelihood 最大化

GMM 能建模 elliptical clusters（不只是 K-Means 那样的球形），并天然处理 overlapping clusters。

### When to Use Which / 什么时候用哪个

| Method / 方法 | Best for / 适合 | Avoid when / 避免场景 |
|--------|----------|------------|
| K-Means | 大数据集、球形 clusters、已知 K | 不规则形状、有 outliers |
| DBSCAN | 未知 K、任意形状、outlier detection | 密度变化大、非常高维 |
| Hierarchical | 小数据集、需要 dendrogram、未知 K | 大数据集（O(n^2) memory） |
| GMM | Overlapping clusters、需要 soft assignments | 超大数据集、维度太多 |

### Anomaly Detection with Clustering / 用聚类做异常检测

Clustering 很自然地支持 anomaly detection：
- **K-Means**：离任何 centroid 都远的点是 anomalies
- **DBSCAN**：noise points 天然就是 anomalies
- **GMM**：在所有 Gaussians 下概率都低的点是 anomalies

```figure
kmeans-step
```

## Build It / 动手构建

### Step 1: K-Means from scratch / 第 1 步：从零实现 K-Means

```python
import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n_features = len(data[0])

    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    return assignments, centroids
```

### Step 2: Elbow method and silhouette score / 第 2 步：Elbow method 与 silhouette score

```python
def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("Elbow method:")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: inertia={inertia:.2f}")

    print("\nSilhouette scores:")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: silhouette={score:.4f}")

    return inertias
```

### Step 3: DBSCAN from scratch / 第 3 步：从零实现 DBSCAN

```python
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels
```

### Step 4: Gaussian Mixture Model (EM algorithm) / 第 4 步：Gaussian Mixture Model（EM algorithm）

```python
def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM converged at iteration {iteration + 1}")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities
```

### Step 5: Generate test data and run everything / 第 5 步：生成测试数据并运行全部方法

```python
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== K-Means on 3 blobs ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  Centroids: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  Silhouette score: {sil:.4f}")

    print("\n=== Elbow Method ===")
    find_best_k(data, max_k=6)

    print("\n=== DBSCAN on 3 blobs ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    print("\n=== GMM on 3 blobs ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  Means: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  Weights: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  Silhouette score: {gmm_sil:.4f}")

    print("\n=== DBSCAN on moons (non-spherical clusters) ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  Found {n_moon_clusters} clusters, {n_moon_noise} noise points")

    print("\n=== K-Means on moons (will fail to separate) ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  Silhouette score: {moon_sil:.4f}")
    print("  K-Means splits moons poorly because they are not spherical")

    print("\n=== Anomaly detection with DBSCAN ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  Detected {len(anomalies)} anomalies")
    for a in anomalies[-3:]:
        print(f"    Point {[round(v, 2) for v in a]}")
```

## Use It / 应用它

用 scikit-learn，这些算法都是一行：

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)
```

From-scratch 版本展示了这些库内部到底在计算什么。K-Means 在 assignment 和 recomputing 之间迭代。DBSCAN 从 dense seeds 扩展 clusters。GMM 在 expectation 和 maximization 之间交替。库版本加入了数值稳定性、更聪明的初始化（K-Means++）和 GPU 加速，但核心逻辑相同。

## Ship It / 交付它

本课会产出从零实现的 K-Means、DBSCAN 和 GMM。Clustering 代码可以作为更高级 unsupervised methods 的基础。

## Exercises / 练习

1. 实现 K-Means++ initialization：不是随机选择 centroids，而是先随机选第一个，然后后续每个 centroid 按其到最近已有 centroid 的 squared distance 成比例采样。与随机初始化比较收敛速度。
2. 给代码添加 hierarchical agglomerative clustering。实现 Ward's linkage，并生成 dendrogram（用 merges 的嵌套列表表示）。在不同层级切断它，并与 K-Means 结果比较。
3. 构建一个简单 anomaly detection pipeline：在同一数据上运行 DBSCAN 和 GMM，标记两种方法都认为是 outlier 的点（DBSCAN 中的 noise，GMM 中低概率点）。衡量重叠度，并讨论两种方法什么时候会分歧。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Clustering | “Grouping similar things” | 把数据分成若干子集，使组内相似度高于组间相似度，相似度由具体 distance metric 衡量 |
| Centroid | “The center of a cluster” | 分配给某个 cluster 的所有点的均值；K-Means 用它表示 cluster |
| Inertia | “How tight the clusters are” | 每个点到其所属 centroid 的 squared distances 之和；越低越紧 |
| Silhouette score | “How well-separated clusters are” | 对每个点计算 (b - a) / max(a, b)，a 是组内平均距离，b 是最近其他 cluster 的平均距离 |
| Core point | “A point in a dense region” | DBSCAN 中 eps 距离内至少有 min_samples 个 neighbors 的点 |
| EM algorithm | “Soft K-Means” | Expectation-Maximization：迭代计算 membership probabilities（E-step）并更新 distribution parameters（M-step） |
| Dendrogram | “A tree of clusters” | 展示 hierarchical clustering 中 clusters 合并顺序和距离的树状图 |
| Anomaly | “An outlier” | 不符合预期模式的数据点，可被 DBSCAN 识别为 noise，或被 GMM 识别为低概率点 |

## Further Reading / 延伸阅读

- [Stanford CS229 - Unsupervised Learning](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) - Andrew Ng 关于 clustering 和 EM 的讲义
- [scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html) - 所有 clustering algorithms 的实践比较与可视化示例
- [DBSCAN original paper (Ester et al., 1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) - 提出 density-based clustering 的原始论文
