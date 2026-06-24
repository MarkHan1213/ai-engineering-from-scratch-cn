# Image Classification / 图像分类

> Classifier 是从 pixels 到类别概率分布的函数。其他部分都是工程管线。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 2 Lesson 09 (Model Evaluation), Phase 3 Lesson 10 (Mini Framework), Phase 4 Lesson 03 (CNNs)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 在 CIFAR-10 上构建端到端 image classification pipeline：dataset、augmentation、model、training loop、evaluation
- 解释每个组件（dataloader、loss、optimizer、scheduler、augmentation）的作用，并预测其中任意一环出错会如何体现在 loss curve 上
- 从零实现 mixup、cutout 和 label smoothing，并说明何时值得加入它们
- 阅读 confusion matrix 和 per-class precision/recall table，用它们诊断 aggregate accuracy 之外的 dataset 和 model failure

## The Problem / 问题

几乎每个上线的视觉任务，在某个层面都会归约成 image classification。Detection 会分类区域，segmentation 会分类 pixel，retrieval 会按与 class centroid 的相似度排序。把 classification 做对，也就是把 dataset loop、augmentation policy、loss、evaluation 做对，是能迁移到本 phase 其他任务的核心能力。

大多数 classification bug 不在 model 里，而在 pipeline 里：坏掉的 normalisation、没有 shuffle 的 training set、会扭曲 label 的 augmentation、被 training data 污染的 validation split、epoch 30 之后悄悄发散的 learning rate。一个正确配置下能在 CIFAR-10 达到 93% 的 CNN，在坏 pipeline 中常常只有 70-75%，而 loss curve 看起来依然很合理。

本课会手写接完整个 pipeline，让每一部分都可检查。你不会使用 `torchvision.datasets` 里任何可能隐藏 bug 的东西。

## The Concept / 概念

### The classification pipeline / classification pipeline

```mermaid
flowchart LR
    A["Dataset<br/>(images + labels)"] --> B["Augment<br/>(random transforms)"]
    B --> C["Normalise<br/>(mean/std)"]
    C --> D["DataLoader<br/>(batch + shuffle)"]
    D --> E["Model<br/>(CNN)"]
    E --> F["Logits<br/>(N, C)"]
    F --> G["Cross-entropy loss"]
    F --> H["Argmax<br/>at eval"]
    G --> I["Backward"]
    I --> J["Optimizer step"]
    J --> K["Scheduler step"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

这个 loop 中的每条线都可能藏 bug。Cross-entropy 接收 raw logits，而不是 softmax outputs，所以在 loss 之前调用任何 `model(x).softmax()` 都会悄悄算出错误 gradient。Augmentation 只作用于 inputs，不作用于 labels，mixup 除外，因为它会混合两者。`optimizer.zero_grad()` 必须每 step 执行一次；漏掉它会累积 gradient，看起来像 learning rate 极度不稳定。每个 bug 都会让 learning curve 变平，但不会抛错。

### Cross-entropy, logits, and softmax / Cross-entropy、logits 与 softmax

Classifier 对每张图输出 `C` 个数字，叫 logits。应用 softmax 会把它们转换成概率分布：

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

Cross-entropy 衡量 correct class 的 negative log probability：

```
CE(z, y) = -log( softmax(z)_y )
        = -z_y + log( sum_j exp(z_j) )
```

右边的形式是 numerically stable 的 log-sum-exp。PyTorch 的 `nn.CrossEntropyLoss` 把 softmax + NLL 融合成一个 op，并直接接收 raw logits。先自己应用 softmax 几乎永远是 bug，因为你会计算 log(softmax(softmax(z)))，这是没有意义的量。

### Why augmentation works / 为什么 augmentation 有效

CNN 因为 weight sharing 对 translation 有 inductive bias，但它没有内置 crop、flip、colour jitter 或 occlusion 的 invariance。教会它这些 invariance 的唯一办法，是展示包含这些变化的 pixels。训练时的每个 random transform 都是在告诉模型：“这两张图 label 相同；请学习忽略差异的 feature。”

```
Original crop:  "dog facing left"
Flip:           "dog facing right"       <- same label, different pixels
Rotate(+15):    "dog, slight tilt"
Colour jitter:  "dog in warmer light"
RandomErasing:  "dog with patch missing"
```

规则：augmentation 必须保持 label 不变。Cutout 和 rotation 用在 digit 上时可能把 “6” 变成 “9”；对这类 dataset，你要使用更小的 rotation range，并选择尊重 digit-specific invariance 的 augmentation。

### Mixup and cutmix / Mixup 与 cutmix

普通 augmentation 改 pixels，但保持 label one-hot。**Mixup** 和 **cutmix** 会打破这一点，同时插值 input 和 label。

```
Mixup:
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix:
  paste a random rectangle of x_j into x_i
  y = area-weighted mix of y_i and y_j
```

它有效的原因是：模型不再记忆尖锐的 one-hot target，而是学习在类别之间插值。Training loss 会升高，test accuracy 会升高。它是任何 classifier 最便宜的 robustness upgrade。

### Label smoothing / Label smoothing

它是 mixup 的近亲。不要用 `[0, 0, 1, 0, 0]` 训练，而是用 `[eps/C, eps/C, 1-eps, eps/C, eps/C]` 这样的 target，`eps` 常取 0.1。它阻止模型产生任意尖锐的 logits，几乎零成本改善 calibration。PyTorch 1.10 起已经内置在 `nn.CrossEntropyLoss(label_smoothing=0.1)`。

### Evaluation beyond accuracy / accuracy 之外的评估

Aggregate accuracy 会隐藏 imbalance。一个 90-10 的 binary classifier 如果永远预测 majority class，也能得到 90%。真正告诉你发生了什么的工具是：

- **Per-class accuracy / 逐类 accuracy**：每个 class 一个数字，立刻暴露表现差的类别。
- **Confusion matrix / 混淆矩阵**：C x C grid，row i col j 表示 true class i 被预测成 class j 的数量；diagonal 是正确结果，off-diagonal 才是模型问题所在。
- **Top-1 / Top-5**：正确 class 是否位于 top 1 或 top 5 predictions；Top-5 对 ImageNet 很重要，因为 “Norwich terrier” 和 “Norfolk terrier” 这类 class 本来就很容易混淆。
- **Calibration (ECE)**：confidence 为 0.8 的 prediction，是否真的有 80% 正确？现代网络系统性 over-confident；可用 temperature scaling 或 label smoothing 修复。

```figure
receptive-field
```

## Build It / 动手构建

### Step 1: A deterministic synthetic dataset / Step 1：确定性的 synthetic dataset

CIFAR-10 存在磁盘上。为了让本课可复现且快速，我们构建一个看起来像 CIFAR 的 synthetic dataset：32x32 RGB image，带有 model 必须学习的 class-specific structure。完全相同的 pipeline 可以不做改动地用于真实 CIFAR-10。

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

每个 class 都有自己的 colour palette 和 frequency pattern，再加上 Gaussian noise，迫使模型学习信号而不是记住 pixel。十个 class，每类一千张图，并进行 permutation。

### Step 2: Normalisation and augmentation / Step 2：normalisation 与 augmentation

每个 vision pipeline 都有的两个 transform。

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

Crop 前使用 reflect-pad，而不是 zero-pad，因为黑边是一种模型可能学会利用、但没有实际价值的信号。

### Step 3: Mixup / Step 3：Mixup

在 training step 内部混合两张 image 和两个 label。这里把它实现为 batch transform，所以它靠近 forward pass，而不是藏在 dataset 里面。

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy` 是针对 soft-label distribution 的 cross-entropy。当 target 恰好是 one-hot 时，它会退化成普通版本。

### Step 4: The training loop / Step 4：training loop

完整配方：对数据跑一遍，每个 batch 做一次 gradient，scheduler 每个 epoch step 一次。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # Training accuracy vs the un-mixed labels `y` is only an approximation
        # when mixup is on (the model saw soft targets, not y). Treat it as a
        # rough progress signal; rely on val accuracy for real performance.
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

每次写 training loop 都要检查五个 invariant：

1. 训练前 `model.train()`，评估前 `model.eval()`，以切换 dropout 和 batchnorm 行为。
2. `.zero_grad()` 在 `.backward()` 前执行。
3. 累积 metric 时使用 `.item()`，避免保留 computation graph。
4. Evaluation 期间使用 `@torch.no_grad()`，节省内存和时间，防止细微事故。
5. 对 raw logits 做 argmax，而不是对 softmax 做 argmax；结果相同，少一个 op。

### Step 5: Put it together / Step 5：组装起来

使用上一课的 `TinyResNet`，训练几个 epoch，然后评估。

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet comes from the previous lesson (03-cnns-lenet-to-resnet).
# Adjust the import path to wherever you stored the previous lesson's code.
from cnns_lenet_to_resnet import TinyResNet  # example placeholder

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

在 synthetic dataset 上，这会在五个 epoch 内达到接近完美的 validation accuracy。这正是重点：pipeline 是正确的，model 能学到可学习的东西。把 dataset 换成真实 CIFAR-10，同一个 loop 不改也能训练到约 90%。

### Step 6: Read the confusion matrix / Step 6：读取 confusion matrix

只有 accuracy 永远无法告诉你模型在哪里失败。Confusion matrix 可以。

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

Rows 是 true classes，columns 是 predictions。如果 classes 3 和 5 之间出现一簇 off-diagonal counts，说明模型会混淆这两个类别，你也就有了 targeted data collection 或 class-specific augmentation 的起点。

## Use It / 应用它

`torchvision` 会把上面的一切包装成惯用组件。对真实 CIFAR-10，完整 pipeline 是四行 transform 加一个 training loop。

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

注意两点：mean/std 是 **dataset-specific**，是在 CIFAR-10 training set 上计算的，不是 ImageNet；reflect pad 是社区默认 crop policy。把 ImageNet stats 复制到这里会漏掉约 1% accuracy，通常直到有人 profile model 才会发现。

## Ship It / 交付它

本课产出：

- `outputs/prompt-classifier-pipeline-auditor.md`：一个 prompt，用来审计 training script 是否满足上面的五个 invariant，并暴露第一个 violation。
- `outputs/skill-classification-diagnostics.md`：一个 skill，给定 confusion matrix 和 class name list，总结 per-class failure，并提出最有影响力的单个修复。

## Exercises / 练习

1. **(Easy / 简单)** 在 synthetic dataset 上分别使用和不使用 mixup 训练同一个模型五个 epoch。绘制两者的 train 和 val loss。解释为什么带 mixup 的 train loss 更高，但 val accuracy 相近或更好。
2. **(Medium / 中等)** 实现 Cutout，也就是在每张 training image 中随机把一个 8x8 square 置零。做 ablation：no augmentation、hflip+crop、hflip+crop+cutout、hflip+crop+mixup。报告每组 val accuracy。
3. **(Hard / 困难)** 构建 CIFAR-100 pipeline（100 classes，相同 input size），复现一次 ResNet-34 训练，使结果落在 published accuracy 的 1% 内。额外任务：扫三个 learning rates 和两个 weight decays，记录到 local CSV，输出最终的 confusion-matrix-top-confusions table。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Logits | “Raw outputs” | 每张图一个 pre-softmax 的 C 维向量；cross-entropy 期待它，而不是 softmax 后的值 |
| Cross-entropy | “loss” | Correct class 的 negative log-probability；在一个稳定 op 中组合 log-softmax 和 NLL |
| DataLoader | “batcher” | 用 shuffling、batching 和可选 multi-worker loading 包装 dataset；一半训练 bug 都会先被怀疑到它头上 |
| Augmentation | “random transforms” | 训练时任何保持 label 不变的 pixel-level transform；教会 CNN 原生不具备的 invariance |
| Mixup / Cutmix | “混两张图” | 同时混合 inputs 和 labels，让 classifier 学习平滑插值，而不是硬边界 |
| Label smoothing | “更软的 target” | 用 (1-eps, eps/(C-1), ...) 替代 one-hot；改善 calibration，并略微提升 accuracy |
| Top-k accuracy | “Top-5” | Correct class 位于 k 个最高概率 prediction 中；用于类别确实容易混淆的 dataset |
| Confusion matrix | “错误在哪里” | C x C table，entry (i, j) 统计 true class i 被预测成 j 的图像数；diagonal 是正确，off-diagonal 告诉你该修什么 |

## Further Reading / 延伸阅读

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/)：单页内最清楚的 training pipeline 导览
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187)：把 ResNet 在 ImageNet 上提升 3-4% 的一组小技巧
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412)：mixup 原论文；三页理论加上有说服力的实验
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599)：证明现代网络 miscalibrated，并用一个 scalar parameter 修复它的论文
