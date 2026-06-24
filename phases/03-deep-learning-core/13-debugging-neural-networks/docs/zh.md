# Debugging Neural Networks / 调试神经网络

> 你的 network 编译通过了，跑起来了，也输出了一个数字。这个数字是错的，但没有任何地方崩溃。欢迎来到最难的一类 debugging：没有 error message 的 debugging。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python, PyTorch
**Prerequisites / 前置知识：** Phase 03 Lessons 01-10 (especially backpropagation, loss functions, optimizers)
**Time / 时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 使用系统化 debugging strategies 诊断常见 neural network failures（NaN loss、flat loss curve、overfitting、oscillation）
- 应用 “overfit one batch” technique，验证 model architecture 和 training loop 是否正确
- 检查 gradient magnitudes、activation distributions 和 weight norms，以识别 vanishing/exploding gradient problems
- 构建一个覆盖 data pipeline、model architecture、loss function、optimizer 和 learning rate issues 的 debugging checklist

## The Problem / 问题

传统软件坏了通常会崩溃。Null pointer 会抛 exception。Type mismatch 会在 compile time 失败。Off-by-one error 会产生明显错误的 output。

Neural networks 不会给你这种奢侈。

一个坏掉的 neural network 会完整运行，打印 loss value，并输出 predictions。Loss 可能还会下降。Predictions 可能看起来也合理。但 model 是悄悄错的：它在学习 shortcuts、memorizing noise，或者收敛到一个无用的 local minimum。Google researchers 估计，60-70% 的 ML debugging 时间都花在 “silent” bugs 上：它们不报错，但会降低 model quality。

一个 working model 和 broken model 之间的区别，往往只是一行放错的位置：漏掉 `zero_grad()`、dimension 转置、learning rate 差了 10 倍。经典的 "Recipe for Training Neural Networks" (2019) 一开头就写道："The most common neural net mistakes are bugs that don't crash."

本课会教你找到这些 bugs。

## The Concept / 概念

### The Debugging Mindset / 调试心态

忘掉 print-and-pray debugging。Neural network debugging 需要系统方法，因为 feedback loop 很慢（每次 training run 可能几分钟到几小时），而 symptoms 很模糊（bad loss 可能意味着 20 种不同事情）。

黄金规则：**从简单开始，一次只增加一个 complexity，并独立验证每个部分。**

```mermaid
flowchart TD
    A["Loss not decreasing"] --> B{"Check learning rate"}
    B -->|"Too high"| C["Loss oscillates or explodes"]
    B -->|"Too low"| D["Loss barely moves"]
    B -->|"Reasonable"| E{"Check gradients"}
    E -->|"All zeros"| F["Dead ReLUs or vanishing gradients"]
    E -->|"NaN/Inf"| G["Exploding gradients"]
    E -->|"Normal"| H{"Check data pipeline"}
    H -->|"Labels shuffled"| I["Random-chance accuracy"]
    H -->|"Preprocessing bug"| J["Model learns noise"]
    H -->|"Data is fine"| K{"Check architecture"}
    K -->|"Too small"| L["Underfitting"]
    K -->|"Too deep"| M["Optimization difficulty"]
```

### Symptom 1: Loss Not Decreasing / 症状 1：Loss 不下降

这是最常见的抱怨。Training loop 在跑，epochs 不断推进，但 loss 保持平坦或剧烈振荡。

**Wrong learning rate.** 太高：loss oscillates 或跳到 NaN。太低：loss 下降得太慢，看起来像是 flat。Adam 从 1e-3 开始。SGD 从 1e-1 或 1e-2 开始。在判断是其他问题之前，始终先试 3 个相隔 10x 的 learning rates（例如 1e-2、1e-3、1e-4）。

**Dead ReLUs.** 如果 ReLU neuron 收到很大的 negative input，它会输出 0，gradient 也是 0。它再也不会激活。如果足够多 neurons 死掉，network 就学不动。检查方式：打印每个 ReLU layer 之后 activations 精确等于 0 的比例。如果 >50% 都死了，切换到 LeakyReLU 或降低 learning rate。

**Vanishing gradients.** 在带 sigmoid 或 tanh activations 的 deep networks 中，gradients 反向传播时会指数级缩小。到达第一层时，几乎是 0。前几层停止学习。修复：使用 ReLU/GELU、加入 residual connections，或使用 batch normalization。

**Exploding gradients.** 相反的问题：gradients 指数级增长。RNNs 和非常 deep networks 中常见。Loss 跳到 NaN。修复：gradient clipping（`torch.nn.utils.clip_grad_norm_`）、降低 learning rate，或加入 normalization。

### Symptom 2: Loss Decreasing But Model is Bad / 症状 2：Loss 在下降，但 model 很差

Loss 下降了。Training accuracy 达到 99%。但 test accuracy 只有 55%。或者 model 在真实 data 上输出无意义结果。

**Overfitting.** Model 记住了 training data，而不是学习 patterns。Training loss 和 validation loss 之间的 gap 随时间扩大。修复：更多 data、dropout、weight decay、early stopping、data augmentation。

**Data leakage.** Test data 泄漏进 training。Accuracy 高得可疑。常见原因：split 之前先 shuffle、preprocessing 使用了完整 dataset 的 statistics、splits 之间有 duplicate samples。修复：先 split，再 preprocess，并检查 duplicates。

**Label errors.** 大多数真实 datasets 中有 5-10% 的 labels 是错的（Northcutt et al., 2021 -- "Pervasive Label Errors in Test Sets"）。Model 会学习 noise。修复：使用 confident learning 找出并修正 mislabeled examples，或使用 loss truncation 忽略 high-loss samples。

### Symptom 3: NaN or Inf in Loss / 症状 3：Loss 中出现 NaN 或 Inf

Loss value 变成 `nan` 或 `inf`。Training 已经死了。

**Learning rate too high.** Gradient updates overshoot 太远，weights 爆炸。修复：降低 10x。

**log(0) or log(negative).** Cross-entropy loss 会计算 `log(p)`。如果 model 输出精确 0 或 negative probability，log 会爆掉。修复：把 predictions clamp 到 `[eps, 1-eps]`，其中 `eps=1e-7`。

**Division by zero.** Batch normalization 会除以 standard deviation。如果 batch 中 values 恒定，std=0。修复：在 denominator 加 epsilon（PyTorch 默认这样做，但 custom implementations 可能没有）。

**Numerical overflow.** 大 activations 进入 `exp()` 会产生 Inf。Softmax 特别容易遇到。修复：指数化前先减去 max（log-sum-exp trick）。

### Technique 1: Gradient Checking / 技巧 1：Gradient checking

把 analytical gradients（来自 backprop）与 numerical gradients（来自 finite differences）比较。如果它们不一致，你的 backward pass 有 bug。

Parameter `w` 的 numerical gradient：

```
grad_numerical = (loss(w + eps) - loss(w - eps)) / (2 * eps)
```

Agreement metric（relative difference）：

```
rel_diff = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

如果 `rel_diff < 1e-5`：正确。如果 `rel_diff > 1e-3`：几乎肯定有 bug。

```mermaid
flowchart LR
    A["Parameter w"] --> B["w + eps"]
    A --> C["w - eps"]
    B --> D["Forward pass"]
    C --> E["Forward pass"]
    D --> F["loss+"]
    E --> G["loss-"]
    F --> H["(loss+ - loss-) / 2eps"]
    G --> H
    H --> I["Compare to backprop gradient"]
```

### Technique 2: Activation Statistics / 技巧 2：Activation statistics

训练期间监控每层之后 activations 的 mean 和 standard deviation。健康的 networks 会让 activations 保持 mean 接近 0、std 接近 1（normalization 之后），至少也应该有界。

| Health indicator / 健康指标 | Mean | Std | Diagnosis / 诊断 |
|-----------------|------|-----|-----------|
| Healthy | ~0 | ~1 | Network 正常学习 |
| Saturated | >>0 or <<0 | ~0 | Activations 卡在极端值 |
| Dead | 0 | 0 | Neurons 死亡（全 zero） |
| Exploding | >>10 | >>10 | Activations 无界增长 |

### Technique 3: Gradient Flow Visualization / 技巧 3：Gradient flow 可视化

绘制每一层的 average gradient magnitude。健康 network 中，各层 gradient magnitudes 应该大致相近。如果 early layers 的 gradients 比 later layers 小 1000 倍，你就有 vanishing gradients。

```mermaid
graph LR
    subgraph "Healthy Gradient Flow"
        L1["Layer 1<br/>grad: 0.05"] --- L2["Layer 2<br/>grad: 0.04"] --- L3["Layer 3<br/>grad: 0.06"] --- L4["Layer 4<br/>grad: 0.05"]
    end
```

```mermaid
graph LR
    subgraph "Vanishing Gradient Flow"
        V1["Layer 1<br/>grad: 0.0001"] --- V2["Layer 2<br/>grad: 0.003"] --- V3["Layer 3<br/>grad: 0.02"] --- V4["Layer 4<br/>grad: 0.08"]
    end
```

### Technique 4: The Overfit-One-Batch Test / 技巧 4：Overfit-one-batch 测试

这是 deep learning 中最重要的单个 debugging technique。

取一个小 batch（8-32 samples）。在它上面训练 100+ iterations。Loss 应该接近 zero，training accuracy 应该达到 100%。如果做不到，你的 model 或 training loop 有根本性 bug，不要继续完整训练。

这个测试能抓到：
- Broken loss functions
- Broken backward passes
- Architecture 太小，表示不了 data
- Optimizer 没有连接到 model parameters
- Data 和 labels 错位

它只需要 30 秒，却能节省数小时完整 training runs 的调试时间。

### Technique 5: Learning Rate Finder / 技巧 5：Learning rate finder

Leslie Smith（2017）提出：在一个 epoch 中把 learning rate 从很小（1e-7）扫到很大（10），同时记录 loss。绘制 loss vs learning rate。最优 learning rate 大约是 loss 开始最快下降位置的 1/10。

```mermaid
graph TD
    subgraph "LR Finder Plot"
        direction LR
        A["1e-7: loss=2.3"] --> B["1e-5: loss=2.3"]
        B --> C["1e-3: loss=1.8"]
        C --> D["1e-2: loss=0.9 -- steepest"]
        D --> E["1e-1: loss=0.5"]
        E --> F["1.0: loss=NaN -- too high"]
    end
```

这个例子中的最佳 LR：约 1e-3（steepest point 前一个数量级）。

### Common PyTorch Bugs / 常见 PyTorch bugs

这些是 PyTorch 社区浪费最多时间的 bugs：

| Bug | Symptom / 症状 | Fix / 修复 |
|-----|---------|-----|
| Forgetting `optimizer.zero_grad()` | Gradients 跨 batches 累积，loss oscillates | 在 `loss.backward()` 前加入 `optimizer.zero_grad()` |
| Forgetting `model.eval()` at test time | Dropout 和 batch norm 行为不同，test accuracy 每次变化 | 加入 `model.eval()` 和 `torch.no_grad()` |
| Wrong tensor shapes | Silent broadcasting 产生错误结果但不报错 | Debugging 时在每个 operation 后打印 shapes |
| CPU/GPU mismatch | `RuntimeError: expected CUDA tensor` | 对 model 和 data 都使用 `.to(device)` |
| Not detaching tensors | Computation graph 无限增长，OOM | 使用 `.detach()` 或 `with torch.no_grad()` |
| In-place operations breaking autograd | `RuntimeError: modified by in-place operation` | 把 `x += 1` 替换为 `x = x + 1` |
| Data not normalized | Loss 卡在 random-chance level | 把 inputs normalize 到 mean=0、std=1 |
| Labels as wrong dtype | Cross-entropy expects `Long`, got `Float` | Cast labels: `labels.long()` |

### The Master Debugging Table / 总调试表

| Symptom / 症状 | Likely cause / 可能原因 | First thing to try / 第一件事 |
|---------|-------------|-------------------|
| Loss stuck at -log(1/num_classes) | Model 正在预测 uniform distribution | 检查 data pipeline，验证 labels 与 inputs 匹配 |
| Loss NaN after a few steps | Learning rate 太高 | 把 LR 降低 10x |
| Loss NaN immediately | log(0) 或 division by zero | 给 log/division operations 加 epsilon |
| Loss oscillating wildly | LR 太高或 batch size 太小 | 降低 LR，增大 batch size |
| Loss decreasing then plateaus | Fine-tuning phase 中 LR 太高 | 加 LR schedule（cosine 或 step decay） |
| Training acc high, test acc low | Overfitting | 加 dropout、weight decay、更多 data |
| Training acc = test acc = chance | Model 完全没学到 | 运行 overfit-one-batch test |
| Training acc = test acc but both low | Underfitting | 更大的 model、更多 layers、更多 features |
| Gradients all zero | Dead ReLUs 或 detached computation graph | 切到 LeakyReLU，检查 `.requires_grad` |
| Out of memory during training | Batch 太大或 graph 没释放 | 降低 batch size，eval 用 `torch.no_grad()` |

```figure
learning-curves
```

## Build It / 动手构建

一个 diagnostic toolkit，用来监控 activations、gradients 和 loss curves。你会故意破坏一个 network，再用 toolkit 诊断每个问题。

### Step 1: The NetworkDebugger Class / 第 1 步：NetworkDebugger class

通过 hooks 接入 PyTorch model，记录每一层的 activation 和 gradient statistics。

```python
import torch
import torch.nn as nn
import math


class NetworkDebugger:
    def __init__(self, model):
        self.model = model
        self.activation_stats = {}
        self.gradient_stats = {}
        self.loss_history = []
        self.lr_losses = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.LeakyReLU)):
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self.hooks.append(hook)
                hook = module.register_full_backward_hook(self._make_gradient_hook(name))
                self.hooks.append(hook)

    def _make_activation_hook(self, name):
        def hook(module, input, output):
            with torch.no_grad():
                out = output.detach().float()
                self.activation_stats[name] = {
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "fraction_zero": (out == 0).float().mean().item(),
                    "min": out.min().item(),
                    "max": out.max().item(),
                }
        return hook

    def _make_gradient_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                with torch.no_grad():
                    grad = grad_output[0].detach().float()
                    self.gradient_stats[name] = {
                        "mean": grad.mean().item(),
                        "std": grad.std().item(),
                        "abs_mean": grad.abs().mean().item(),
                        "max": grad.abs().max().item(),
                    }
        return hook

    def record_loss(self, loss_value):
        self.loss_history.append(loss_value)

    def check_loss_health(self):
        if len(self.loss_history) < 2:
            return "NOT_ENOUGH_DATA"
        recent = self.loss_history[-10:]
        if any(math.isnan(v) or math.isinf(v) for v in recent):
            return "NAN_OR_INF"
        if len(self.loss_history) >= 20:
            first_half = sum(self.loss_history[:10]) / 10
            second_half = sum(self.loss_history[-10:]) / 10
            if second_half >= first_half * 0.99:
                return "NOT_DECREASING"
        if len(recent) >= 5:
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            if max(diffs) - min(diffs) > 2 * abs(sum(diffs) / len(diffs)):
                return "OSCILLATING"
        return "HEALTHY"

    def check_activations(self):
        issues = []
        for name, stats in self.activation_stats.items():
            if stats["fraction_zero"] > 0.5:
                issues.append(f"DEAD_NEURONS: {name} has {stats['fraction_zero']:.0%} zero activations")
            if abs(stats["mean"]) > 10:
                issues.append(f"EXPLODING_ACTIVATIONS: {name} mean={stats['mean']:.2f}")
            if stats["std"] < 1e-6:
                issues.append(f"COLLAPSED_ACTIVATIONS: {name} std={stats['std']:.2e}")
        return issues if issues else ["HEALTHY"]

    def check_gradients(self):
        issues = []
        grad_magnitudes = []
        for name, stats in self.gradient_stats.items():
            grad_magnitudes.append((name, stats["abs_mean"]))
            if stats["abs_mean"] < 1e-7:
                issues.append(f"VANISHING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
            if stats["abs_mean"] > 100:
                issues.append(f"EXPLODING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
        if len(grad_magnitudes) >= 2:
            first_mag = grad_magnitudes[0][1]
            last_mag = grad_magnitudes[-1][1]
            if last_mag > 0 and first_mag / last_mag > 100:
                issues.append(f"GRADIENT_RATIO: first/last = {first_mag/last_mag:.0f}x (vanishing)")
        return issues if issues else ["HEALTHY"]

    def print_report(self):
        print("\n=== NETWORK DEBUGGER REPORT ===")
        print(f"\nLoss health: {self.check_loss_health()}")
        if self.loss_history:
            print(f"  Last 5 losses: {[f'{v:.4f}' for v in self.loss_history[-5:]]}")
        print("\nActivation diagnostics:")
        for item in self.check_activations():
            print(f"  {item}")
        print("\nGradient diagnostics:")
        for item in self.check_gradients():
            print(f"  {item}")
        print("\nPer-layer activation stats:")
        for name, stats in self.activation_stats.items():
            print(f"  {name}: mean={stats['mean']:.4f} std={stats['std']:.4f} zero={stats['fraction_zero']:.1%}")
        print("\nPer-layer gradient stats:")
        for name, stats in self.gradient_stats.items():
            print(f"  {name}: abs_mean={stats['abs_mean']:.2e} max={stats['max']:.2e}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
```

### Step 2: The Overfit-One-Batch Test / 第 2 步：Overfit-one-batch test

```python
def overfit_one_batch(model, x_batch, y_batch, criterion, lr=0.01, steps=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print("\n=== OVERFIT ONE BATCH TEST ===")
    print(f"Batch size: {x_batch.shape[0]}, Steps: {steps}")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                preds = (output > 0).float() if output.shape[-1] == 1 else output.argmax(dim=1)
                targets = y_batch if y_batch.dim() == 1 else y_batch.squeeze()
                acc = (preds.squeeze() == targets).float().mean().item()
            print(f"  Step {step:3d} | Loss: {loss.item():.6f} | Accuracy: {acc:.1%}")

    final_loss = loss.item()
    if final_loss > 0.1:
        print(f"\n  FAIL: Loss did not converge ({final_loss:.4f}). Model or training loop is broken.")
        return False
    print(f"\n  PASS: Loss converged to {final_loss:.6f}")
    return True
```

### Step 3: Learning Rate Finder / 第 3 步：Learning rate finder

```python
def find_learning_rate(model, x_data, y_data, criterion, start_lr=1e-7, end_lr=10, steps=100):
    import copy
    original_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / steps)

    model.train()
    results = []
    best_loss = float("inf")
    current_lr = start_lr

    print("\n=== LEARNING RATE FINDER ===")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)

        if math.isnan(loss.item()) or loss.item() > best_loss * 10:
            break

        best_loss = min(best_loss, loss.item())
        results.append((current_lr, loss.item()))

        loss.backward()
        optimizer.step()

        current_lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    model.load_state_dict(original_state)

    if len(results) < 10:
        print("  Could not complete LR sweep -- loss diverged too quickly")
        return results

    min_loss_idx = min(range(len(results)), key=lambda i: results[i][1])
    suggested_lr = results[max(0, min_loss_idx - 10)][0]

    print(f"  Swept {len(results)} steps from {start_lr:.0e} to {results[-1][0]:.0e}")
    print(f"  Minimum loss {results[min_loss_idx][1]:.4f} at lr={results[min_loss_idx][0]:.2e}")
    print(f"  Suggested learning rate: {suggested_lr:.2e}")

    return results
```

### Step 4: Gradient Checker / 第 4 步：Gradient checker

```python
def _flat_to_multi_index(flat_idx, shape):
    multi_idx = []
    remaining = flat_idx
    for dim in reversed(shape):
        multi_idx.insert(0, remaining % dim)
        remaining //= dim
    return tuple(multi_idx)


def gradient_check(model, x, y, criterion, eps=1e-4):
    model.train()
    x_double = x.double()
    y_double = y.double()
    model_double = model.double()

    print("\n=== GRADIENT CHECK ===")
    overall_max_diff = 0
    checked = 0

    for name, param in model_double.named_parameters():
        if not param.requires_grad:
            continue

        layer_max_diff = 0

        model_double.zero_grad()
        output = model_double(x_double)
        loss = criterion(output, y_double)
        loss.backward()
        analytical_grad = param.grad.clone()

        num_checks = min(5, param.numel())
        for i in range(num_checks):
            idx = _flat_to_multi_index(i, param.shape)
            original = param.data[idx].item()

            param.data[idx] = original + eps
            with torch.no_grad():
                loss_plus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original - eps
            with torch.no_grad():
                loss_minus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = analytical_grad[idx].item()

            denom = max(abs(numerical), abs(analytical), 1e-8)
            rel_diff = abs(numerical - analytical) / denom

            layer_max_diff = max(layer_max_diff, rel_diff)
            checked += 1

        overall_max_diff = max(overall_max_diff, layer_max_diff)
        status = "OK" if layer_max_diff < 1e-5 else "MISMATCH"
        print(f"  {name}: max_rel_diff={layer_max_diff:.2e} [{status}]")

    model.float()

    print(f"\n  Checked {checked} parameters")
    if overall_max_diff < 1e-5:
        print("  PASS: Gradients match (rel_diff < 1e-5)")
    elif overall_max_diff < 1e-3:
        print("  WARN: Small differences (1e-5 < rel_diff < 1e-3)")
    else:
        print("  FAIL: Gradient mismatch detected (rel_diff > 1e-3)")
    return overall_max_diff
```

### Step 5: Deliberately Broken Networks / 第 5 步：故意损坏 networks

现在把 toolkit 应用到 broken networks，并诊断每个问题。

```python
def demo_broken_networks():
    torch.manual_seed(42)
    x = torch.randn(64, 10)
    y = (x[:, 0] > 0).long()

    print("\n" + "=" * 60)
    print("BUG 1: Learning rate too high (lr=10)")
    print("=" * 60)
    model1 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger1 = NetworkDebugger(model1)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=10.0)
    criterion = nn.CrossEntropyLoss()
    for step in range(20):
        optimizer1.zero_grad()
        out = model1(x)
        loss = criterion(out, y)
        debugger1.record_loss(loss.item())
        loss.backward()
        optimizer1.step()
    debugger1.print_report()
    debugger1.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 2: Dead ReLUs from bad initialization")
    print("=" * 60)
    model2 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    with torch.no_grad():
        for m in model2.modules():
            if isinstance(m, nn.Linear):
                m.weight.fill_(-1.0)
                m.bias.fill_(-5.0)
    debugger2 = NetworkDebugger(model2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for step in range(50):
        optimizer2.zero_grad()
        out = model2(x)
        loss = criterion(out, y)
        debugger2.record_loss(loss.item())
        loss.backward()
        optimizer2.step()
    debugger2.print_report()
    debugger2.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 3: Missing zero_grad (gradients accumulate)")
    print("=" * 60)
    model3 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger3 = NetworkDebugger(model3)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01)
    for step in range(50):
        out = model3(x)
        loss = criterion(out, y)
        debugger3.record_loss(loss.item())
        loss.backward()
        optimizer3.step()
    debugger3.print_report()
    debugger3.remove_hooks()

    print("\n" + "=" * 60)
    print("HEALTHY NETWORK: Correct setup for comparison")
    print("=" * 60)
    model_good = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger_good = NetworkDebugger(model_good)
    optimizer_good = torch.optim.Adam(model_good.parameters(), lr=1e-3)
    for step in range(50):
        optimizer_good.zero_grad()
        out = model_good(x)
        loss = criterion(out, y)
        debugger_good.record_loss(loss.item())
        loss.backward()
        optimizer_good.step()
    debugger_good.print_report()
    debugger_good.remove_hooks()

    print("\n" + "=" * 60)
    print("OVERFIT-ONE-BATCH TEST (healthy model)")
    print("=" * 60)
    model_test = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    overfit_one_batch(model_test, x[:8], y[:8], criterion)

    print("\n" + "=" * 60)
    print("LEARNING RATE FINDER")
    print("=" * 60)
    model_lr = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    find_learning_rate(model_lr, x, y, criterion)

    print("\n" + "=" * 60)
    print("GRADIENT CHECK")
    print("=" * 60)
    model_grad = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 2))
    gradient_check(model_grad, x[:4], y[:4], criterion)
```

## Use It / 应用它

### PyTorch Built-in Tools / PyTorch 内置工具

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

with torch.autograd.detect_anomaly():
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.abs().mean():.2e}")
```

### Weights & Biases Integration / Weights & Biases 集成

```python
import wandb

wandb.init(project="debug-training")

for epoch in range(100):
    loss = train_one_epoch()
    wandb.log({
        "loss": loss,
        "lr": optimizer.param_groups[0]["lr"],
        "grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")),
    })

    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"grad/{name}": wandb.Histogram(param.grad.cpu().numpy())})
```

### TensorBoard / TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/debug_experiment")

for epoch in range(100):
    loss = train_one_epoch()
    writer.add_scalar("Loss/train", loss, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param, epoch)
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)
```

### The Debug Checklist (Before Full Training) / Debug checklist（完整训练前）

1. 运行 overfit-one-batch test。如果失败，停止。
2. 打印 model summary，确认 parameter count 合理。
3. 用 random data 跑一次 forward pass，检查 output shape。
4. 训练 5 epochs，确认 loss 下降。
5. 检查 activation statistics：没有 dead layers，没有 explosions。
6. 检查 gradient flow：没有 vanishing，也没有 exploding。
7. 验证 data pipeline：打印 5 个 random samples 及其 labels。

## Ship It / 交付它

本课产出：
- `outputs/prompt-nn-debugger.md` -- 一个用于诊断 neural network training failures 的 prompt
- `outputs/skill-debug-checklist.md` -- 一个用于调试 training issues 的 decision-tree checklist

关键 debugging deployment patterns：
- 给 production training scripts 添加 monitoring hooks
- 每 N steps 把 activation 和 gradient statistics 记录到 W&B 或 TensorBoard
- 为 NaN loss、dead neurons（>80% zero）或 gradient explosion 实现 automatic alerts
- 每次修改 architectures 或 data pipelines 时，都运行 overfit-one-batch test

## Exercises / 练习

1. **增加 exploding gradient detector。** 修改 `NetworkDebugger`，当 gradients 超过阈值时检测出来，并自动建议 gradient clipping value。在一个没有 normalization 的 20-layer network 上测试它。

2. **构建 dead neuron resurrector。** 写一个 function，识别 dead ReLU neurons（总是输出 0），并用 Kaiming initialization 重新初始化它们的 incoming weights。展示它能恢复一个 >70% neurons 死亡的 network。

3. **给 learning rate finder 加 plotting。** 扩展 `find_learning_rate`，把结果保存为 CSV，再写一个单独 script 读取 CSV 并用 matplotlib 显示 LR vs loss curve。找出 ResNet-18 在 CIFAR-10 上的最优 LR。

4. **创建 data pipeline validator。** 写一个 function，检查：train/test splits 之间的 duplicate samples、label distribution imbalance（>10:1 ratio）、input normalization（mean near 0、std near 1），以及 data 中的 NaN/Inf values。在故意损坏的 dataset 上运行它。

5. **调试一个真实 failure。** 拿 Lesson 10 的 mini-framework，引入一个 subtle bug（例如在 backward 中转置 weight matrix），并用 gradient checking 精确定位哪个 parameter 的 gradients 不正确。记录 debugging process。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Silent bug | “能跑但结果差” | 不产生 error、但会降低 model quality 的 bug，是 ML 中最主要的 failure mode |
| Dead ReLU | “Neurons 死了” | ReLU neuron 的 input 总为负，因此 output 为 0，并永久收到 0 gradient |
| Vanishing gradients | “Early layers 不学了” | Gradients 穿过 layers 时指数级缩小，让 early layers 中的 weights 实际被冻结 |
| Exploding gradients | “Loss 变成 NaN” | Gradients 穿过 layers 时指数级增长，导致 weight updates 大到 overflow |
| Gradient checking | “验证 backprop 正确” | 把 backprop 的 analytical gradients 与 finite differences 的 numerical gradients 做比较 |
| Overfit-one-batch | “最重要的 debug test” | 在一个小 batch 上训练，验证 model 能否学习；如果不能，说明有根本问题 |
| LR finder | “扫一遍找到正确 learning rate” | 在一个 epoch 中指数提高 learning rate，并选择 loss diverges 前的 rate |
| Data leakage | “Test data 泄漏进 training” | Test set 信息污染 training，产生虚高 accuracy |
| Activation statistics | “监控 layer health” | 跟踪每层 output 的 mean、std 和 zero-fraction，用于检测 dead、saturated 或 exploding neurons |
| Gradient clipping | “限制 gradient magnitude” | 当 gradients norm 超过阈值时缩小它们，防止 exploding gradient updates |

## Further Reading / 延伸阅读

- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017) -- 引入 learning rate range test（LR finder）的论文
- Northcutt et al., "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks" (2021) -- 展示 ImageNet、CIFAR-10 和其他主要 benchmarks 中 3-6% labels 是错的
- Zhang et al., "Understanding Deep Learning Requires Rethinking Generalization" (2017) -- 展示 neural networks 可以记住 random labels，也解释了 overfit-one-batch test 为什么有效
- PyTorch documentation on `torch.autograd.detect_anomaly` and `torch.autograd.set_detect_anomaly` for built-in NaN/Inf detection
