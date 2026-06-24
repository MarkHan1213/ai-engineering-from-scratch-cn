# Introduction to JAX / JAX 入门

> PyTorch 会 mutate tensors。TensorFlow 会 build graphs。JAX 会 compile pure functions。最后这一点会改变你思考 deep learning 的方式。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 03 Lessons 01-10, basic NumPy
**Time / 时间：** 约 90 分钟

## Learning Objectives / 学习目标

- 使用 JAX 的 functional API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写 pure-function neural network code
- 解释 PyTorch 的 eager mutation 与 JAX 的 functional compilation model 之间的关键设计差异
- 应用 jit compilation 和 vmap vectorization，让 training loops 相比 naive Python 更快
- 在 JAX 中训练一个简单 network，并把显式 state management 与 PyTorch 的 object-oriented approach 做对比

## The Problem / 问题

你已经知道如何在 PyTorch 中构建 neural networks。定义一个 `nn.Module`，调用 `.backward()`，让 optimizer step。它能工作，数百万人都在用。

但 PyTorch 的 DNA 里有一个限制：它会在 Python 中 eager 地逐个 trace operations。每个 `tensor + tensor` 都是一次单独 kernel launch。每个 training step 都会重新解释同一段 Python code。这在普通规模下没问题，直到你需要跨 2,048 个 TPUs 训练一个 540-billion-parameter model。此时 overhead 会杀死你。

Google DeepMind 用 JAX 训练 Gemini。Anthropic 用 JAX 训练 Claude。这些都不是小规模操作，而是地球上最大的 neural network training runs。它们选择 JAX，是因为 JAX 把你的 training loop 当作可编译程序，而不是一串 Python calls。

JAX 是带三种超能力的 NumPy：automatic differentiation、JIT compilation to XLA，以及 automatic vectorization。你写一个处理单个 example 的 function。JAX 给你一个能处理 batch、计算 gradients、compile 成 machine code、并跨多个 devices 运行的 function。原始 function 不需要改。

## The Concept / 概念

### The JAX Philosophy / JAX 哲学

JAX 是 functional framework。没有 classes，没有 mutable state，没有 `.backward()` method。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

这不是风格偏好，而是 compiler constraint。JIT compilation 要求 pure functions：同样 inputs 总是产生同样 outputs，没有 side effects。正是这个限制让 100x speedups 成为可能。

### jax.numpy: The Familiar Surface / jax.numpy：熟悉的表层

JAX 在 accelerators 上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同 function names，相同 broadcasting rules，相同 slicing semantics。但 arrays 运行在 GPU/TPU 上，而且每个 operation 都能被 compiler trace。

一个关键差异：JAX arrays 是 immutable。不能写 `a[0] = 5`。要写：`a = a.at[0].set(5)`。这会别扭一周，然后你会明白：immutability 正是 `grad`、`jit` 和 `vmap` 这类 transformations 可组合的原因。

### jax.grad: Functional Autodiff / 函数式自动求导

PyTorch 把 gradients 附着到 tensors（`.grad`）。JAX 把 gradients 附着到 functions。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一个 function，并返回一个计算 gradient 的新 function。没有 `.backward()` call，也没有存储在 tensors 上的 computation graph。Gradient 只是另一个可以 call、compose 或 JIT-compile 的 function。

它可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导、三阶导、Jacobians、Hessians，都通过组合 `grad` 实现。PyTorch 也能做到（`torch.autograd.functional.hessian`），但更像后加功能。在 JAX 中，这是基础。

限制是：`grad` 只适用于 pure functions。里面不能随便 print（它们会在 tracing 时运行，而不是执行时）。不能 mutate external state。没有 explicit key management，就不能随机数生成。

### jit: Compile to XLA / jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX 会 trace function：它记录发生了哪些 operations，但不真正执行它们。然后把 trace 交给 XLA（Accelerated Linear Algebra），这是 Google 面向 TPUs 和 GPUs 的 compiler。XLA 会 fuse operations、消除多余 memory copies，并生成优化过的 machine code。

后续调用会完全跳过 Python。Compiled code 直接在 accelerator 上以 C++ speed 运行。

JIT 有帮助的场景：
- Training steps（同一 computation 重复数千次）
- Inference（同一 model，不同 inputs）
- 任何以相似 shape inputs 调用多次的 function

JIT 有害的场景：
- Python control flow 依赖 values 的函数（比如 `if x > 0` 且 x 是 traced array）
- One-shot computations（compilation overhead 超过 runtime）
- Debugging（tracing 会隐藏实际执行）

Control flow restriction 是真实存在的。`jax.lax.cond` 替代 `if/else`。`jax.lax.scan` 替代 `for` loops。这些不是可选项，而是 compilation 的代价。

### vmap: Automatic Vectorization / vmap：自动向量化

你写一个处理单个 example 的 function：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 会把它提升为处理 batch 的 function：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的意思是：不要在 `params` 上 batch（共享 params），在 `x` 的 axis 0 上 batch。不需要手写 `for` loop。不需要 reshaping。不需要手动穿针引线 batch dimension。JAX 会找出 batch dimension，并 vectorize 整个 computation。

这不是语法糖。`vmap` 会生成 fused vectorized code，比 Python loop 快 10-100 倍。而且它能和 `jit`、`grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Per-example gradients，一行代码。在 PyTorch 中没有 hacks 很难做到。

### pmap: Data Parallelism Across Devices / pmap：跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 会把 function 复制到所有可用 devices（GPUs/TPUs），并切分 batch。Function 内部用 `jax.lax.pmean` 和 `jax.lax.psum` 在 devices 之间同步 gradients。

Google 使用 `pmap`（以及它的后继 `shard_map`）跨数千个 TPU v5e chips 训练 Gemini。Programming model 是：写 single-device 版本，用 `pmap` 包起来，完成。

### Pytrees: The Universal Data Structure / Pytrees：通用数据结构

JAX 操作 “pytrees”，也就是 lists、tuples、dicts 和 arrays 的嵌套组合。你的 model parameters 是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX transformation，包括 `grad`、`jit`、`vmap`，都知道如何遍历 pytrees。`jax.tree.map(f, tree)` 会把 `f` 应用到每个 leaf。这就是 optimizers 一次性更新所有 parameters 的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` method，没有 parameter registration。Tree structure 就是 model。

### Functional vs Object-Oriented / 函数式 vs 面向对象

PyTorch 把 state 存在 objects 里：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带显式 state 的 pure functions：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Params 被传入。没有东西被存储，没有东西被 mutate。这让每个 function 都可测试、可组合、可编译。它也意味着你要自己管理 params，或者使用 Flax、Equinox 这样的 library。

### The JAX Ecosystem / JAX 生态

JAX 给你 primitives。Libraries 给你 ergonomics：

| Library | Role / 角色 | Style / 风格 |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax 是标准 optimizer library。它把 gradient transformation（Adam、SGD、clipping）与 parameter update 分开，让组合变得很简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### When to Use JAX vs PyTorch / 何时使用 JAX 或 PyTorch

| Factor / 因素 | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class（Google 同时构建两者） | Community-maintained（torch_xla） |
| GPU support | Good（CUDA via XLA） | Best-in-class（native CUDA） |
| Debugging | Hard（tracing + compilation） | Easy（eager、line-by-line） |
| Ecosystem | Research-focused（Flax、Equinox） | Massive（HuggingFace、torchvision 等） |
| Hiring | Niche（Google/DeepMind/Anthropic） | Mainstream（everywhere） |
| Large-scale training | Superior（XLA、pmap、mesh） | Good（FSDP、DeepSpeed） |
| Prototyping speed | Slower（functional overhead） | Faster（mutate and go） |
| Production inference | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| Who uses it | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

诚实答案是：除非你有明确理由用 JAX，否则用 PyTorch。这些理由包括：TPU access、需要 per-example gradients、大规模 multi-device training，或者你在 Google/DeepMind/Anthropic 工作。

### Random Numbers in JAX / JAX 中的随机数

JAX 没有 global random state。每个 random operation 都需要显式 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始这很烦。但它能保证跨 devices 和 compilations 的 reproducibility，这是 PyTorch 的 `torch.manual_seed` 在 multi-GPU settings 中无法保证的性质。

```figure
batchnorm-effect
```

## Build It / 动手构建

### Step 1: Setup and Data / 第 1 步：Setup 与 data

我们会用 JAX 和 Optax 在 MNIST 上训练 3-layer MLP。784 inputs，两个 hidden layers，分别为 256 和 128 neurons，10 个 output classes。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Step 2: Initialize Parameters / 第 2 步：初始化 parameters

没有 class。只是一个返回 pytree 的 function：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He-initialization，手动完成。三个 PRNG keys 从一个 seed 拆分出来。每个 weight 都是 nested dict 中的 immutable array。

### Step 3: Forward Pass / 第 3 步：Forward pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Pure functions。Params 进去，prediction 出来。没有 `self`，没有 stored state。`loss_fn` 从零计算 cross-entropy：softmax、log、negative mean。

### Step 4: JIT-Compiled Training Step / 第 4 步：JIT 编译的 training step

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 会在一次 pass 中同时返回 loss value 和 gradients。`@jax.jit` decorator 会把两个 functions 都编译到 XLA。第一次调用之后，每个 training step 都不会再接触 Python。

### Step 5: Training Loop / 第 5 步：Training loop

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epochs。约 97% test accuracy。第一个 epoch 慢（JIT compilation），epochs 2-10 很快。

注意缺失的东西：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个 update 是一次 composed function call。Gradients 被计算、被 Adam 转换、并应用到 parameters，全都在 `train_step` 内完成。

## Use It / 应用它

### Flax: The Google Standard / Flax：Google 标准

Flax 是最常见的 JAX neural network library。它把 `nn.Module` 加了回来，但保留显式 state management：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构和 PyTorch 相同，但 `params` 与 model 分离。`model.init()` 创建 params。`model.apply(params, x)` 运行 forward pass。Model object 本身没有 state。

### Equinox: The Pythonic Alternative / Equinox：更 Pythonic 的替代方案

Equinox（Patrick Kidger）把 models 表示为 pytrees：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Model 本身就是 pytree。不需要 `.apply()`。Parameters 就是 model 的 leaves。这更接近 JAX 的思维方式。

### Optax: Composable Optimizers / Optax：可组合优化器

Optax 把 gradient transformation 和 update 解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping、learning rate warmup、weight decay 都可以作为 transforms 链式组合。每个 transform 看到 gradients，修改它们，然后传给下一个。没有 monolithic optimizer class。

## Ship It / 交付它

**Installation:**

```bash
pip install jax jaxlib optax flax
```

For GPU support:

```bash
pip install jax[cuda12]
```

For TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Performance gotchas:**

- 第一次 JIT call 很慢（compilation）。Benchmark 前先 warm up。
- 避免在 JIT 内部用 Python loops 遍历 JAX arrays。使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 可以在 JIT 内工作。普通 `print()` 不行。
- 使用 `jax.profiler` 或 TensorBoard 做 profile。XLA compilation 可能隐藏 bottlenecks。
- JAX 默认会预分配 75% GPU memory。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可以关闭。

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` -- 一个用于选择正确 JAX optimizer configuration 的 prompt
- `outputs/skill-jax-patterns.md` -- 一个覆盖 JAX functional patterns 的 skill

## Exercises / 练习

1. 给 MLP 增加 dropout。在 JAX 中，dropout 需要 PRNG key，因此要把 key 穿过 forward pass，并为每个 dropout layer split。比较有无 dropout 的 test accuracy。

2. 使用 `jax.vmap` 为一个包含 32 张 MNIST images 的 batch 计算 per-example gradients。计算每个 example 的 gradient norm。哪些 examples 的 gradients 最大？为什么？

3. 把 manual forward function 替换成 generic `mlp_forward(params, x)`，让它适用于任意层数。使用 `jax.tree.leaves` 自动确定 depth。

4. Benchmark 有无 `@jax.jit` 的 training step。各运行 100 steps 并计时。在你的硬件上 speedup 有多大？第一次 call 的 compilation overhead 是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现 gradient clipping。分别在有无 clipping 的情况下训练。绘制 training 期间的 gradient norm，观察效果。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| XLA | “让 JAX 变快的东西” | Accelerated Linear Algebra，一个 compiler，会从 computation graph 中 fuse operations 并生成优化过的 GPU/TPU kernels |
| JIT | “Just-in-time compilation” | JAX 在第一次 call 时 trace function、compile 到 XLA，后续 calls 运行 compiled version |
| Pure function | “没有 side effects” | 输出只依赖 inputs 的 function；没有 global state、mutation，随机性也必须显式传 key |
| vmap | “Auto-batching” | 把处理单个 example 的 function 转换成处理 batch 的 function，无需重写 |
| pmap | “Auto-parallelism” | 把 function 复制到多个 devices，并切分 input batch |
| Pytree | “Nested dict of arrays” | JAX 可以遍历和 transform 的任意嵌套 lists、tuples、dicts 和 arrays |
| Tracing | “记录 computation” | JAX 使用 abstract values 执行 function 来构建 computation graph，而不是计算真实结果 |
| Functional autodiff | “function 的 grad” | 通过转换 functions 来计算 derivatives，而不是把 gradient storage 附在 tensors 上 |
| Optax | “JAX 的 optimizer library” | 一个可组合的 gradient transformations library，包括 Adam、SGD、clipping、scheduling 等 |
| Flax | “JAX 的 nn.Module” | Google 的 JAX neural network library，在保持 state 显式的同时加入 layer abstractions |

## Further Reading / 延伸阅读

- JAX documentation: https://jax.readthedocs.io/ -- 官方文档，包含很好的 grad、jit 和 vmap 教程
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- 解释设计哲学的原始论文
- Flax documentation: https://flax.readthedocs.io/ -- Google 的 JAX neural network library
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flax 的 Pythonic 替代方案
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 标准 optimizer library
- "You Don't Know JAX" (Colin Raffel, 2020) -- T5 作者之一撰写的 JAX gotchas 和 patterns 实用指南
