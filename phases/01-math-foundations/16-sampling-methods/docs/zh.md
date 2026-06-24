# Sampling Methods / 采样方法

> 采样是 AI 探索可能性空间的方式。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**时间：** 约 120 分钟

## Learning Objectives / 学习目标

- 只使用 uniform random numbers，从零实现 inverse CDF、rejection 和 importance sampling
- 为 language model token generation 构建 temperature、top-k 和 top-p（nucleus）sampling
- 解释 reparameterization trick，以及它为什么能让 VAEs 中的 sampling 支持 backpropagation
- 运行 Metropolis-Hastings MCMC，从 unnormalized target distribution 中采样

## The Problem / 问题

Language model 处理完你的 prompt 后，会产生一个包含 50,000 个 logits 的向量。词表中每个 token 一个 logit。现在它必须选择一个 token。怎么选？

如果总是选最高概率 token，每个回答都一样。确定、无聊。如果均匀随机选择，输出就是胡言乱语。答案在这两个极端之间，而这个中间位置由 sampling 控制。

Sampling 不只用于 text generation。Reinforcement learning 通过采样 trajectories 来估计 policy gradients。VAEs 从 learned distributions 中采样 latent representations，并让 randomness 支持 backpropagation。Diffusion models 通过采样 noise 并迭代 denoise 生成图像。Monte Carlo methods 估计没有 closed-form solution 的 integrals。MCMC algorithms 探索无法枚举的 high-dimensional posterior distributions。

每个 generative AI system 都是 sampling system。Sampling strategy 决定输出质量、多样性和可控性。本课会从零构建主要 sampling methods，从 uniform random numbers 出发，直到现代 LLMs 和 generative models 使用的技术。

## The Concept / 概念

### Why Sampling Matters / 为什么采样重要

Sampling 在 AI 和 machine learning 中有四个基础角色：

**Generation。** Language models、diffusion models 和 GANs 都通过 sampling 产生输出。Sampling algorithm 直接控制 creativity、coherence 和 diversity。Temperature、top-k 和 nucleus sampling 是工程师每天调的旋钮。

**Training。** Stochastic gradient descent 会采样 mini-batches。Dropout 会采样要 deactivate 的 neurons。Data augmentation 会采样随机变换。Importance sampling 会重加权 samples，降低 reinforcement learning 中的 gradient variance（PPO、TRPO）。

**Estimation。** ML 中很多量没有 closed-form solution。数据分布上的 expected loss、energy-based model 的 partition function、Bayesian inference 中的 evidence。Monte Carlo estimation 都通过 sample average 近似这些量。

**Exploration。** MCMC algorithms 在 Bayesian inference 中探索 posterior distributions。Evolutionary strategies 采样 parameter perturbations。Thompson sampling 在 bandits 中平衡 exploration 与 exploitation。

核心挑战：你只能直接从简单 distributions（uniform、normal）采样。对其他所有目标分布，你需要一种方法，把简单 samples 转换成 target distribution 的 samples。

### Uniform Random Sampling / 均匀随机采样

每种 sampling method 都从这里开始。Uniform random number generator 会产生 [0, 1) 中的值，并且每个等长子区间概率相同。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个 items 的 discrete set 中 uniform sample，生成 U 并返回 floor(n * U)。要从 continuous range [a, b] 中 sample，计算 a + (b - a) * U。

关键洞见：一个 uniform random number 恰好包含产生任意 distribution 的一个 sample 所需的随机性。难点是找到正确 transformation。

### Inverse CDF Method (Inverse Transform Sampling) / 逆 CDF 方法（Inverse Transform Sampling）

Cumulative distribution function（CDF）把 values 映射到 probabilities：

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

Inverse CDF 把 probabilities 映射回 values。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 服从 target distribution。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Exponential distribution 示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

当你能写出 closed form 的 F_inverse 时，它会完美工作。Normal distribution 没有 closed-form inverse CDF，所以我们会使用其他方法，例如 Box-Muller 或 numerical approximation。

**Discrete version：** 对 discrete distributions，构造 cumulative sum 形式的 CDF，生成 U，找到第一个 cumulative sum 超过 U 的 index。这就是 Lesson 06 中 `sample_categorical` 的工作方式。

### Rejection Sampling / 拒绝采样

当你无法反转 CDF，但可以评估 target PDF 到某个常数比例时，rejection sampling 可以工作。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

Bound M 越紧，acceptance rate 越高。在低维（1-3）中，rejection sampling 效果很好。在高维中，acceptance rate 会指数下降，因为 proposal volume 大多被拒绝。这是 rejection sampling 的 curse of dimensionality。

**例子：从 truncated normal 采样。** 在截断范围上使用 uniform proposal。Envelope M 是该范围内 normal PDF 的最大值。

**例子：从半圆采样。** 在外接矩形中 uniform propose。如果点落在半圆内就 accept。这就是 Monte Carlo 计算 pi 的方式：acceptance rate 等于面积比 pi/4。

### Importance Sampling / 重要性采样

有时候你不需要 target distribution p(x) 的 samples。你需要估计 p(x) 下的 expectation，但手里有另一个 distribution q(x) 的 samples。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这在 reinforcement learning 中非常关键。在 PPO（Proximal Policy Optimization）中，你用 old policy pi_old 收集 trajectories，但想优化 new policy pi_new。Importance weight 是 pi_new(a|s) / pi_old(a|s)。PPO 会 clip 这些 weights，防止 new policy 偏离 old policy 太远。

Importance sampling estimator 的 variance 取决于 q 与 p 有多相似。如果 q 和 p 差别很大，少数 samples 会获得巨大 weights 并主导估计。Self-normalized importance sampling 会除以 weights 总和，以缓解问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo Estimation / Monte Carlo 估计

Monte Carlo estimation 通过 random samples 的平均值近似 integrals。Law of large numbers 保证收敛。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

误差率与维度无关。这就是为什么 Monte Carlo methods 在高维中占主导，而 grid-based integration 在高维中不可行。

**估计 pi：**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估计 expectations：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings / MCMC：Metropolis-Hastings

MCMC 会构造一条 stationary distribution 是 target distribution p(x) 的 Markov chain。经过足够多 steps 后，这条 chain 的 samples 就近似来自 p(x)。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对 symmetric proposals（q(x'|x) = q(x|x')），ratio 会简化为 p(x')/p(x)。这就是原始 Metropolis algorithm。

**为什么有效。** Acceptance rule 保证 detailed balance：位于 x 并移动到 x' 的概率，等于位于 x' 并移动到 x 的概率。Detailed balance 意味着 p(x) 是这条 chain 的 stationary distribution。

**实践注意事项：**
- Burn-in：丢弃 chain 达到 equilibrium 前的早期 samples
- Thinning：每 k 个 sample 保留一个，降低 autocorrelation
- Proposal scale：太小，chain 移动很慢（high acceptance、slow exploration）；太大，大多数 proposals 被拒绝（low acceptance、stuck in place）
- 高维 Gaussian proposal 的 optimal acceptance rate 约为 0.234

### Gibbs Sampling / Gibbs 采样

Gibbs sampling 是 multivariate distributions 的一种特殊 MCMC。它不是一次性 proposal 所有 dimensions，而是每次从 conditional distribution 中更新一个变量。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling 要求你能从每个 conditional distribution p(x_i | x_{-i}) 中采样。许多 models 中这很直接：
- Bayesian networks：conditionals 来自 graph structure
- Gaussian mixtures：conditionals 是 Gaussian
- Ising models：每个 spin 的 conditional 只依赖 neighbors

Acceptance rate 始终为 1，因为从精确 conditional 中采样会自动满足 detailed balance。

**限制。** 当 variables 高度相关时，Gibbs sampling mixing 很慢，因为一次只更新一个变量，无法沿 distribution 做大幅 diagonal moves。

### Temperature Sampling (Used in LLMs) / Temperature sampling（LLMs 中使用）

Language models 为词表中每个 token 输出 logits z_1, ..., z_V。Softmax 把它们转成 probabilities。Temperature 会在 softmax 前重新缩放 logits：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**为什么有效。** 用 T < 1 除 logits 会放大 logits 之间的差异。如果 z_1 = 2、z_2 = 1，除以 T = 0.5 后得到 z_1/T = 4、z_2/T = 2，差距变大。经过 softmax 后，最高 logit 的 token 会得到更大概率质量。

**实践中：**
- T = 0.0：greedy decoding，最适合 factual Q&A
- T = 0.3-0.7：略有创造性，适合 code generation
- T = 0.7-1.0：平衡，适合 general conversation
- T = 1.0-1.5：creative writing、brainstorming
- T > 1.5：越来越随机，通常没什么用

Temperature 不会改变哪些 tokens 是可能的。它改变每个 token 分到的 probability mass。

### Top-k Sampling / Top-k 采样

Top-k sampling 把候选集限制为概率最高的 k 个 tokens，然后 renormalize，并从这个限制后的集合中采样。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 会阻止模型选择 long tail 中极不可能的 tokens，例如 typo 或 nonsense。问题是：k 是固定的，不随上下文变化。当模型很自信（某个 token 有 95% 概率）时，k = 40 仍允许 39 个 alternatives。当模型不确定（概率分散在 1000 个 tokens 上）时，k = 40 又会截断一些 plausible options。

### Top-p (Nucleus) Sampling / Top-p（Nucleus）采样

Top-p sampling 会动态调整候选集大小。它不是保留固定数量的 tokens，而是保留 cumulative probability 超过 p 的最小 token 集合。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

当模型自信时，nucleus sampling 保留很少 tokens，也许 2-3 个。当模型不确定时，它会保留很多，也许 200 个。这种自适应行为是 nucleus sampling 通常比 top-k 生成更好文本的原因。

**常见组合：**
- Temperature 0.7 + top-p 0.9：通用设置
- Temperature 0.0（greedy）：最适合 deterministic tasks
- Temperature 1.0 + top-k 50：Fan et al. (2018) 原始论文设置

Top-k 和 top-p 可以组合。先应用 top-k，再在剩余集合上应用 top-p。

### Reparameterization Trick (Used in VAEs) / Reparameterization trick（VAEs 中使用）

Variational autoencoders（VAEs）会把 inputs 编码成 latent space 中的 distribution，从该 distribution 中采样，再解码 sample。问题是：sampling operation 本身无法 backpropagate。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Reparameterization trick 把 randomness 从 parameters 中分离出来：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

它之所以成立，是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 具有同一个 distribution。关键洞见：把 randomness 移到 parameter-free source（epsilon）中，然后把 sample 表达成 parameters 的 differentiable transformation。

**在 VAE training loop 中：**
1. Encoder 为每个 input 输出 mu 和 log(sigma^2)
2. Sample epsilon ~ N(0, 1)
3. Compute z = mu + sigma * epsilon
4. Decode z 来 reconstruct input
5. Backpropagate through steps 4, 3, 2, 1，因为 step 3 是 differentiable

没有 reparameterization trick，VAEs 无法用标准 backpropagation 训练。这个洞见让 VAEs 变得实用。

### Gumbel-Softmax (Differentiable Categorical Sampling) / Gumbel-Softmax（可微 categorical sampling）

Reparameterization trick 适用于 continuous distributions（Gaussian）。对 discrete categorical distributions，需要另一种方法。Gumbel-Softmax 提供了 categorical sampling 的 differentiable approximation。

**Gumbel-Max trick（不可微）：**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微近似）：**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax 会产生离散 sample 的 continuous relaxation。输出是 probability vector（soft one-hot），而不是 hard one-hot。Gradients 会穿过 softmax。训练 forward pass 中，你可以使用 "straight-through" estimator：forward pass 用 hard argmax，backward pass 用 soft Gumbel-Softmax gradients。

**应用：**
- VAEs 中的 discrete latent variables
- Neural architecture search（选择 discrete operations）
- Hard attention mechanisms
- 带 discrete actions 的 reinforcement learning

### Stratified Sampling / 分层采样

标准 Monte Carlo sampling 可能因为随机性在 sample space 中留下空洞。Stratified sampling 会把空间划分成 strata，并从每个 stratum 中采样，强制覆盖均匀。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Stratified sampling 的 variance 永远小于或等于 standard Monte Carlo：

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**应用：**
- Numerical integration（quasi-Monte Carlo）
- Training data splits（确保每个 fold class balance）
- 带 stratification 的 importance sampling（结合两种技术）
- NeRF（Neural Radiance Fields）沿 camera rays 使用 stratified sampling

### Connection to Diffusion Models / 与 Diffusion Models 的连接

Diffusion models 通过 sampling process 生成图像。Forward process 会在 T steps 中向图像加入 Gaussian noise，直到它变成纯噪声。Reverse process 学会 denoise，一步步恢复原图。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

与本课方法的连接：
- 每个 denoising step 都使用 reparameterization trick（sample noise，再应用 deterministic transform）
- Noise schedule {alpha_t} 控制一种 temperature annealing
- 训练使用 Monte Carlo estimation 近似 ELBO（evidence lower bound）
- Diffusion models 中的 ancestral sampling 是 Markov chain，每一步只依赖当前状态

整个 image generation process 都是 iterative sampling：从 noise 开始，每一步都在 learned denoising model 条件下采样一个稍微没那么 noisy 的版本。

```figure
monte-carlo-pi
```

## Build It / 动手构建

### Step 1: Uniform and inverse CDF sampling / 第 1 步：Uniform 与 inverse CDF sampling

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个 exponential samples，并验证 mean 是 1/lambda。

### Step 2: Rejection sampling / 第 2 步：Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用 rejection sampling 从 truncated normal distribution 中抽样。通过 histogram 验证 samples 的形状。

### Step 3: Importance sampling / 第 3 步：Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用 uniform proposal 估计 normal distribution 下的 E[X^2]。与已知答案（mu^2 + sigma^2）对比。

### Step 4: Monte Carlo estimation of pi / 第 4 步：Monte Carlo 估计 pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Step 5: Metropolis-Hastings MCMC / 第 5 步：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从 bimodal distribution（两个 Gaussians 的 mixture）中采样。可视化 chain 的 trajectory。

### Step 6: Gibbs sampling / 第 6 步：Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Step 7: Temperature sampling / 第 7 步：Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示 temperature 如何改变一组 token logits 的 output distribution。

### Step 8: Top-k and top-p sampling / 第 8 步：Top-k 与 top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Step 9: Reparameterization trick / 第 9 步：Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示 gradients 可以穿过 reparameterized sample 流动，但不能穿过 direct sampling。

### Step 10: Gumbel-Softmax / 第 10 步：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低 temperature 如何让输出接近 one-hot vector。

包含所有可视化的完整实现位于 `code/sampling.py`。

## Use It / 应用它

使用 NumPy 和 SciPy 的 production versions：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

大规模 MCMC 使用专门 libraries：
- PyMC：带 NUTS（adaptive HMC）的完整 Bayesian modeling
- emcee：ensemble MCMC sampler
- NumPyro/JAX：GPU-accelerated MCMC

你已经从零构建了这些方法。现在你知道 library calls 背后在做什么。

## Ship It / 交付它

本课交付一套从基础随机数到现代生成式模型的 sampling toolkit：inverse CDF、rejection、importance、Monte Carlo、MCMC，以及 LLM decoding 中的 temperature/top-k/top-p。

## Exercises / 练习

1. 为 Cauchy distribution 实现 inverse CDF sampling。CDF 是 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个 samples，并把 histogram 与真实 PDF 对比。观察 heavy tails，也就是远离中心的 extreme values。

2. 使用 Uniform(0, 1) proposal，通过 rejection sampling 生成 Beta(2, 5) distribution 的 samples。把 accepted samples 与真实 Beta PDF 对比绘图。理论 acceptance rate 是多少？

3. 用 1,000、10,000 和 100,000 个 samples，通过 Monte Carlo 估计 sin(x) 在 0 到 pi 上的积分。比较每个样本量下的 error。验证 error 按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings，从 2D distribution p(x, y) proportional to exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 中采样。绘制 samples 和 chain trajectory。尝试不同 proposal standard deviations。

5. 构建一个完整 text generation demo：给定一个包含 10 个词的 vocabulary 和 logits，用 (a) greedy、(b) temperature=0.7、(c) top-k=3、(d) top-p=0.9 生成长度为 20 的 token sequences。比较 5 次运行中的 output diversity。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Sampling | “抽随机值” | 按 probability distribution 生成值。所有 generative AI 背后的机制 |
| Uniform distribution | “全部等可能” | [a, b] 中每个值都有相同 probability density 1/(b-a)。所有 sampling methods 的起点 |
| Inverse CDF | “概率变换” | F_inverse(U) 把 uniform sample 转成任意已知 CDF 的 distribution sample。精确且高效 |
| Rejection sampling | “提议再接受/拒绝” | 从简单 proposal 生成，根据 target/proposal ratio 成比例接受。精确但浪费 samples |
| Importance sampling | “重加权 samples” | 用 q(x) 的 samples 估计 p(x) 下的 expectations，每个 sample 按 p(x)/q(x) 加权。RL 中 PPO 的核心 |
| Monte Carlo | “平均 random samples” | 用 sample averages 近似 integrals。Error 是 O(1/sqrt(N))，与维度无关 |
| MCMC | “会收敛的 random walk” | 构造 stationary distribution 为 target 的 Markov chain。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | “上坡接受，下坡有时接受” | 提议 moves，根据 density ratio 接受。Detailed balance 保证收敛到 target distribution |
| Gibbs sampling | “一次一个变量” | 固定其他变量，从条件分布中更新每个变量。Acceptance rate 为 100% |
| Temperature | “Confidence knob” | Softmax 前用 T 除 logits。T<1 sharpen（更自信），T>1 flatten（更多样） |
| Top-k sampling | “保留最好的 k 个” | 把除 k 个最高概率 tokens 外的概率置零，renormalize 后采样。候选集大小固定 |
| Nucleus sampling (top-p) | “保留有概率质量的那部分” | 保留 cumulative probability 超过 p 的最小 token 集合。候选集大小自适应 |
| Reparameterization trick | “把随机性移出去” | 写成 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。让 sampling 可微。VAE 训练必需 |
| Gumbel-Softmax | “Soft categorical sampling” | 使用 Gumbel noise + 带 temperature 的 softmax，对 categorical sampling 做可微近似 |
| Stratified sampling | “强制覆盖” | 把 sample space 划分成 strata，并从每个 stratum 中采样。Variance 总是不高于 naive Monte Carlo |
| Burn-in | “Warm-up period” | MCMC chain 到达 stationary distribution 前丢弃的初始 samples |
| Detailed balance | “可逆性条件” | p(x) * T(x->y) = p(y) * T(y->x)。这是 p 成为 Markov chain stationary distribution 的充分条件 |
| Diffusion sampling | “迭代去噪” | 从 noise 开始，通过 learned denoising steps 生成数据。每一步都是 conditional sampling operation |

## Further Reading / 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原始论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus（top-p）sampling 论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 引入 reparameterization trick 的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM，把 sampling 与 image generation 连接起来
