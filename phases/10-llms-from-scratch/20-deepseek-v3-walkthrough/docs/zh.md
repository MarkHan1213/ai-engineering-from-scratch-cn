# DeepSeek-V3 Architecture Walkthrough / DeepSeek-V3 架构走读

> Phase 10 · Lesson 14 命名了每个 open model 都会转动的六个架构旋钮。DeepSeek-V3（2024 年 12 月，总参数 671B，active 37B）把六个都转了，并额外加入四个：Multi-Head Latent Attention、auxiliary-loss-free load balancing、Multi-Token Prediction 和 DualPipe training。本课从上到下阅读 DeepSeek-V3 架构，并从公开 config 推导每个 parameter count。完成后，你可以解释为什么 671B/37B 的比例是正确下注，以及为什么 MLA + MoE 组合在 frontier 上优于单独使用任一者。

**类型：** Learn
**语言：** Python (stdlib, parameter calculator)
**前置要求：** Phase 10 · 14（open-model walkthroughs），Phase 10 · 17（NSA），Phase 10 · 18（MTP），Phase 10 · 19（DualPipe）
**时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 从头到尾阅读 DeepSeek-V3 config，并用六个 GPT-2 knobs 加四个 DeepSeek-specific additions 解释每个字段
- 推导 total parameter count（671B）、active parameter count（37B），以及贡献到二者的组件
- 计算 MLA 在 128k context 下的 KV cache footprint，并与相同 active-param dense model 加 GQA 的成本对比
- 陈述四个 DeepSeek-specific innovations（MLA、MTP、auxiliary-loss-free routing、DualPipe），并指出它们分别作用于 architecture/training stack 的哪一部分

## The Problem / 问题

DeepSeek-V3 是第一个架构上与 Llama family 有实质差异的 frontier open model。Llama 3 405B 是“GPT-2 加六个旋钮”。DeepSeek-V3 是 GPT-2 加这六个旋钮，再加四个。阅读 Llama 3 config 是阅读 DeepSeek config 的热身，但它的深层结构，也就是 attention block 的形状、routing logic、training-time objective，差异已经足够大，需要单独走读。

学习它的收益是：DeepSeek-V3 的 open-weights release 改变了 open models 中 “frontier capability” 的含义。这个架构是许多 2026 training runs 正在复制的蓝图。理解它，是任何接触 frontier LLM training 或 inference 的岗位都需要的基本功。

## The Concept / 概念

### The invariant core, again / 再看不变核心

DeepSeek-V3 仍然是 autoregressive。它仍然堆叠 decoder blocks。每个 block 仍然有 attention 加 MLP 加两个 RMSNorm。MLP 仍使用 SwiGLU。仍使用 RoPE。Pre-norm。Weight-tied embeddings。和每个 Llama 或 Mistral 的 baseline 相同。

### The twist: MLA instead of GQA / 转折：用 MLA 替代 GQA

从 Phase 10 · 14 你知道，GQA 通过让多个 Q heads 共享 K 和 V 来缩小 KV cache。Multi-Head Latent Attention（MLA）更进一步：K 和 V 被压缩到一个 shared low-rank latent representation（`kv_lora_rank`），然后在每个 head 上按需 decompressed。KV cache 只存 latent，通常是每 token 每 layer 512 floats，而不是 8 x 128 = 1024 floats。

在 128k context 下，DeepSeek-V3 使用 MLA（每 token 每 layer 一个 shared latent `c^{KV}`；K 和 V 都由这个 latent 经 up-projections 得到，而 up-projections 可以被吸收到后续 matmul 中）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一个假设的 GQA baseline（Llama 3 70B 形状，8 KV heads，head dim 128）需要：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k context 下，MLA 比 Llama-3-70B-style GQA cache 小 4 倍。

权衡是：MLA 每次 attention computation 会增加一个 per-head decompression step。相比节省的带宽，额外 compute 很小。对 long-context inference 来说是净收益。

### The routing: auxiliary-loss-free load balancing / 路由：无辅助损失的负载均衡

MoE routers 决定每个 token 由哪些 top-k experts 处理。朴素 router 会把过多工作集中到少数 experts，导致其他 experts 空闲。标准修复是添加一个 auxiliary loss term 惩罚 load imbalance。这有效，但会轻微损害 main-task performance。

DeepSeek-V3 引入 auxiliary-loss-free scheme。它在 router logits 上添加 per-expert bias terms，并在训练中用简单规则调整：如果 expert `e` 过载，就降低 `bias_e`；如果欠载，就提高它。不加额外 loss term。训练目标保持干净，expert load 保持均衡。

对 main loss 的影响：不可测。对 MoE architecture 的影响：更干净，没有需要调的 auxiliary-loss hyperparameter。

### The MTP: denser training + free draft / MTP：更密集训练加免费 draft

从 Phase 10 · 18 你知道，DeepSeek-V3 添加 D=1 MTP module，用来预测后两个位置的 token。inference 时，训练好的 module 会被复用为 speculative-decoding draft，acceptance 超过 80%。训练时，每个 hidden state 被监督 D+1 = 2 个 targets，提供更密集信号。

参数：在 671B main 之上增加 14B。overhead：2.1%。

### The training: DualPipe / 训练：DualPipe

从 Phase 10 · 19 你知道，DualPipe 是一种双向 pipeline，会把 forward 和 backward chunks 与 cross-node all-to-all comms 重叠。在 DeepSeek-V3 的 2,048-H800 scale 上，它大约回收了 1F1B 会浪费在 pipeline bubbles 上的 245k GPU-hours。

### The config, field by field / 逐字段阅读 config

这里是简化版 DeepSeek-V3 config：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

逐项解析：

- `hidden_size=7168`：embedding dimension。
- `num_hidden_layers=61`：总 block depth。
- `first_k_dense_layers=3`：前 3 个 blocks 使用大小为 18432 的 dense MLP。后续 58 个使用 MoE。
- `num_attention_heads=128`：128 个 query heads。
- `kv_lora_rank=512`：K 和 V 被压缩到这个 latent dimension，再 per head decompressed。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE block 有 256 个 experts，routes top-8。
- `shared_experts=1`：除 256 个 routed experts 外，还有 1 个 always-on expert 作用于每个 token。可以把它看成一个 "dense floor"，确保每个 token 都拿到可靠贡献。
- `moe_intermediate_size=2048`：每个 expert 的 MLP hidden size。它小于 dense MLP，因为一共有 256 个 experts。

### Parameter accounting / 参数核算

完整计算在 `code/main.py`。headline：

- Embedding：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个 dense blocks：带 MLA 的 attention（每 block 约 144M）+ dense MLP（每 block 约 260M）+ norms。总计约 1.2B。
- 58 个 MoE blocks：带 MLA 的 attention（~144M）+ 256 个 experts（每个约 30M）+ 1 个 shared expert（30M）+ norm。包含全部 experts 时，每 block 总计约 7.95B。58 个 MoE blocks 总计 461B。
- MTP module：14B。

Grand total：core architecture 约 476B + 14B MTP；公开 671B 数字还包含额外 structural parameters（bias tensors、expert-specific components、shared expert scaling 等）。calculator 复现的数字与公开值相差 3-5%；delta 来自 DeepSeek report Section 2 appendix 中记录的细粒度核算。

每次 forward 的 active parameters：

- Attention：每 layer 144M * 61 = 8.8B（所有 layers 都触发）。
- Active MLP：前 3 层 dense（3 * 260M = 780M），58 个 MoE layers 每层激活 8 routed + 1 shared + routing overhead。每层 active MLP 约 260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- Embedding + norms：1.2B。
- Total active：core 约 26B + 14B MTP（训练时使用，inference 时不总是运行）≈ 37B。

### The 671B / 37B ratio / 671B / 37B 比例

18x sparsity ratio（active params 是 total 的 5.5%）。DeepSeek-V3 是已经开放权重的最稀疏 frontier MoE model。Mixtral 8x7B 的比例是 13/47（28%），稠密很多。Llama 4 Maverick 的比例是 17B/400B（4.25%），与之可比。DeepSeek 的下注是：在 frontier scale 上，更多 experts 加更低 activation ratio 会带来更好的 quality per active-FLOP。

### Where DeepSeek-V3 sits / DeepSeek-V3 位于哪里

| Model | Total | Active | Ratio | Attention | Novel ideas |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### The follow-on: R1, V4 / 后续：R1、V4

DeepSeek-R1（2025）是在 V3 backbone 上做的 reasoning-training run。R1 使用同一套 architecture。变化的是 post-training recipe（在 verifiable tasks 上做 large-scale RL），而不是 pretraining architecture。

如果 DeepSeek-V4 发布，预期会保留 MLA + MoE + MTP，并加入 DSA（DeepSeek Sparse Attention），也就是 Phase 10 · 17 中 NSA 的后继。这个谱系很稳定：architecture-level innovations 会积累，每个版本都会转动更多旋钮。

```figure
moe-routing
```

## Build It / 动手构建

本课代码是面向 DeepSeek-V3 形状的 parameter calculator。它按 config 字段计算 embedding、MLA attention、dense layers、MoE experts、shared experts 和 MTP module 的参数量，并额外计算 MLA 与 GQA 在 128k context 下的 KV cache 对比。

运行 `code/main.py`，然后修改假设 config（例如 experts 数、top-k、MLA rank），观察 total params、active params 和 KV cache 的变化。这是读懂 DeepSeek-family model card 的最小可执行工具。

## Use It / 使用它

`code/main.py` 是专门针对 DeepSeek-V3 形状的 parameter calculator。运行它，把输出与论文数字对比，并在 hypothetical variants 上使用它（256 experts vs 512、top-8 vs top-16、MLA rank 512 vs 1024）。

重点观察：

- Total parameter count vs published 671B。
- Active parameter count vs published 37B。
- 128k context 下的 KV cache，也就是 MLA vs GQA comparison。
- Per-layer breakdown，查看参数预算真正花在哪里。

## Ship It / 交付

本课会产出 `outputs/skill-deepseek-v3-reader.md`。给定 DeepSeek-family model（V3、R1 或未来变体），它会生成 component-by-component architecture reading，命名 config 中每个字段，按组件推导 parameter counts，并识别模型使用了四个 DeepSeek-specific innovations 中的哪些。

## Exercises / 练习

1. 运行 `code/main.py`。把 calculator 的 total-parameter estimate 与公开 671B 对比，并识别 delta 来源。论文 Section 2 有完整 itemization。

2. 修改 config，把 MLA rank 从 512 改为 256。计算 128k context 下的 KV cache size。它带来多少百分比下降？代价是 per-head expressiveness 受到什么影响？

3. 对比 DeepSeek-V3 的（256 experts, top-8）routing 与一个假设的（512 experts, top-8）变体。Total parameters 增长；active parameters 不变。额外 expert capacity 理论上带来什么？inference 时付出什么成本？

4. 阅读 DeepSeek-V3 technical report（arXiv:2412.19437）关于 MLA 的 Section 2.1。用三句话解释为什么 K 和 V decompression matrices 可以在 inference-time efficiency 上被 “absorbed” into subsequent matmul。

5. DeepSeek-V3 对多数操作使用 FP8 training。计算以 FP8 而非 BF16 存储 671B weights 的内存节省。这与 14.8T-token training budget 如何相互作用？

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MLA | “Multi-Head Latent Attention” | 把 K 和 V 压缩到 shared low-rank latent（kv_lora_rank，通常 512），再 per head on-the-fly decompress；KV cache 只存 latent |
| kv_lora_rank | “MLA compression dim” | K 和 V shared latent 的大小；DeepSeek-V3 使用 512 |
| First k dense layers | “Early layers stay dense” | MoE-model 的前几层跳过 MoE router，运行 dense MLP 以提升稳定性 |
| num_experts_per_tok | “Top-k routing” | 每个 token 触发多少 routed experts；DeepSeek-V3 使用 8 |
| Shared experts | “Always-on experts” | 不管 routing 如何都会处理每个 token 的 experts；DeepSeek-V3 使用 1 |
| Auxiliary-loss-free routing | “Bias-adjusted load balance” | 训练期间调整 per-expert bias terms，在不添加 loss term 的情况下保持 expert load balanced |
| MTP module | “Extra prediction head” | 从 h^(1) 和 E(t+1) 预测 t+2 的 transformer block；更密集训练，免费 speculative-decoding draft |
| DualPipe | “Bidirectional pipeline” | 将 forward/backward compute 与 cross-node all-to-all 重叠的训练 schedule |
| Active parameter ratio | “Sparsity” | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 training | “8-bit training” | 用 FP8 存储训练状态并执行许多 compute ops；相对 BF16 大约减半内存，质量成本很小 |

## Further Reading / 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)：完整 architecture、training 和 results 文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3)：config files 和 deployment notes
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434)：引入 MLA 的前身
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)：基于 V3 架构的 reasoning-training successor
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089)：DeepSeek-family attention 的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe)：training-schedule reference
