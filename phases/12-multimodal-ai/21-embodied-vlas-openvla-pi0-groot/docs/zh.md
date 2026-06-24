# Embodied VLAs: RT-2, OpenVLA, π0, GR00T / 具身 VLA：RT-2、OpenVLA、π0、GR00T

> 第一次有模型从网页读取菜谱并让厨房机器人执行，是 RT-2（Google DeepMind, 2023 年 7 月）。RT-2 把 actions 离散成 text tokens，在 web data 与 robot-action data 上共同 fine-tune 一个 VLM，并证明 web-scale vision-language knowledge 能迁移到 robotic control。OpenVLA（2024 年 6 月）发布了 open 7B reference。Physical Intelligence 的 π0 series（2024-2025）加入 flow-matching action experts。NVIDIA 的 GR00T N1（2025 年 3 月）为 humanoid robots 提供了大规模 dual-system（System 1 / System 2）控制。VLA primitive（vision-language-action，一个能看、能读、能行动的单模型）是本 phase 的 understanding models 与 Phase 15 autonomous systems 之间的桥。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python (stdlib, action tokenizer + VLA inference skeleton)
**Prerequisites / 前置知识：** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, referenced)
**Time / 时间：** 约 180 分钟

## Learning Objectives / 学习目标

- 描述 action tokenization：discrete bin encoding（RT-2）、FAST efficient action tokens、continuous flow-matching actions（π0）。
- 解释为什么在 web + robot data 上 co-fine-tuning 可以保留对新任务的 general-knowledge transfer。
- 在同一 robot task 上比较 OpenVLA（open 7B Llama+VLM）、π0（flow-matching）和 GR00T N1（dual-system）。
- 说出 Open X-Embodiment dataset，以及它作为 RT-X training corpus 的作用。

## The Problem / 问题

让机器人根据自然语言指令做家务，是从 1970 年代就存在的研究目标。2020 年代的答案是 vision-language-action（VLA）模型。它使用类似 VQA 的 VLM architecture，但输出是动作（joint torques、end-effector poses、discrete commands），不是文本。

VLA 的特殊挑战：

1. Action spaces 连续且高维（7-DOF arm + 3-DOF gripper = 10 dims @ 30 Hz）。
2. Robot-specific training data 稀缺。Open X-Embodiment 约 1M trajectories；web text-image 是 5B+。
3. Control frequency 很关键。30 Hz control loop 意味着每个 action 只有 33ms budget。
4. Safety。错误动作会损坏硬件、伤害人或财物。

## The Concept / 概念

### Action tokenization (RT-2) / Action tokenization（RT-2）

RT-2 的技巧是：把每个 joint target 表示成 quantized text token。把归一化的 [-1, 1] 范围离散成 256 个 bins，每个 bin 映射到一个 vocabulary ID。10-DOF action 在每个 control step 变成 10 个 tokens。

在混合数据上 co-fine-tune PaLM-X VLM：

- Web image-text pairs（captioning、VQA）。
- Robot demonstrations，action 作为 tokens。

模型看到 “pick up the red cube”（language）→ image（vision）→ 10-token action sequence（discretized joint targets）。Web pretraining 保留 general-knowledge transfer：RT-2 能遵循 “move towards the fast-moving object”，即使 “fast-moving” 不在 robot training data 中。

RT-2 论文中的 inference 是 3-5 Hz，受限于 VLM autoregressive decode。

### OpenVLA — the open 7B reference / OpenVLA：open 7B reference

OpenVLA（Kim et al., 2024 年 6 月）是 open-weights RT-2 equivalent。7B Llama backbone，DINOv2 + SigLIP dual vision encoder，基于 256 bins 的 action tokenization。

训练数据是 Open X-Embodiment（970k trajectories，覆盖 22 robots）。提供 LoRA fine-tuning 支持，用于适配新机器人。

Inference：A100 上量化后 4-5 Hz。足以做慢速 manipulation，但不够高频控制。

### FAST tokenizer — faster action decode / FAST tokenizer：更快 action decode

Pertsch et al.（2024）指出 discrete-bin tokenization 低效：多数 action 聚集在 bin-space 的小区域。FAST（Frequency-domain Action Sequence Tokenizer）通过 DCT 压缩 action sequences，并量化 coefficients。

30-step action trajectory 变成约 10 个 FAST tokens，而不是 300 个 discrete-bin tokens。推理加速 3-5x，且不损失质量。

### π0 and flow-matching actions / π0 与 flow-matching actions

Physical Intelligence 的 π0（Black et al., 2024 年 10 月）用 flow-matching action expert 替换 discrete action tokens：

- 一个小 action transformer 读取 VLM hidden states，并通过 rectified flow 输出连续的 50-step action sequence。
- Action head 用 flow-matching loss 训练；VLM pretraining 保持不变。
- Inference：约 5 个 denoising steps 输出完整 action sequence，实质上可做 50 Hz control。

π0 声称在广泛 manipulation tasks 上超过 OpenVLA 和 Octo。连续 action formulation 保留了 discretization 会破坏的 smoothness。

π0.5 和 π0-FAST 是 incremental upgrades。π0-FAST 结合 FAST tokenization 与 flow matching。

### GR00T N1 — dual-system for humanoids / GR00T N1：面向 humanoid 的 dual-system

NVIDIA 的 GR00T N1（2025 年 3 月）面向 humanoid robots（>30 DOF，全身）：

- System 2：大型 VLM，读取 scene + instruction，以约 1 Hz 产生 high-level subgoals。
- System 1：小 action-head transformer，根据 subgoals 产生 50-100 Hz low-level joint commands。

这个拆分对应 Kahneman 的快慢系统：System 2 规划，System 1 执行。好处是 VLM 级别的慢规划不会阻塞快速控制；System 1 保持小模型以满足 latency。

GR00T N1.7（2025 年末）改善 data scaling。GR00T 通过 Omniverse 的 sim-to-real data 微调。

### Open X-Embodiment / Open X-Embodiment

训练数据。RT-X（2023 年 10 月）汇集了 22 个数据集，覆盖 22 个机器人上的 1M trajectories。Open X-Embodiment 是大家都用的 corpus：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每个样本包含 `(robot state, camera views, instruction, action sequence)`。
- 训练卫生：统一 action space，normalize joint ranges，resize cameras。

OpenVLA 和 π0 都在 Open X-Embodiment 上训练。到具体机器人的 domain gap 通常通过 100-1000 条 task-specific demos 上的 LoRA fine-tuning 关闭。

### Co-fine-tuning vs robot-only / Co-fine-tuning 与 robot-only

Co-fine-tuning 混合 web VQA data 与 robot trajectories。比例很关键：VQA 太多会忘动作；robot data 太多会丢 general knowledge。

RT-2 ratio 约 1:1。OpenVLA 约 0.5:1 web-to-robot。π0 类似。精确比例是每个 dataset size 都要调的 hyperparameter。

Robot-only training 会产生 task-specific models，在 out-of-distribution instructions 上失败。Co-fine-tuning 的差别就是：从“pick up the red cube（demo 里有）”到“pick up the third largest object from the left（新说法）”。

### Safety and action limits / 安全与动作限制

每个 production VLA 都应带：

- Hard joint limits（不能超过机械规格）。
- Velocity limits（soft clipping）。
- Workspace bounds（end-effector 不能离开桌面）。
- Human-in-the-loop approval for novel tasks。

这些位于 VLA 之外的 control-layer checks。VLA 输出的是建议，不是直接命令。

## Build It / 动手构建

本课构建 action tokenizer 与 VLA inference skeleton：先实现 256-bin action tokenization/de-tokenization，再比较 discrete-bin、FAST 和 continuous-flow 的 token 成本，最后把视觉输入、指令和 action output 串成最小 VLA loop。

## Use It / 应用它

`code/main.py`：

- 实现 256-bin action tokenization 和 de-tokenization。
- 基于 DCT + quantization 勾勒 FAST tokenizer。
- 比较 discrete-bin、FAST、continuous-flow 三种 action step 的 token-count。
- 打印 RT-2 → OpenVLA → π0 → GR00T 的 lineage summary。

## Ship It / 交付它

本课产出 `outputs/skill-vla-action-format-picker.md`。给定 robot task（manipulation、navigation、humanoid whole-body），它会在 discrete-bin + RT-2、FAST + OpenVLA、flow-matching + π0 或 dual-system + GR00T 之间选择。

## Exercises / 练习

1. 一个 10-DOF arm，30 Hz control rate。256 bins 的 discrete-bin tokenization 每秒输出多少 tokens？7B VLM 能跟上吗？

2. FAST tokenization 把 30-step trajectories 压缩到约 10 tokens。如果轨迹有 high-frequency motion（例如 drumming），用户会损失什么？

3. π0 的 flow-matching head 约 5 步 denoise。与 OpenVLA 4-5 Hz 的 autoregressive decode 比较 throughput。

4. GR00T 的 System 1 / System 2 拆分映射到 Kahneman。提出一个不同拆分（System 3?），可能帮助 bipedal walking。

5. 阅读 Open X-Embodiment Section 4 关于 dataset curation。说出防止 domain leakage 的三条 curation rules。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| VLA | “Vision-language-action” | 接收 image + instruction 并输出 action commands 的模型 |
| Action tokenization | “Discrete bins” | 把连续 joint targets 量化成每维 256 bins，每个 bin 是 vocab ID |
| FAST tokenizer | “Frequency action tokens” | 用 DCT + quantize 把 30-step trajectories 压缩到约 10 tokens |
| Co-fine-tune | “Mix web + robot” | 同时在 web VQA data 与 robot demos 上训练，以保留 general knowledge |
| Flow-matching action head | “π0 continuous output” | 小 transformer 通过 rectified flow 输出 50-step action sequence |
| System 1 / System 2 | “Dual-system control” | 大 VLM 慢速规划，小 action head 快速行动；GR00T pattern |
| Open X-Embodiment | “RT-X dataset” | 跨机器人 1M-trajectory dataset；核心训练 corpus |

## Further Reading / 延伸阅读

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
