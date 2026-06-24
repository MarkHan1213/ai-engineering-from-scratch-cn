# Capstone 07 — End-to-End Fine-Tuning Pipeline (Data to SFT to DPO to Serve) / 端到端微调流水线（数据到 SFT 到 DPO 到服务）

> 一个用你自己的数据训练的 8B 模型，用你自己的偏好做 DPO 对齐，量化、speculative-decoded，并以可测的 $/1M tokens 对外服务。2026 年的 open stack 是 Axolotl v0.8、TRL 0.15、Unsloth 用于快速迭代、GPTQ/AWQ/GGUF 用于量化、vLLM 0.7 加 EAGLE-3 用于服务。本 capstone 要你可复现地跑完整 pipeline：YAML 输入，served endpoint 输出，并按 2026 Model Openness Framework 发布 model card。

**类型：** 综合项目
**语言：** Python（pipeline）, YAML（configs）, Bash（scripts）
**前置知识：** 第 02 阶段（ML）, 第 03 阶段（DL）, 第 07 阶段（transformers）, 第 10 阶段（LLMs from scratch）, 第 11 阶段（LLM engineering）, 第 17 阶段（infrastructure）, 第 18 阶段（safety）
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间：** 35 小时

## Learning Objectives / 学习目标

- 构建从 data hygiene 到 SFT、DPO / GRPO、quantization、serving、eval 的端到端流水线
- 编写可复现的 Axolotl 和 TRL 配置，并记录 seeds、YAMLs 和 commit SHAs
- 比较 base、SFT-only、SFT+DPO、SFT+GRPO 在多个 benchmark 上的增益
- 测量 speculative decoding、quantization 和 batch size 对 tokens/s、p99、$/1M tokens 的影响
- 交付带 safety eval 和 2026 MOF model card 的可服务模型

## Problem / 问题

到 2026 年，每个严肃 AI 团队都会保留一条 fine-tuning pipeline。不是因为他们要发布 frontier base model，而是因为 downstream adaptation 才是可测收益所在：domain SFT、基于标注偏好的 DPO、用于 speculative decoding 的 distilled drafts、以及 EAGLE-3 serving。Axolotl v0.8 处理 multi-GPU SFT configs。TRL 0.15 处理 DPO 和 GRPO。Unsloth 让单卡迭代更快。vLLM 0.7 加 EAGLE-3 能在不损失质量的情况下把 decode throughput 推高 2-3x。工具已经可用；工艺在 YAML、data hygiene 和 eval discipline。

你要把一个 8B base（Llama 3.3、Qwen3 或 Gemma 3）用 task-specific data 跑过 SFT 再 DPO，量化后用于 serving，并用 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 测量收益。你还要按 2026 Model Openness Framework 产出 model card。重点是可复现：一条命令重跑完整 pipeline。

## Concept / 概念

pipeline 有五个阶段。**Data**：dedup（MinHash / Datatrove）、quality filter（Nemotron-CC style classifier）、PII scrub、针对公开 benchmark contamination 的 split-hygiene check。**SFT**：Axolotl YAML、8xH100 上 ZeRO-3、cosine schedule、packed sequences、2-3 epochs。**DPO or GRPO**：TRL config、1 epoch，preference pairs 来自人工标注或 model-judged，调 beta。**Quantize**：GPTQ + AWQ + GGUF，方便多种部署。**Serve**：vLLM 0.7 with EAGLE-3 speculative heads（或 SGLang with SpecForge）、K8s deployment、基于 queue-wait 的 HPA。

ablation 是交付物：在三个 task-specific benchmarks 上比较 SFT-only、SFT+DPO、SFT+GRPO。Serving metrics 包括 batch 1 / 8 / 32 下 tokens/s、EAGLE-3 acceptance rate、$/1M tokens。Safety eval 用 Llama Guard 4 pass rate。Model card 需要覆盖 bias evaluations、reproducibility seeds、data licensing。

## Architecture / 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## Stack / 技术栈

- Data: Datatrove 做 dedup，Nemotron-CC classifier 做 quality，Presidio 做 PII
- Base: Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT: Axolotl v0.8 with ZeRO-3、Flash Attention 3、packed sequences
- Preference tuning: TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于 single-GPU iteration
- Quantization: GPTQ (Marlin)、AWQ、通过 llama.cpp 生成 GGUF
- Serving: vLLM 0.7 with EAGLE-3 speculative decoding（或 SGLang 0.4 + SpecForge）
- Eval: lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- Safety eval: Llama Guard 4、ShieldGemma-2
- Infrastructure: Kubernetes + NVIDIA device plugin，HPA 使用 queue-wait metric
- Observability: 训练用 W&B，推理用 Langfuse

## Build It / 动手构建

1. **Data pipeline.** 对 raw corpus 运行 Datatrove dedup。应用 Nemotron-CC-style quality classifier。Presidio 清理 PII。用明确 seed 写出 train/val splits。

2. **Contamination check.** 对每个 validation split，用 MinHash 与 MMLU-Pro、MT-Bench-v2、RewardBench-2 test sets 对比。拒绝任何 overlap。

3. **Axolotl SFT.** YAML 配置 ZeRO-3、FA3、sequence packing。在 8xH100 上跑 2-3 epochs。记录到 W&B。

4. **TRL DPO / GRPO.** 取 SFT checkpoint，在 preference pairs 上跑一轮 DPO（或对 math/code 使用 verifiable reward 的 GRPO）。扫 beta。

5. **Quantize.** 产出三种 quants：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M for llama.cpp。记录大小和标称 throughput。

6. **Serve with speculative decoding.** vLLM 0.7 config 加 EAGLE-3 draft heads，draft heads 由 Red Hat Speculators 训练。测量 batch 1 / 8 / 32 下的 acceptance rate 和 tail latency。报告同一 eval 上相对 Anthropic / OpenAI 的 $/1M tokens。

7. **Eval matrix.** 在 base、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。生成表格。

8. **Safety eval.** 在 dev set 上测 Llama Guard 4 pass rate。使用 ShieldGemma-2 output filter。

9. **Model card.** 使用 MOF 2026 template：data、training、eval、safety、license，以及包含 YAMLs 和 commit SHAs 的 reproducibility section。

## Use It / 应用它

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## Ship It / 交付它

`outputs/skill-finetuning-pipeline.md` 描述交付物。一条命令让数据依次经过 SFT、DPO、quant、serve 和 eval，并输出 model card + served endpoint。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Eval delta vs base | 目标任务上的 measured gain（MMLU-Pro、MT-Bench-v2、task-specific） |
| 20 | Pipeline reproducibility | 一条命令用相同 seeds 端到端重跑 |
| 20 | Data hygiene | Dedup rate、PII scrub coverage、contamination check green |
| 20 | Serving efficiency | bs=1/8/32 下 tokens/s、EAGLE-3 acceptance rate、$/1M tokens |
| 15 | Model card + safety eval | 2026 MOF completeness + Llama Guard 4 pass rate |
| **100** | | |

## Exercises / 练习

1. 在同一个 task-specific benchmark 上运行 SFT-only、SFT+DPO、SFT+GRPO。报告哪种 preference method 胜出以及幅度。

2. 把 Llama 3.3 8B 换成 Qwen3 14B。在匹配质量下测量 $/1M tokens。

3. 测量 EAGLE-3 acceptance rate 在 domain data 与 generic ShareGPT 上的差异。报告 delta 以及它对 latency budget 的含义。

4. 注入 1% contamination（把 MMLU-Pro answers 泄漏进 training data）并重跑 eval。观察 MMLU-Pro accuracy 不真实地跳升。构建能捕获它的 contamination-check CI gate。

5. 添加 LoRA SFT 作为 full fine-tune 的替代。测量 10x lower memory 下的 quality gap。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Axolotl | “SFT trainer” | YAML-driven trainer，统一支持 SFT、DPO 和 distillation |
| TRL | “Preference tuner” | Hugging Face library，支持 DPO、GRPO、PPO on LLMs |
| GRPO | “Group-relative policy optimization” | DeepSeek R1 的 RL recipe，使用 verifiable rewards |
| EAGLE-3 | “Speculative decoding draft” | 预测未来 N tokens 的 draft heads；vLLM 用 target model 验证 |
| MOF | “Model Openness Framework” | 2026 年用于按 data、code、license 给 model releases 分级的标准 |
| Contamination check | “Split hygiene” | 基于 MinHash 检测 test-set leakage into training |
| Acceptance rate | “EAGLE / MTP metric” | target model 接受 drafted tokens 的比例 |

## Further Reading / 延伸阅读

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — reference SFT / DPO trainer
- [TRL documentation](https://huggingface.co/docs/trl) — DPO and GRPO reference implementations
- [Unsloth](https://github.com/unslothai/unsloth) — single-GPU iteration reference
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO methodology
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — reference serving stack
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — alternate speculative-decoding trainer
- [Model Openness Framework 2026](https://isocpp.org/) — open-release grading standard
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — canonical eval runner
