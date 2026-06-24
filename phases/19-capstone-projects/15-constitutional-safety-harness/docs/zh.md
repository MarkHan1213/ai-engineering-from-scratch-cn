# Capstone 15 — Constitutional Safety Harness + Red-Team Range / 宪法式安全 Harness 与 Red-Team 靶场

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety，以及覆盖多语言的 X-Guard，共同定义了 2026 年 safety-classifier stack。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准 adversarial evaluation tools。NeMo Guardrails v0.12 把它们接入 production pipeline。本 capstone 要把所有这些接起来：围绕目标 app 的 layered safety harness、运行 6+ attack families 的 autonomous red-team agent，以及能产出可测 harmlessness delta 的 constitutional self-critique run。

**类型：** 综合项目
**语言：** Python（safety pipeline, red team）, YAML（policy configs）
**前置知识：** 第 10 阶段（LLMs from scratch）, 第 11 阶段（LLM engineering）, 第 13 阶段（tools）, 第 14 阶段（agents）, 第 18 阶段（ethics, safety, alignment）
**Phases exercised:** P10 · P11 · P13 · P14 · P18
**时间：** 25 小时

## Learning Objectives / 学习目标

- 设计 input sanitize、policy rails、classifier gate、model、output filter、HITL 组成的 layered safety pipeline
- 组合 Llama Guard 4、X-Guard、ShieldGemma-2、Nemotron 3 Content Safety 与 NeMo Guardrails
- 构建 red-team range，覆盖 PAIR、TAP、GCG、encoding、multi-turn、multilingual code-switch 等 attack families
- 实现 constitutional self-critique 训练干预，并测量 before/after harmlessness delta
- 交付 CVSS-scored findings、disclosure timeline 和可重复运行的安全评测自动化

## Problem / 问题

2026 年 LLM safety 的前沿问题，不是 classifiers 是否有效（大体有效），而是如何把它们正确组合在 production app 周围：既不过度拒答，也不留下明显漏洞。Llama Guard 4 处理英文 policy violations。X-Guard（132 languages）处理 multilingual jailbreak。ShieldGemma-2 捕获 image-based prompt injection。NVIDIA Nemotron 3 Content Safety 覆盖 enterprise categories。Anthropic 的 Constitutional Classifiers 是训练期使用的另一种方法，而不是 serving-time guard。

攻击也在进化。PAIR 和 TAP 自动发现 jailbreak。GCG 运行 gradient-based suffix attacks。Multi-turn 和 code-switch attacks 利用 Agent memory。任何已部署 LLM 都需要 red-team range，garak 和 PyRIT 是 canonical drivers，并且需要记录 mitigations 和 CVSS-scored findings。

你要加固一个目标应用（8B instruction-tuned model 或其他 capstones 中的 RAG chatbot），用 6+ attack families 攻击它，并产出 before/after harmlessness measurement。

## Concept / 概念

safety pipeline 有五层。**Input sanitize**：去掉 zero-width chars，解码 base64/rot13，normalize Unicode。**Policy layer**：NeMo Guardrails v0.12 rails（off-domain、toxicity、PII extraction）。**Classifier gate**：input 上跑 Llama Guard 4，非英语跑 X-Guard，image inputs 跑 ShieldGemma-2。**Model**：目标 LLM。**Output filter**：output 上跑 Llama Guard 4、Presidio PII scrub，以及需要时的 citation enforcement。**HITL tier**：高风险输出进入 Slack queue。

red-team range 按 scheduler 运行。PAIR 和 TAP 自主发现 jailbreaks。GCG 跑 gradient-based suffix attacks。ASCII / base64 / rot13 encoding attacks。Multi-turn attacks（persona adoption、memory exploitation）。Code-switch attacks（混合英语与 Swahili 或 Thai）。每次运行产出带 CVSS scoring 和 disclosure timeline 的 structured findings file。

constitutional-self-critique run 是训练期干预。取 1k harmful-attempt prompts，让模型先草拟 response，再对照书面 constitution（do-not-harm rules）做 critique，并用 critique loop 重新训练。用 held-out eval 测量 before/after harmlessness delta。

## Architecture / 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## Stack / 技术栈

- Safety classifiers: Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- Guardrail framework: NeMo Guardrails v0.12 + OPA
- Red-team drivers: garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- Jailbreak agents: PAIR（Chao et al., 2023）、Tree-of-Attacks (TAP)、GCG suffix
- Constitutional training: Anthropic-style self-critique loop + SFT on critiques
- PII scrub: Presidio
- Target: 8B instruction-tuned model 或其他 capstones 的 RAG chatbots

## Build It / 动手构建

1. **Target setup.** 在 vLLM 上启动一个 8B instruction-tuned model（或复用另一个 capstone 的 RAG chatbot）。这是被测 app。

2. **Safety pipeline wrap.** 围绕目标接入五层 pipeline。验证每一层都可单独观测（Langfuse 中每层一个 span）。

3. **Classifier coverage.** 加载 Llama Guard 4、X-Guard（multilingual）、ShieldGemma-2（image）。在小型 labeled set 上运行，建立 baseline。

4. **Red-team scheduler.** 调度 garak、PyRIT、PAIR agent、TAP agent、GCG runner、multi-turn attacker 和 code-switch attacker。每类攻击跑在独立 queue 上。

5. **Attack suite.** 六类 attack families：(1) PAIR automated jailbreak，(2) TAP tree-of-attacks，(3) GCG gradient suffix，(4) ASCII / base64 / rot13 encoding，(5) multi-turn persona，(6) multilingual code-switch。按 family 报告 success rate。

6. **Constitutional self-critique.** 整理 1k harmful-attempt prompts。每个 prompt 先让 target 草拟 response。critic LLM 对照书面 constitution（“do no harm,” “cite evidence,” “refuse illegal requests”）打分。critic 反对的 prompts 会被重写；target 用 critique-improved pairs 进行 fine-tune。用 held-out eval 测量 before/after harmlessness。

7. **Over-refusal measurement.** 在 benign prompt suite（例如 XSTest）上追踪 false-positive rate。目标在 benign questions 上保持 helpful。

8. **CVSS scoring.** 对每个 successful jailbreak 按 CVSS 4.0 评分（attack vector、complexity、impact）。生成 disclosure timeline 和 mitigation plan。

9. **Range automation.** 上述所有内容都通过 cron 运行；findings 写入 queue；over-refusal regression alerts 发到 Slack。

## Use It / 应用它

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## Ship It / 交付它

`outputs/skill-safety-harness.md` 是交付物。它是 production-grade layered safety pipeline，加上可复现 red-team range 和 before/after harmlessness deltas。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | Attack-surface coverage | 6+ attack families、2+ languages 被演练 |
| 20 | True-positive / false-positive trade-off | Attack block rate vs XSTest benign pass rate |
| 20 | Self-critique delta | held-out eval 上的 before/after harmlessness |
| 20 | Documentation and disclosure | 带 timeline 的 CVSS-scored findings |
| 15 | Automation and repeatability | 所有内容通过 cron 运行并带 alerts |
| **100** | | |

## Exercises / 练习

1. 在 RAG chatbot 上运行 garak 的 prompt-injection plugin，比较有无 output-filter layer 时的 attack success rate。

2. 添加第七类 attack family：通过 retrieved documents 进行 indirect prompt injection。测量所需额外防护。

3. 实现 “refuse-with-help” mode：guardrail 阻断时，target 提供一个更安全的相关回答，而不是直接拒绝。测量 XSTest delta。

4. Multilingual coverage gap：找一种 X-Guard 表现不佳的语言。提出针对它的 fine-tune dataset。

5. 在 30B model 上运行 constitutional self-critique，测量 delta 是否随模型规模扩大。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Layered safety | “Defense in depth” | input、gate、output、HITL 上的多层 guardrails |
| Llama Guard 4 | “Meta's safety classifier” | 2026 reference input/output content classifier |
| PAIR | “Jailbreak agent” | Chao et al. 关于 LLM-driven jailbreak discovery 的论文方法 |
| TAP | “Tree-of-Attacks” | PAIR 的 tree-search variant |
| GCG | “Greedy coordinate gradient” | gradient-based adversarial suffix attack |
| Constitutional self-critique | “Anthropic-style training” | Target drafts -> critic scores -> rewrite -> retrain |
| XSTest | “Benign probe set” | 用于 over-refusal regression 的 benchmark |
| CVSS 4.0 | “Severity score” | safety findings 的标准 vulnerability scoring |

## Further Reading / 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — training-time reference
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 input/output classifier
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — image + multimodal safety
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — enterprise reference
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132-language multilingual safety
- [garak](https://github.com/NVIDIA/garak) — NVIDIA red-team toolkit
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft red-team framework
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail framework
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — jailbreak agent paper
