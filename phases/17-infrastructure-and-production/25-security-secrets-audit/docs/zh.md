# Security — Secrets, API Key Rotation, Audit Logs, Guardrails / 安全：Secrets、API Key Rotation、Audit Logs、Guardrails

> 用 centralized vaults（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除 secret sprawl。绝不要把 credentials 存在 config files、VCS 里的 env files、spreadsheets。用 IAM roles 代替 static keys；CI/CD 使用 OIDC。AI-gateway pattern 是 2026 年解法：apps → gateway → model provider，gateway 在 runtime 从 vault 拉 credentials。只要在 vault 中 rotate，所有 apps 几分钟内拿到新 key；无需 redeploy，也无需 Slack 里问“谁有新 key”。Rotation policy ≤90 days；每次 commit 都用 TruffleHog / GitGuardian / Gitleaks 扫描。Zero-trust：MFA、SSO、RBAC/ABAC、short-lived tokens、device posture。PII scrubbing 使用 entity recognition 在转发前 mask PHI/PII；consistent tokenization（Mesh approach）把敏感值映射到稳定 placeholders，让 LLM 保留代码/关系语义。Network egress：LLM services 放在 dedicated VPC/VNet subnet，只 whitelist `api.openai.com`、`api.anthropic.com` 等；阻断所有其他 outbound。2026 年事件驱动：Vercel supply-chain attack 通过 compromised CI/CD credentials 外泄了数千 customer deployments 的 env vars。

**类型：** 学习
**语言：** Python（stdlib, toy PII-scrubber + audit-log writer）
**前置知识：** 第 17 阶段 · 19（AI Gateways）, 第 17 阶段 · 13（Observability）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 枚举四种 secret-management anti-patterns（config files in VCS、hardcoded env、spreadsheets、static keys），并说出替代方案。
- 解释 AI-gateway-pulls-from-vault pattern 作为 2026 production standard。
- 实现带 consistent tokenization 的 PII scrubber（same value → same placeholder），让语义得以保留。
- 说出 2026 Vercel supply-chain incident，以及它对 CI/CD credential hygiene 的教训。

## The Problem / 问题

实习生提交了带 API keys 的 `.env`。他们很快删除。Keys 已经进入 git history；GitGuardian scan 抓到了。你的 rotation process 是“Slack 通知团队，更新 40 个 config files，redeploy 所有 services”。8 小时后，一半 services 已上线新 key，另一半还在等 deploy windows。

另外，用户 prompts 包含 “My SSN is 123-45-6789.” Prompt 发给 OpenAI。你有 BAA，但内部策略要求转发前 mask PII。你没做。

再另外，你的 EKS cluster 中 LLM pod 可以访问任意互联网 host。有人通过 DNS lookup 向攻击者控制域名 exfil 数据。没有任何东西阻断它。

LLM services 的安全必须覆盖这三个向量：Vault-backed credentials、PII scrubbing、Network egress filtering、Audit logs。

## The Concept / 概念

### Centralized vault + IAM-role pull / Centralized vault + IAM-role pull

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。唯一事实源。

**IAM role**：app/gateway 用自身 IAM identity 认证，而不是 static key。Vault 根据 token lifetime 返回 secret。

**The AI-gateway pattern**：gateway 在 request time 从 vault 拉 `OPENAI_API_KEY`。在 vault 中 rotate；下一个请求拿到新 key。无需 redeploy。

### Rotation policy ≤ 90 days / Rotation policy ≤ 90 天

所有 API keys、vault root tokens、CI/CD credentials。能自动 rotate 就自动。手动 rotation 要 logged and tracked。

### Secret scanning / Secret scanning

- **TruffleHog** — 对 commits 做 regex + entropy。
- **GitGuardian** — commercial，高准确率。
- **Gitleaks** — OSS，跑在 CI。

每次 commit 都跑。检测到新 secret 就 block PR。

### Zero-trust posture / Zero-trust 姿态

- 所有账号必须 MFA。
- SSO via SAML/OIDC。
- RBAC（role-based）或 ABAC（attribute-based）做细粒度访问。
- Short-lived tokens（小时级，不是天级）。
- Device posture：仅允许启用 disk encryption 的 corp devices。

### PII / PHI scrubbing / PII / PHI 清洗

在 prompt 离开你的 infra 之前：

1. Entity recognition（spaCy NER、Presidio、commercial）。
2. Mask 匹配实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. Consistent tokenization（Mesh approach）：同一值映射到同一 placeholder，让 LLM 保留关系。
4. 可选：对 LLM response 做 reverse mapping。

Static regex filters 能抓基本 patterns；NER 能抓更多。两者都用。

### Input + output guardrails / Input + output guardrails

Input：阻断 known jailbreaks、forbidden topics；按用户 rate-limit。

Output：regex scrub leaked secrets（API key patterns、refusal contexts 中的 email patterns），classifier 检查 policy violations。

### Network egress whitelist / Network egress whitelist

LLM services 放在 dedicated subnet：
- Whitelist：`api.openai.com`、`api.anthropic.com`、vector DB endpoints、vault endpoints。
- 其他全部 drop。
- DNS 通过 allowlist-only resolver（避免 DNS-tunneling exfil）。

### Audit log / Audit log

每次 LLM call 的 immutable log：
- Timestamp。
- User / tenant。
- Prompt hash（出于隐私，不存 raw prompt）。
- Model + version。
- Token counts。
- Cost。
- Response hash。
- Any guardrail trips。

按监管要求保留（SOC 2 1 year，HIPAA 6 years）。

### The 2026 Vercel incident / 2026 Vercel 事件

Supply-chain attack：compromised CI/CD credentials 外泄了数千 customer deployments 的 env vars。教训：CI/CD credentials 等同生产权限。放进 vault。缩小 scope。激进 rotate。

### Numbers you should remember / 你应该记住的数字

- Rotation policy：≤ 90 days。
- 每次 commit 扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD creds compromised → 数千 customer env vars leaked。
- Audit log retention：SOC 2 = 1 year，HIPAA = 6 years。

## Build It / 动手构建

用 `code/main.py` 实现最小 PII scrubber 和 append-only audit log，重点检查 consistent tokenization 是否能在多次 prompt 中保持同一 placeholder。

## Use It / 应用它

`code/main.py` 实现一个带 consistent tokenization 的 toy PII scrubber 和 append-only audit log。

## Ship It / 交付它

本课产出 `outputs/skill-llm-security-plan.md`。给定 regulatory scope 和 current state，它会规划 vault migration、scrubber、egress、audit log。

## Exercises / 练习

1. 运行 `code/main.py`。发送两个引用同一 SSN 的 prompts。确认两者得到同一 placeholder。
2. 为一个调用 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS deployment 设计 network egress policy。
3. 你在 git history 中发现一个 2 年前的 key。正确响应是 rotate key、scrub history，还是两者？说明理由。
4. Audit log 每天增长 10 GB。设计 retention tiers（hot 30d、warm 12mo、cold 6yr）。
5. 论证 reverse-tokenization（把真实值替换回 LLM response）相比保持 placeholders 可见，是否值得复杂度。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Vault | “secrets store” | Centralized credential management service |
| IAM role | “identity-based auth” | App assumed role；返回 short-lived creds |
| OIDC for CI/CD | “cloud-issued tokens” | CI 不放 static keys，通过 OIDC identity |
| TruffleHog / GitGuardian / Gitleaks | “secret scanners” | Commit-time secret detection |
| RBAC / ABAC | “access control” | Role-based vs attribute-based |
| PII scrubbing | “data masking” | 删除或 tokenize sensitive entities |
| Consistent tokenization | “stable placeholders” | 同一值 → 每次同一 token |
| Mesh approach | “Mesh tokenization” | 语义保留的 tokenization pattern |
| Egress whitelist | “outbound allowlist” | 只允许访问 permitted domains |
| Audit log | “immutable history” | 合规用 append-only record |

## Further Reading / 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection and anonymization。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
