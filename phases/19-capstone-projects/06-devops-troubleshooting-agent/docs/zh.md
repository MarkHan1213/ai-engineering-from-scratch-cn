# Capstone 06 — DevOps Troubleshooting Agent for Kubernetes / 面向 Kubernetes 的 DevOps 故障排查 Agent

> AWS 的 DevOps Agent 进入 GA，Resolve AI 发布 K8s playbooks，NeuBird 演示 semantic monitoring，Metoro 把 AI SRE 绑定到 per-service SLO。生产形态已经清楚：alert webhook 触发，Agent 读取 telemetry，遍历 K8s objects graph，排序 root-cause hypotheses，然后把带 approval buttons 的 Slack brief 发出去。默认只读。所有 remediation 都需要人类批准。本 capstone 就是构建这个 Agent，在 20 个 synthetic incidents 上评估，并在 3 个共享案例上与 AWS Agent 对比。

**类型：** 综合项目
**语言：** Python（agent）, TypeScript（Slack integration）
**前置知识：** 第 11 阶段（LLM engineering）, 第 13 阶段（tools and MCP）, 第 14 阶段（agents）, 第 15 阶段（autonomous）, 第 17 阶段（infrastructure）, 第 18 阶段（safety）
**Phases exercised:** P11 · P13 · P14 · P15 · P17 · P18
**时间：** 30 小时

## Learning Objectives / 学习目标

- 用 K8s object + telemetry edge 构建可用于 RCA 的知识图谱
- 设计 read-only-by-default MCP tool surface 与 destructive action approval gate
- 实现 LangGraph root-cause agent，输出带 graph path 和 telemetry citation 的 hypotheses
- 建立 Slack 审批、audit log 和 synthetic incident suite
- 评估 RCA accuracy、time-to-hypothesis、safety 和 integration completeness

## Problem / 问题

2025-2026 年的 SRE 叙事变成了：“AI agents triage incidents, humans approve remediations.” AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产中交付了这种形态。Agent 读取 Prometheus metrics、Loki logs、Tempo traces、kube-state-metrics，以及 K8s objects knowledge graph。它在五分钟内产出带 telemetry citations 的 ranked root-cause hypothesis。没有 Slack 中明确的人类批准，它绝不执行破坏性命令。

多数难点在 scope 和 safety，而不是推理。Agent 需要 read-only-by-default RBAC surface、hardened MCP tool server，以及记录每个“被考虑过”和“真正执行过”的命令的 audit logs。它还需要知道何时超出能力范围并升级处理。而且它必须足够便宜，不能让一次 OOM-kill cascade 触发 $5k 的 Agent 账单。

## Concept / 概念

Agent 在 knowledge graph 上工作。Nodes 是 K8s objects（Pods、Deployments、Services、Nodes、HPAs、PVCs）以及 telemetry sources（Prometheus series、Loki streams、Tempo traces）。Edges 编码 ownership（Pod -> ReplicaSet -> Deployment）、scheduling（Pod -> Node）和 observation（Pod -> Prometheus series）。graph 由 kube-state-metrics sync 保持新鲜，并在每次 alert 时重新采样。

alert 触发后，Agent 从 affected object 开始 root-cause。它沿 edges 遍历，拉取相关 telemetry slices（last 15 minutes），并草拟 hypothesis。hypothesis 按 evidence 排序：有多少 telemetry citations 支持、证据有多新、多具体。top-3 hypotheses 会带着 graph-path visualizations 和 remediation approval buttons 发到 Slack。

Remediation 必须 gate。默认允许的 actions 是 read-only。破坏性 actions（scaling down、rolling back、deleting Pods）需要 Slack approval；ArgoCD rollback hooks 需要 Agent 永远不持有的 auth token。audit log 记录 Agent *considered* 的每条命令，不只是 executed 的命令，这样 review process 可以捕获 near-misses。

## Architecture / 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## Stack / 技术栈

- Observability sources: Prometheus、Loki、Tempo、kube-state-metrics
- Knowledge graph: Neo4j（managed）或 kuzu（embedded），表示 K8s objects + telemetry edges
- Agent: LangGraph，带 per-tool allow-list，默认 read-only
- Tool transport: FastMCP over StreamableHTTP；destructive tools 放在 approval gate 后的独立 server
- Models: Claude Sonnet 4.7 做 root-cause reasoning，Gemini 2.5 Flash 做 log summarization
- Remediation: ArgoCD rollback webhook、PagerDuty escalate、Slack approval card
- Audit: append-only structured log（considered、executed、approved、outcome）
- Deployment: 独立 namespace 中的 K8s deployment，使用窄 RBAC role

## Build It / 动手构建

1. **Graph ingestion.** 每 30s 把 kube-state-metrics 同步到 Neo4j/kuzu。Nodes: Pod, Deployment, Node, Service, PVC, HPA。Edges: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES。Telemetry overlay edges: OBSERVED_BY（一个 Pod 被某条 Prometheus series 观测）。

2. **Alert receiver.** FastAPI endpoint 接收 PagerDuty 或 Alertmanager webhooks。提取 affected object(s) 和 SLO breach。

3. **Read-only tool surface.** 通过 FastMCP 包装 kubectl、Prometheus query、Loki logql、Tempo traceql。每个工具都只有窄 RBAC verb（"get"、"list"、"describe"）。默认 server 中没有 "delete"、"exec"、"scale"。

4. **Root-cause agent.** 三节点 LangGraph：`sample` 拉取 last-15-minutes telemetry slice，`walk` 查询 graph 中的 neighboring objects，`hypothesize` 草拟带 telemetry citations 的 ranked root-cause candidates。

5. **Evidence scoring.** 每个 hypothesis 的 score = recency * specificity * graph-path length inverse * citation count。返回 top-3。

6. **Slack brief.** 发送一个 attachment，包含 hypothesis、graph-path visualization（server-side 渲染的 subgraph image），以及最多一个 remediation action 的 approval buttons。

7. **Remediation gate.** Destructive tools（scale down、roll back、delete）位于第二个 MCP server，背后有 approval token。只有 Slack card 被人类批准后，Agent 才能调用它们。

8. **Audit log.** append-only JSONL：对每个 candidate command，记录它是否被 considered、是否被 executed、是谁批准的。每天写入 S3。

9. **Synthetic incident suite.** 构建 20 个场景：OOMKill cascade、DNS flap、HPA thrash、PVC fill、noisy neighbor、faulty sidecar、bad ConfigMap rollout、certificate rotation、image-pull backoff 等。按 root-cause accuracy 和 time-to-hypothesis 给 Agent 打分。

## Use It / 应用它

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## Ship It / 交付它

`outputs/skill-devops-agent.md` 是交付物。给定 K8s cluster 和 alert source，Agent 会产出 ranked root-cause hypotheses，并提供 Slack-gated remediation flow。

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | RCA accuracy on scenario suite | 20 个 synthetic incidents 中 ≥80% root cause 正确 |
| 20 | Safety | audit log 中 destructive-action guard 从不在无 Slack approval 时触发 |
| 20 | Time-to-hypothesis | 从 alert 到 Slack brief 的 p50 低于 5 分钟 |
| 20 | Explainability | 每个 hypothesis 都有 graph paths 和 telemetry citations |
| 15 | Integration completeness | PagerDuty、Slack、ArgoCD、Prometheus 端到端工作 |
| **100** | | |

## Exercises / 练习

1. 在 AWS DevOps Agent demo 的相同三个 incidents 上运行你的 Agent。发布 side-by-side，并报告 Agent 发散的位置。

2. 添加 “near-miss” audit，标记 Agent *considered* 但若无审批就会造成破坏的命令。测量一周内 near-miss rate。

3. 把 hypothesis model 从 Claude Sonnet 4.7 换成自托管 Llama 3.3 70B。测量 RCA accuracy delta 和 dollar per incident。

4. 构建 causal filter：区分相关 telemetry spike 和真正 root cause。用 20-scenario labels 训练一个小 classifier。

5. 添加 rollback dry-run：在具有相同 manifest 的 staging cluster 上执行 ArgoCD rollback。Slack approval button 前先在 live cluster 中验证 rollback plan。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| K8s knowledge graph | “Cluster graph” | Nodes = K8s objects + telemetry series；edges = ownership、scheduling、observation |
| Read-only-by-default | “Scoped RBAC” | Agent 的 service account 只有 get/list/describe verbs；destructive verbs 在 approval 后的独立 server |
| Audit log | “Considered vs executed” | 每个 candidate command 的 append-only record，包含是否运行、谁批准 |
| Hypothesis ranking | “Evidence score” | Recency × specificity × graph-path length inverse × citation count |
| Slack approval card | “HITL gate” | 带 remediation buttons 的 Slack 交互消息；人类点击前 Agent 不能继续 |
| Telemetry citation | “Evidence pointer” | 支撑 claim 的 Prometheus query、Loki selector 或 Tempo trace URL |
| MTTR | “Time to resolution” | 从 alert fire 到 SLO recovery 的 wall-clock 时间 |

## Further Reading / 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 canonical reference
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — competitor reference
- [NeuBird semantic monitoring](https://www.neubird.ai) — semantic-graph approach
- [Metoro AI SRE](https://metoro.io) — SLO-first production framing
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — cluster-state source
- [LangGraph](https://langchain-ai.github.io/langgraph/) — reference agent orchestrator
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP server framework
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — gated remediation target
