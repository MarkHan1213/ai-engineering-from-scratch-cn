# 3D Generation / 3D 生成

> 3D 是最能借力 2D-to-3D 的模态。2023 年的突破是 3D Gaussian Splatting。2024-2026 年的生成路线，是在它上面叠 multi-view diffusion + 3D reconstruction，用单个 prompt 或照片生成物体和场景。

**Type / 类型：** Learn / 学习
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 比较 mesh、point cloud、voxel、SDF、NeRF 和 3D Gaussian 的表示权衡
- 解释 2026 年 text/image-to-3D 为什么常拆成 multi-view diffusion + reconstruction
- 理解 3D Gaussian Splatting 的参数、渲染和优化路径
- 为游戏、电商、novel-view 和研究任务选择合适的 3D 生成 pipeline

## The Problem / 问题

3D content 很难：

- **Representation / 表示。** Meshes、point clouds、voxel grids、signed distance fields（SDFs）、neural radiance fields（NeRFs）、3D Gaussians。每种都有 trade-offs。
- **Data scarcity / 数据稀缺。** ImageNet 有 14M images。最大的干净 3D dataset（Objaverse-XL, 2023）约 10M objects，而且很多质量不高。
- **Memory / 内存。** 512³ voxel grid 是 128M voxels；有用的 scene NeRF 每条 ray 需要 1M samples。Generation 比 reconstruction 更难。
- **Supervision / 监督。** 2D 图像有 pixels。3D 通常只有少量 2D views，必须 lift 到 3D。

2026 年的 stack 把问题拆成两步。第一，使用 diffusion model 生成 *2D multi-view images*。第二，对这些图拟合一个 *3D representation*（通常是 Gaussian splatting）。

## The Concept / 概念

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### Representation: 3D Gaussian Splatting (Kerbl et al., 2023) / 表示：3D Gaussian Splatting

把一个场景表示成约 1M 个 3D Gaussians 的云。每个 Gaussian 有 59 个参数：position（3）、covariance（6，或 quaternion 4 + scale 3）、opacity（1）、spherical-harmonics color（degree 3 时 48，degree 0 时 3）。

Rendering = projection + alpha-compositing。快（4090 上 1080p 约 100 fps）。可微。用 gradient descent 对 ground-truth photos 拟合。一个场景在消费级 GPU 上 5-30 分钟拟合完成。

上面叠了两个 2023-2024 创新：
- **Generative Gaussian splats。** LGM、LRM、InstantMesh 等模型直接从一张或几张图预测 Gaussian cloud。
- **4D Gaussian Splatting。** 给 Gaussians 加 per-frame offsets，用于动态场景。

### Multi-View Diffusion / 多视角 diffusion

Fine-tune pretrained image diffusion model，让它从 text prompt 或 single image 生成同一物体的多个一致视角。Zero123（Liu et al., 2023）、MVDream（Shi et al., 2023）、SV3D（Stability, 2024）、CAT3D（Google, 2024）。通常输出物体周围 4-16 个 views，再通过 Gaussian splatting 或 NeRF lift 到 3D。

### Text-to-3D Pipelines / Text-to-3D 流水线

| Model / 模型 | Input / 输入 | Output / 输出 | Time / 时间 |
|-------|-------|--------|------|
| DreamFusion (2022) | text | NeRF via SDS | ~1 hour per asset |
| Magic3D | text | mesh + texture | ~40 min |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1 min |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30 min |
| LRM (Meta, 2023) | image | triplane | ~5 s |
| InstantMesh (2024) | image | mesh | ~10 s |
| SV3D (Stability, 2024) | image | novel views | ~2 min |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1 min |
| TripoSR (2024) | image | mesh | ~1 s |
| Meshy 4 (2025) | text + image | PBR mesh | ~30 s |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30 s |

2025-2026 方向：直接输出适合 game engines 的 PBR materials 的 text-to-mesh 模型。Multi-view diffusion 中间步骤仍是通用 objects 上表现最好的 recipe。

### NeRF (for Context) / NeRF 背景

Neural Radiance Field（Mildenhall et al., 2020）。一个小 MLP 接收 `(x, y, z, view direction)`，输出 `(color, density)`。通过沿 rays 积分来 render。质量上超过 mesh-based novel-view synthesis，但渲染慢 100-1000 倍。大多数实时用途已经被 Gaussian splatting 替代，但研究里仍很重要。

## Build It / 动手构建

`code/main.py` 实现 toy 2D “Gaussian splatting” fit：把一个 synthetic target image（平滑 gradient）表示成一组 2D Gaussian splats。通过 gradient descent 优化 positions、colors 和 covariances，让它匹配 target。你会看到两个核心操作：forward render（splat + alpha-composite）和 gradient descent 拟合。

### Step 1: 2D Gaussian splat / 第 1 步：2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Step 2: render by summing splats / 第 2 步：把 splats 求和渲染

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实 3D Gaussian splatting 会按深度排序 Gaussians，并按顺序 alpha-composite。我们的 2D toy 只是求和。

### Step 3: fit by gradient descent / 第 3 步：用 gradient descent 拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Pitfalls / 常见坑

- **View inconsistency。** 如果独立生成 4 个 views，且它们对 object structure 彼此矛盾，3D fit 会变模糊。修复：用 shared attention 的 multi-view diffusion。
- **Back-side hallucination。** Single-image → 3D 必须发明看不见的背面，质量波动很大。
- **Gaussian splat explosion。** 不加约束训练会增长到 10M splats 并 overfit。3D-GS 原论文中的 densification + pruning heuristics 是必要的。
- **Topology issues。** 从 implicit fields（SDFs）提取的 meshes 经常有洞或 self-intersections。上线前跑 remesher（如 blender 的 voxel remesh）。
- **License of training data。** Objaverse 的 licenses 混杂；商业使用随模型而异。

## Use It / 应用它

| Task / 任务 | 2026 pick |
|------|-----------|
| Scene reconstruction from photos | Gaussian splatting（3DGS、Gsplat、Scaniverse） |
| Text-to-3D object for games | Meshy 4 或 Rodin Gen-1.5（PBR output） |
| Image-to-3D | Hunyuan3D 2.0、TripoSR、InstantMesh |
| Novel-view synthesis from few images | CAT3D、SV3D |
| Dynamic scene reconstruction | 4D Gaussian Splatting |
| Avatar / clothed human | Gaussian Avatar、HUGS |
| Research / SOTA | 上周刚发布的那个模型 |

对游戏或电商 pipeline 中的 production 3D：Meshy 4 或 Rodin Gen-1.5 能直接输出可进 Unity / Unreal 的 PBR meshes。

## Ship It / 交付它

保存 `outputs/skill-3d-pipeline.md`。Skill 接收 3D brief（input：text / one image / few images；output：mesh / splat / NeRF；usage：render / game / VR），并输出：pipeline（multi-view diffusion + fit 或 direct mesh model）、base model、iteration budget、topology post-processing 和所需 material channels。

## Exercises / 练习

1. **Easy / 简单。** 用 4、16、64 个 Gaussians 运行 `code/main.py`。报告 final MSE vs target。
2. **Medium / 中等。** 扩展到 color Gaussians（RGB）。确认 reconstruction 匹配 target color pattern。
3. **Hard / 困难。** 使用 gsplat 或 Nerfstudio，从 50-photo capture 重建真实物体。报告 fit time 和 held-out views 上的 final SSIM。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | 把场景表示成 3D Gaussians 云；可微 alpha-composite render。 |
| NeRF | "Neural radiance field" | 在 3D 点输出 color + density 的 MLP；通过 ray integration 渲染。 |
| Triplane | "Three 2-D planes" | 把 3D 分解成三张轴对齐 2-D feature grids；比 volumetric 便宜。 |
| SDS | "Score distillation sampling" | 用 2D-diffusion score 当 pseudo-gradient 来训练 3D model。 |
| Multi-view diffusion | "Many views at once" | 一次输出一组一致 camera views 的 diffusion model。 |
| PBR | "Physically-based rendering" | 含 albedo、roughness、metallic、normal channels 的 material。 |
| Densification | "Grow splats" | 3DGS training heuristic：在高梯度区域 split / clone splats。 |

## Production Note: 3D Has No Shared Substrate Yet / 生产备注：3D 还没有统一底座

不像图像（latent diffusion + DiT）和视频（spatiotemporal DiT），3D 在 2026 年还没有单一 dominant runtime。生产决策树会按表示分叉：

- **NeRF / triplane。** Inference 是 ray-marching + 每个 sample 一次 MLP forward。512² render 需要数百万次 MLP forwards。要激进 batch ray samples；SDPA/xformers 适用。
- **Multi-view diffusion + LRM reconstruction。** Two-stage pipeline。Stage 1（multi-view DiT）就是 Lesson 07 里的 diffusion server。Stage 2（LRM transformer）是对 views 的 one-shot forward pass。整体 latency profile 是 “diffusion + one-shot”，需要分别选择 serving primitives。
- **SDS / DreamFusion。** 这是 per-asset optimization，不是 inference。构建 jobs，而不是 request handlers。

对大多数 2026 产品，正确答案是：“请求时跑 multi-view diffusion model，异步 reconstruct 到 3DGS，再服务 3DGS 做实时查看”。这样能把 workload 干净拆成 GPU-inference server（快）和 offline optimizer（慢）。

## Further Reading / 延伸阅读

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — multi-view diffusion。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
