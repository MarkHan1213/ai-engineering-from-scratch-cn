# Multi-Object Tracking & Video Memory / 多目标跟踪与视频记忆

> Tracking 就是 detection 加 association。每一帧做 detection。把这一帧的 detections 与上一帧的 tracks 按 ID 匹配起来。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 4 Lesson 06 (YOLO Detection), Phase 4 Lesson 08 (Mask R-CNN), Phase 4 Lesson 24 (SAM 3)
**Time / 时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 区分 tracking-by-detection 和 query-based tracking，并说出算法家族（SORT、DeepSORT、ByteTrack、BoT-SORT、SAM 2 memory tracker、SAM 3.1 Object Multiplex）
- 从零实现 IoU + Hungarian assignment，用于 classical tracking-by-detection
- 解释 SAM 2 的 memory bank，以及为什么它比 IoU-based association 更能处理 occlusion
- 读懂三个 tracking metrics（MOTA、IDF1、HOTA），并为给定 use case 选择重要指标

## The Problem / 问题

Detector 告诉你单帧里 objects 在哪里。Tracker 告诉你 frame `t` 中的哪个 detection 与 frame `t-1` 中的某个 detection 是同一个 object。没有这个能力，你就无法统计 objects 穿过一条线的数量，无法在 occlusion 后继续跟踪一个球，也无法知道 “car #4 has been in the lane for 8 seconds.”

Tracking 是每个 video-facing product 的必备能力：sports analytics、surveillance、autonomous driving、medical video analysis、wildlife monitoring、wordmark counting。核心 building blocks 是共享的：per-frame detector、motion model（Kalman filter 或更丰富的模型）、association step（在 IoU / cosine / learned features 上跑 Hungarian algorithm），以及 track lifecycle（birth、update、death）。

2026 年带来了两个新模式：**SAM 2 memory-based tracking**（用 feature-memory 替代 motion-model association）和 **SAM 3.1 Object Multiplex**（为同一 concept 的多个 instances 使用 shared memory）。本课先走 classical stack，再讲 memory-based approach。

## The Concept / 概念

### Tracking-by-detection / Tracking-by-detection

```mermaid
flowchart LR
    F1["Frame t"] --> DET["Detector"] --> D1["Detections at t"]
    PREV["Tracks up to t-1"] --> PREDICT["Motion predict<br/>(Kalman)"]
    PREDICT --> PRED["Predicted tracks at t"]
    D1 --> ASSOC["Hungarian assignment<br/>(IoU / cosine / motion)"]
    PRED --> ASSOC
    ASSOC --> UPDATE["Update matched tracks"]
    ASSOC --> NEW["Birth new tracks"]
    ASSOC --> DEAD["Age unmatched tracks; delete after N"]
    UPDATE --> NEXT["Tracks at t"]
    NEW --> NEXT
    DEAD --> NEXT

    style DET fill:#dbeafe,stroke:#2563eb
    style ASSOC fill:#fef3c7,stroke:#d97706
    style NEXT fill:#dcfce7,stroke:#16a34a
```

2026 年你会遇到的每个 tracker 都是这个 loop 的变体。差异在于：

- **SORT**（2016）：Kalman filter + IoU Hungarian。简单、快速，没有 appearance model。
- **DeepSORT**（2017）：SORT + 每个 track 的 CNN-based appearance feature（ReID embedding）。更能处理 crossing。
- **ByteTrack**（2021）：把 low-confidence detections 作为第二阶段关联；无需 appearance features，却是 MOT17 top performer。
- **BoT-SORT**（2022）：Byte + camera motion compensation + ReID。
- **StrongSORT / OC-SORT**：ByteTrack descendants，拥有更好的 motion 和 appearance。

### Kalman filter in one paragraph / 用一段话理解 Kalman filter

Kalman filter 为每个 track 维护 state `(x, y, w, h, dx, dy, dw, dh)` 以及 covariance。每一帧先用 constant-velocity model **predict** state，再用 matched detection **update**。当 predict uncertainty 很高时，update 会更信任 detection。这会产生平滑 trajectories，并让 track 可以穿过短暂 occlusion（1-5 frames）。

每个 classical tracker 都在 motion-prediction step 中使用 Kalman filter。

### The Hungarian algorithm / Hungarian algorithm

给定一个 `M x N` cost matrix（tracks x detections），找出 one-to-one assignment，使 total cost 最小。Cost 通常是 `1 - IoU(track_bbox, detection_bbox)`，或 appearance features 的 negative cosine similarity。Runtime 是 O((M+N)^3)；当 M、N 到约 1000 时，通过 `scipy.optimize.linear_sum_assignment` 在 Python 中也足够快。

### ByteTrack's key idea / ByteTrack 的核心思想

Standard trackers 会丢弃 low-confidence detections（< 0.5）。ByteTrack 会保留它们作为 **second-stage candidates**：先把 tracks 匹配到 high-confidence detections，再让 unmatched tracks 用稍微宽松的 IoU threshold 去匹配 low-confidence detections。这样可以恢复短 occlusions，并减少 crowds 附近的 ID switches。

### SAM 2 memory-based tracking / SAM 2 memory-based tracking

SAM 2 通过保存 per-instance spatio-temporal features 的 **memory bank** 来处理 video。给定某一帧上的 prompt（click、box、text），它会把 instance 编码进 memory。在后续 frames 中，memory 与新 frame 的 features 做 cross-attention，decoder 为同一个 instance 生成新 frame 中的 mask。

没有 Kalman filter，也没有 Hungarian assignment。Association 隐式发生在 memory-attention operation 中。

优点：

- 对大 occlusions 鲁棒（memory 会跨多帧携带 instance identity）。
- 与 SAM 3 text prompts 结合时支持 open-vocabulary。
- 不需要 separate motion model。

缺点：

- 对 many-object tracking 来说比 ByteTrack 慢。
- Memory bank 会增长；context window 受限。

### SAM 3.1 Object Multiplex / SAM 3.1 Object Multiplex

之前 SAM 2 / SAM 3 tracking 会为每个 instance 保留 separate memory bank。50 个 objects 就是 50 个 memory banks。Object Multiplex（March 2026）把它们折叠成一个 shared memory，并使用 **per-instance query tokens**。Cost 随 instances 数量的增长是 sub-linear。

Multiplex 是 2026 年 crowd tracking 的新默认选择：concert crowds、warehouse workers、traffic intersections。

### Three metrics to know / 三个必须知道的指标

- **MOTA（Multi-Object Tracking Accuracy）**：1 - (FN + FP + ID switches) / GT。按 error type 加权；一个混合了 detection 与 association failures 的单一指标。
- **IDF1（ID F1）**：ID precision 和 recall 的 harmonic mean。专注于每个 ground-truth track 在时间上保持 ID 的能力。对 ID-switch-sensitive tasks，比 MOTA 更合适。
- **HOTA（Higher Order Tracking Accuracy）**：分解为 detection accuracy（DetA）和 association accuracy（AssA）。2020 年以来的 community standard；最全面。

Surveillance（who is who）报告 IDF1。Sports analytics（counting passes）报告 HOTA。一般 academic comparison 报告 HOTA。

## Build It / 动手构建

### Step 1: IoU-based cost matrix / 步骤 1：基于 IoU 的 cost matrix

```python
import numpy as np


def bbox_iou(a, b):
    """
    a, b: (N, 4) arrays of [x1, y1, x2, y2].
    Returns (N_a, N_b) IoU matrix.
    """
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

### Step 2: Minimal SORT-style tracker / 步骤 2：Minimal SORT-style tracker

这里为了简洁省略 fixed constant-velocity Kalman，我们只使用简单 IoU association；production 中 Kalman predict 是必需的。`sort` Python package 提供完整版本。

```python
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, tid, bbox, frame):
        self.id = tid
        self.bbox = bbox
        self.last_frame = frame
        self.hits = 1

    def update(self, bbox, frame):
        self.bbox = bbox
        self.last_frame = frame
        self.hits += 1


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def step(self, detections, frame):
        if not self.tracks:
            for d in detections:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1
            return [(t.id, t.bbox) for t in self.tracks]

        track_boxes = np.array([t.bbox for t in self.tracks])
        det_boxes = np.array(detections) if len(detections) else np.empty((0, 4))

        iou = bbox_iou(track_boxes, det_boxes) if len(det_boxes) else np.zeros((len(track_boxes), 0))
        cost = 1 - iou
        cost[iou < self.iou_threshold] = 1e6

        matched_track = set()
        matched_det = set()
        if cost.size > 0:
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < 1.0:
                    self.tracks[r].update(det_boxes[c], frame)
                    matched_track.add(r); matched_det.add(c)

        for i, d in enumerate(det_boxes):
            if i not in matched_det:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        return [(t.id, t.bbox) for t in self.tracks]
```

60 行。输入 per-frame detections，返回 per-frame track IDs。真实系统会加入 Kalman predict、ByteTrack second-stage re-match 和 appearance features。

### Step 3: Synthetic trajectory test / 步骤 3：Synthetic trajectory test

```python
def synthetic_frames(num_frames=20, num_objects=3, H=240, W=320, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(20, 200, size=(num_objects, 2))
    velocities = rng.uniform(-5, 5, size=(num_objects, 2))
    frames = []
    for f in range(num_frames):
        dets = []
        for i in range(num_objects):
            cx, cy = starts[i] + f * velocities[i]
            dets.append([cx - 10, cy - 10, cx + 10, cy + 10])
        frames.append(dets)
    return frames


tracker = SimpleTracker()
for f, dets in enumerate(synthetic_frames()):
    tracks = tracker.step(dets, f)
```

三条直线运动的 objects 应该在全部 20 frames 中保持 ID。

### Step 4: ID-switch metric / 步骤 4：ID-switch metric

```python
def count_id_switches(tracks_per_frame, gt_per_frame):
    """
    tracks_per_frame:  list of list of (track_id, bbox)
    gt_per_frame:      list of list of (gt_id, bbox)
    Returns number of ID switches.
    """
    prev_assignment = {}
    switches = 0
    for tracks, gts in zip(tracks_per_frame, gt_per_frame):
        if not tracks or not gts:
            continue
        t_boxes = np.array([b for _, b in tracks])
        g_boxes = np.array([b for _, b in gts])
        iou = bbox_iou(g_boxes, t_boxes)
        for g_idx, (gt_id, _) in enumerate(gts):
            j = iou[g_idx].argmax()
            if iou[g_idx, j] > 0.5:
                t_id = tracks[j][0]
                if gt_id in prev_assignment and prev_assignment[gt_id] != t_id:
                    switches += 1
                prev_assignment[gt_id] = t_id
    return switches
```

这是一个简化的 IDF1-adjacent metric：计算 ground-truth object 被分配的 predicted track ID 改变了多少次。真实 MOTA / IDF1 / HOTA 工具在 `py-motmetrics` 和 `TrackEval` 中。

## Use It / 使用它

2026 年的 production trackers：

- `ultralytics`：内置 YOLOv8 + ByteTrack / BoT-SORT。`results = model.track(source, tracker="bytetrack.yaml")`。默认选择。
- `supervision`（Roboflow）：ByteTrack wrappers 加 annotation utilities。
- SAM 2 / SAM 3.1：通过 `processor.track()` 做 memory-based tracking。
- Custom stack：detector（YOLOv8 / RT-DETR）+ `sort-tracker` / `OC-SORT` / `StrongSORT`。

选择方式：

- 30+ fps 的 pedestrians / cars / boxes：**ByteTrack with ultralytics**。
- Crowd 中同一类的大量 instances：**SAM 3.1 Object Multiplex**。
- 有明显 appearance 且 occlusions 很重：**DeepSORT / StrongSORT**（ReID features）。
- Sports / complex interactions：**BoT-SORT** 或 learned trackers（MOTRv3）。

## Ship It / 交付内容

本课会产出：

- `outputs/prompt-tracker-picker.md`：根据 scene type、occlusion patterns 和 latency budget，在 SORT / ByteTrack / BoT-SORT / SAM 2 / SAM 3.1 之间做选择。
- `outputs/skill-mot-evaluator.md`：写出完整 evaluation harness，用 ground-truth tracks 评估 MOTA / IDF1 / HOTA。

## Exercises / 练习

1. **（Easy）** 用 3、10、30 个 objects 运行上面的 synthetic tracker。报告每种情况下的 ID-switch count。指出 simple IoU-only association 从哪里开始失败。
2. **（Medium）** 在 association 前加入 constant-velocity Kalman predict step。展示短暂（2-3 frame）occlusions 不再导致 ID switches。
3. **（Hard）** 把 SAM 2 的 memory-based tracker（通过 `transformers`）集成为 alternative tracker backend。在 30 秒 crowd clip 上同时运行 SimpleTracker 和 SAM 2，并为 5 个显著人物手动标注 ground-truth IDs，对比 ID-switch counts。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Tracking-by-detection | “Detect then associate” | Per-frame detector + 基于 IoU / appearance 的 Hungarian assignment |
| Kalman filter | “Motion predict” | Linear dynamics + covariance，用于平滑 track predictions 和处理 occlusion |
| Hungarian algorithm | “Optimal assignment” | 求解 minimum-cost bipartite matching problem；`scipy.optimize.linear_sum_assignment` |
| ByteTrack | “Low-confidence second pass” | 把 unmatched tracks 重新匹配到 low-confidence detections，以恢复短 occlusions |
| DeepSORT | “SORT + appearance” | 增加 ReID feature 进行 cross-frame matching；更利于 ID preservation |
| Memory bank | “SAM 2 trick” | 跨 frames 存储 per-instance spatio-temporal features；用 cross-attention 替代显式 association |
| Object Multiplex | “SAM 3.1 shared memory” | 用带 per-instance queries 的 single shared memory 高效跟踪大量 objects |
| HOTA | “Modern tracking metric” | 分解 detection 和 association accuracy；community standard |

## Further Reading / 延伸阅读

- [SORT (Bewley et al., 2016)](https://arxiv.org/abs/1602.00763) — minimal tracking-by-detection paper
- [DeepSORT (Wojke et al., 2017)](https://arxiv.org/abs/1703.07402) — adds appearance feature
- [ByteTrack (Zhang et al., 2022)](https://arxiv.org/abs/2110.06864) — low-confidence second pass
- [BoT-SORT (Aharon et al., 2022)](https://arxiv.org/abs/2206.14651) — camera motion compensation
- [HOTA (Luiten et al., 2020)](https://arxiv.org/abs/2009.07736) — decomposed tracking metric
- [SAM 2 video segmentation (Meta, 2024)](https://ai.meta.com/sam2/) — memory-based tracker
- [SAM 3.1 Object Multiplex (Meta, March 2026)](https://ai.meta.com/blog/segment-anything-model-3/)
