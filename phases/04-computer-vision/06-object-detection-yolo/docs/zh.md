# Object Detection — YOLO from Scratch / 目标检测：从零实现 YOLO

> Detection 是 classification 加 regression，在 feature map 的每个位置运行，然后用 non-maximum suppression 清理结果。

**Type / 类型：** Build / 构建
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification), Phase 4 Lesson 05 (Transfer Learning)
**Time / 时间：** 约 75 分钟

## Learning Objectives / 学习目标

- 解释把 detection 变成 dense prediction problem 的 grid-and-anchor 设计，并说明 output tensor 中每个数字的含义
- 计算 boxes 之间的 Intersection-over-Union，并从零实现 non-maximum suppression
- 在 pretrained backbone 上构建一个最小 YOLO-style head，包括 classification、objectness 和 box-regression losses
- 阅读 detection metric row（precision@0.5、recall、mAP@0.5、mAP@0.5:0.95），并判断下一步该调哪个旋钮

## The Problem / 问题

Classification 说：“这张图是狗。”Detection 说：“狗在 pixels (112, 40, 280, 210)，猫在 (400, 180, 560, 310)，画面里没有别的东西。”这个结构性变化，也就是从每张图预测一个 label，变成预测数量可变的 labelled boxes，是每个 autonomous system、surveillance product、document layout parser 和 factory vision line 都依赖的能力。

Detection 也是视觉里所有工程权衡同时出现的地方。你希望 box 准确（regression head），希望每个 box 的 class 正确（classification head），希望模型知道什么时候没东西可检测（objectness score），还希望每个真实 object 只产生一个 prediction（non-maximum suppression）。任何一环出错，pipeline 都可能漏检 object、报告幻觉 box，或把同一个 object 用十五个略微不同的位置预测出来。

YOLO（You Only Look Once, Redmon et al. 2016）让这一切通过一次 conv net forward pass 实时运行。现代 detector（YOLOv8、YOLOv9、YOLO-NAS、RT-DETR）仍然保留这些结构性决策。学会核心后，每个 variant 都只是同一组部件的重新排列。

## The Concept / 概念

### Detection as dense prediction / 把 detection 视为 dense prediction

Classifier 每张图输出 C 个数字。YOLO-style detector 每张图输出 `(S x S x (5 + C))` 个数字，其中 S 是 spatial grid size。

```mermaid
flowchart LR
    IMG["Input 416x416 RGB"] --> BB["Backbone<br/>(ResNet, DarkNet, ...)"]
    BB --> FM["Feature map<br/>(C_feat, 13, 13)"]
    FM --> HEAD["Detection head<br/>(1x1 convs)"]
    HEAD --> OUT["Output tensor<br/>(13, 13, B * (5 + C))"]
    OUT --> DEC["Decode<br/>(grid + sigmoid + exp)"]
    DEC --> NMS["Non-max suppression"]
    NMS --> RESULT["Final boxes"]

    style IMG fill:#dbeafe,stroke:#2563eb
    style HEAD fill:#fef3c7,stroke:#d97706
    style NMS fill:#fecaca,stroke:#dc2626
    style RESULT fill:#dcfce7,stroke:#16a34a
```

每个 `S * S` grid cell 预测 `B` 个 boxes。对每个 box：

- 4 个数字描述 geometry：`tx, ty, tw, th`。
- 1 个数字是 objectness score：“这个 cell 的中心附近是否有 object？”
- C 个数字是 class probabilities。

每个 cell 总计：`B * (5 + C)`。对 VOC 来说，`S=13, B=2, C=20`，也就是每个 cell 50 个数字。

### Why grids and anchors / 为什么需要 grid 和 anchor

朴素 regression 会为每个 object 预测绝对坐标 `(x, y, w, h)`。这对 conv network 很难，因为平移图像不应该让所有 prediction 都按相同量平移，每个 object 都有空间锚点。Grid 的解决方式是：把每个 ground-truth box 分配给其中心所在的 grid cell，只有这个 cell 负责该 object。

Anchors 解决第二个问题。3x3 conv 很难从一个 16-pixel receptive field feature cell 中直接 regresses 出 500-pixel-wide box。因此我们为每个 cell 预定义 `B` 个 prior box shapes（anchors），并从每个 anchor 预测 small deltas。模型学会选择正确 anchor 并微调它，而不是从零回归。

```
Anchor box priors (example for 416x416 input):

  small:   (30,  60)
  medium:  (75,  170)
  large:   (200, 380)

At each grid cell, every anchor emits (tx, ty, tw, th, obj, c_1, ..., c_C).
```

现代 detector 常用 FPN，并在不同 resolution 上使用不同 anchor set：浅层 high-resolution map 上放 small anchors，深层 low-resolution map 上放 large anchors。思想相同，只是 scale 更多。

### Decoding predictions / 解码 predictions

Raw `tx, ty, tw, th` 不是 box coordinates；它们是 regression targets，画图前必须转换：

```
centre x  = (sigmoid(tx) + cell_x) * stride
centre y  = (sigmoid(ty) + cell_y) * stride
width     = anchor_w * exp(tw)
height    = anchor_h * exp(th)
```

`sigmoid` 把 center offset 限制在 cell 内。`exp` 允许 width 从 anchor 自由缩放且不发生符号翻转。`stride` 把 grid coordinates 缩放回 pixels。自 YOLO v2 以来，每个 YOLO version 都使用同样的 decode step。

### IoU / IoU

Detection 中衡量两个 boxes 相似度的通用指标：

```
IoU(A, B) = area(A intersect B) / area(A union B)
```

IoU = 1 表示完全相同，IoU = 0 表示没有重叠。Prediction 与 ground-truth box 的 IoU 决定它是否算 true positive（通常 IoU >= 0.5）。两个 prediction 之间的 IoU 是 NMS 用来去重的依据。

### Non-maximum suppression / Non-maximum suppression

在 adjacent anchors 上训练的 conv network，经常会为同一个 object 预测多个 overlapping boxes。NMS 保留 confidence 最高的 prediction，并删除所有与它 IoU 超过阈值的其他 prediction。

```
NMS(boxes, scores, iou_threshold):
    sort boxes by score descending
    keep = []
    while boxes not empty:
        pick the top-scoring box, add to keep
        remove every box with IoU > iou_threshold to the picked box
    return keep
```

Object detection 的典型 threshold 是 0.45。近期 detector 会用 `soft-NMS`、`DIoU-NMS`，或直接学习 suppression（RT-DETR），但结构目的相同。

### The loss / Loss

YOLO loss 是三个加权 loss 的和：

```
L = lambda_coord * L_box(pred, target, where obj=1)
  + lambda_obj   * L_obj(pred, 1,     where obj=1)
  + lambda_noobj * L_obj(pred, 0,     where obj=0)
  + lambda_cls   * L_cls(pred, target, where obj=1)
```

只有包含 object 的 cells 会贡献 box-regression 和 classification losses。没有 object 的 cells 只贡献 objectness loss（教模型保持沉默）。`lambda_noobj` 通常较小（约 0.5），因为绝大多数 cells 都是空的，否则它们会支配总 loss。

现代 variant 会把 MSE box loss 换成 CIoU / DIoU（直接优化 IoU），用 focal loss 处理 class imbalance，并用 quality focal loss 平衡 objectness。三组件结构不变。

### Detection metrics / Detection 指标

Accuracy 无法直接迁移到 detection。真正需要的四个数字是：

- **Precision@IoU=0.5**：被统计为 positive 的 predictions 中，有多少真正正确。
- **Recall@IoU=0.5**：真实 objects 中，有多少被找到了。
- **AP@0.5**：IoU threshold 0.5 时 precision-recall curve 面积；每个 class 一个数字。
- **mAP@0.5:0.95**：在 IoU thresholds 0.5、0.55、...、0.95 上对 AP 求平均。COCO metric；最严格，也最有信息量。

四个都要报告。一个 detector 如果 mAP@0.5 强但 mAP@0.5:0.95 弱，说明 localisation 大致对，但不够紧；用更好的 box-regression loss 修。高 precision 低 recall 的 detector 太保守；降低 confidence threshold，或增加 objectness weight。

## Build It / 动手构建

### Step 1: IoU / Step 1：IoU

整节课的 workhorse。处理 `(x1, y1, x2, y2)` 格式的两组 boxes。

```python
import numpy as np

def box_iou(boxes_a, boxes_b):
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

返回 `(N_a, N_b)` 的 pairwise IoUs matrix。如果要和单个 ground-truth box 比较，把其中一个 array 做成 `(1, 4)` shape。

### Step 2: Non-max suppression / Step 2：Non-max suppression

```python
def nms(boxes, scores, iou_threshold=0.45):
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)
```

确定性实现，排序带来 `O(N log N)`，在相同 input 上与 `torchvision.ops.nms` 行为一致。

### Step 3: Box encoding and decoding / Step 3：box encoding 与 decoding

在 pixel coordinates 和网络实际 regress 的 `(tx, ty, tw, th)` target 之间转换。

```python
def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

测试方式：encode 一个 box 再 decode，应该得到非常接近原始 box 的结果（当 `tx` 不在 post-sigmoid range 中时，sigmoid inverse 不是完全可逆，会有细微差异）。

### Step 4: A minimal YOLO head / Step 4：最小 YOLO head

在 feature map 上做一个 1x1 conv，并 reshape 成 `(B, S, S, num_anchors, 5 + C)`。

```python
import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y
```

Output shape：`(N, H, W, num_anchors, 5 + C)`。最后一维保存 `[tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]`。

### Step 5: Ground-truth assignment / Step 5：ground-truth assignment

对每个 ground-truth box，决定哪个 `(cell, anchor)` 负责它。

```python
def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj
```

Anchor selection 是“与 ground truth 的 best shape IoU”，这是与 YOLOv2/v3 assignment 匹配的 cheap proxy。v5 及之后使用更复杂的策略（task-aligned matching、dynamic k），但都在细化同一个思想。

### Step 6: The three losses / Step 6：三个 loss

```python
def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # box-regression loss: only on cells with objects
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # objectness loss
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # classification loss on cells with objects
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}
```

每个 YOLO tutorial 要么 hardcode、要么 sweep 的五个 hyper-parameters。比例很重要：`lambda_coord=5, lambda_noobj=0.5` 与原始 YOLOv1 论文一致，现在仍是合理默认值。

### Step 7: Inference pipeline / Step 7：inference pipeline

解码 raw head output，应用 sigmoid/exp，基于 objectness threshold 过滤，再做 NMS。

```python
def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

这就是完整 eval path：head -> decode -> threshold -> NMS。

## Use It / 应用它

`torchvision.models.detection` 提供了同样概念结构的 production detector。加载 pretrained model 只需要三行。

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
print(predictions[0].keys())
print(f"boxes:  {predictions[0]['boxes'].shape}")
print(f"scores: {predictions[0]['scores'].shape}")
print(f"labels: {predictions[0]['labels'].shape}")
```

对 real-time inference pipeline，`ultralytics`（YOLOv8/v9）是标准工具：`from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`。模型会在内部处理 decoding 和 NMS，并返回与你上面构建的相同 `boxes / scores / labels` triple。

## Ship It / 交付它

本课产出：

- `outputs/prompt-detection-metric-reader.md`：一个 prompt，把 `precision, recall, AP, mAP@0.5:0.95` 这一行转成一句诊断和最有用的下一个实验。
- `outputs/skill-anchor-designer.md`：一个 skill，给定 ground-truth boxes dataset，对 `(w, h)` 跑 k-means，返回每个 FPN level 的 anchor sets 和选择 anchor 数量所需的 coverage statistics。

## Exercises / 练习

1. **(Easy / 简单)** 实现 `box_iou`，并在 1,000 对 random box 上与 `torchvision.ops.box_iou` 对比。验证 max absolute difference 低于 `1e-6`。
2. **(Medium / 中等)** 把 `yolo_loss` 移植成使用 `CIoU` box loss 而不是 MSE 的版本。在 100-image synthetic dataset 上展示：相同 epoch 数下，CIoU 收敛到的 final mAP@0.5:0.95 优于 MSE。
3. **(Hard / 困难)** 实现 multi-scale inference：把同一张图以三个 resolution 输入模型，合并 box predictions，最后做一次 NMS。在 held-out set 上测量相对 single-scale inference 的 mAP 提升。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Anchor | “Box prior” | 每个 grid cell 预定义的 box shape，网络从它预测 delta，而不是直接预测绝对坐标 |
| IoU | “Overlap” | 两个 boxes 的 intersection-over-union；detection 中的通用相似度量 |
| NMS | “Deduplicate” | 贪心算法：保留最高分 prediction，并移除超过阈值的 overlapping predictions |
| Objectness | “这里有没有东西” | 每个 anchor、每个 cell 的 scalar，预测是否有 object 中心落在这个 cell |
| Grid stride | “Downsample factor” | 每个 grid cell 对应多少 pixel；416-px input 搭配 13-grid head 时 stride 是 32 |
| mAP | “Mean average precision” | precision-recall curve 下方面积的平均值，先按 class 平均，再按 COCO 的 IoU thresholds 平均 |
| AP@0.5 | “PASCAL VOC AP” | IoU threshold 为 0.5 的 average precision；较宽松版本 |
| mAP@0.5:0.95 | “COCO AP” | 在 0.5..0.95、步长 0.05 的 IoU thresholds 上求平均；严格版本，也是当前社区标准 |

## Further Reading / 延伸阅读

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640)：奠基论文；之后每个 YOLO 都是在细化这个结构
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767)：引入 multi-scale FPN-style heads 的论文；图示仍然最清楚
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com)：当前生产参考；覆盖 dataset formats、augmentations、training recipes
- [The Illustrated Guide to Object Detection (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904)：对整个 detector zoo 最好的 plain-English 导览；理解 DETR、RetinaNet、FCOS 和 YOLO 关系时很有价值
