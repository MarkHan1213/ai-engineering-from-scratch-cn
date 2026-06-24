# OCR & Document Understanding / OCR 与文档理解

> OCR 是三阶段 pipeline：detect text boxes、recognise characters、再做 layout。每个现代 OCR system 要么重排这些阶段，要么把它们合并。

**Type / 类型：** Learn + Use / 学习 + 使用
**Languages / 语言：** Python
**Prerequisites / 前置知识：** Phase 4 Lesson 06 (Detection), Phase 7 Lesson 02 (Self-Attention)
**Time / 时间：** 约 45 分钟

## Learning Objectives / 学习目标

- 梳理经典 OCR pipeline（detect -> recognise -> layout）和现代 end-to-end alternatives（Donut、Qwen-VL-OCR）
- 为 sequence-to-sequence OCR training 实现 CTC（Connectionist Temporal Classification）loss
- 使用 PaddleOCR 或 EasyOCR 做 production document parsing，不需要训练
- 区分 OCR、layout parsing 和 document understanding，并按任务选择正确工具

## The Problem / 问题

充满文本的 images 无处不在：receipts、invoices、IDs、scanned books、forms、whiteboards、signs、screenshots。从它们提取 structured data，不只是字符，而是“这是 total amount”，是最有价值的 applied-vision 问题之一。

这个领域分成三层能力：

1. **OCR proper**：把 pixels 变成 text。
2. **Layout parsing**：把 OCR output 分组成 regions（title、body、table、header）。
3. **Document understanding**：从 layout 中抽取 structured fields（`invoice_total = $42.50`）。

每一层都有 classical 和 modern approaches，而“我想从图像里取文字”和“我需要这张 receipt 的 total amount”之间的差距，比多数团队意识到的更大。

## The Concept / 概念

### The classical pipeline / 经典 pipeline

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **Text detection** 产生每行或每词的 quadrilaterals。
- **Recognition** 把每个 region crop 到固定高度，运行 CNN + BiLSTM + CTC 生成 character sequence。
- **Layout** 重建 reading order（Latin 是 top-to-bottom、left-to-right；Arabic、Japanese 不同）。

### CTC in one paragraph / 一段话理解 CTC

OCR recognition 要从 fixed-length feature map 产生 variable-length sequence。CTC（Graves et al., 2006）允许你在没有 character-level alignment 的情况下训练它。Model 在每个 time step 输出 over (vocab + blank) 的 distribution；CTC loss 会对所有 alignment 求边际化，这些 alignment 在合并重复并移除 blanks 后都会变成 target text。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 是 CRNN 在 2015 年能工作、并且 2026 年仍训练大多数 production OCR models 的原因。

### Modern end-to-end models / 现代 end-to-end models

- **Donut**（Kim et al., 2022）：ViT encoder + text decoder；读取 image 后直接 emit JSON。没有 text detector，没有 layout module。
- **TrOCR**：用于 line-level OCR 的 ViT + transformer decoder。
- **Qwen-VL-OCR / InternVL**：针对 OCR tasks fine-tuned 的 full vision-language models；2026 年复杂文档准确率最好。
- **PaddleOCR**：成熟 production package 中的经典 DB + CRNN pipeline；仍是 open-source workhorse。

End-to-end models 需要更多数据和 compute，但跳过了 multi-stage pipelines 的 error accumulation。

### Layout parsing / Layout parsing

对 structured documents，运行 layout detector（LayoutLMv3、DocLayNet），把每个 region 标注为 Title、Paragraph、Figure、Table、Footnote。Reading order 就变成“按 layout order 遍历 regions 并 concatenate”。

对 forms，使用 **Key-Value extraction** models（visually-rich documents 用 Donut，plain scans 用 LayoutLMv3）。它们接收 image + detected text + positions，并预测 structured key-value pairs。

### Evaluation metrics / 评估指标

- **Character Error Rate (CER)**：Levenshtein distance / reference length。越低越好。干净 scans 的 production target：< 2%。
- **Word Error Rate (WER)**：word level 的同类指标。
- **F1 on structured fields**：用于 key-value tasks；衡量 `{invoice_total: 42.50}` 是否正确出现。
- **Edit distance on JSON**：用于 end-to-end document parsing；Donut 论文引入 normalised tree edit distance。

## Build It / 动手构建

### Step 1: CTC loss + greedy decoder / Step 1：CTC loss + greedy decoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用时使用高效 CuDNN implementation。Greedy decoder 比 beam search 简单，通常 CER 只差 1% 以内。

### Step 2: Tiny CRNN recogniser / Step 2：Tiny CRNN recogniser

用于 line OCR 的 minimal CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

输入 fixed-height（CNN 把 height max-pool 到 1）。Width 是 CTC 的 time dimension。

### Step 3: Synthetic OCR / Step 3：synthetic OCR

生成黑字白底 digit strings，用于端到端 smoke test。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实 OCR dataset 会加入 fonts、noise、rotation、blur 和 colour。上面的 pipeline 完全相同。

### Step 4: Training sketch / Step 4：training sketch

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个 trivial synthetic data 上，200 steps 后 loss 应该从 ~3 降到 ~0.2。

## Use It / 应用它

三条生产路径：

- **PaddleOCR**：成熟、快速、多语言。一行使用：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR**：Python-native、多语言、PyTorch backbone。
- **Tesseract**：经典工具；在 model 对老旧 scanned documents 表现不好时仍有用。

End-to-end document parsing 使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对 receipts、invoices 和结构重复的 forms，fine-tune Donut。对任意文档或带 reasoning 的 OCR，Qwen-VL-OCR 这类 VLM 是当前默认选择。

## Ship It / 交付它

本课产出：

- `outputs/prompt-ocr-stack-picker.md`：一个 prompt，根据 document type、language 和 structure 选择 Tesseract / PaddleOCR / Donut / VLM-OCR。
- `outputs/skill-ctc-decoder.md`：一个 skill，从零编写 greedy 和 beam-search CTC decoders，包括 length normalisation。

## Exercises / 练习

1. **(Easy / 简单)** 在 5-digit random numeric strings 上训练 TinyCRNN 500 steps。报告 held-out set 上的 CER。
2. **(Medium / 中等)** 用 beam search（beam_width=5）替换 greedy decoding。报告 CER delta。哪些 inputs 上 beam search 会赢？
3. **(Hard / 困难)** 在 20 张 receipts 上使用 PaddleOCR，抽取 line items，并针对手工标注的 {item_name, price} pairs 计算 F1。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| OCR | “Text from pixels” | 把 image regions 转成 character sequences |
| CTC | “Alignment-free loss” | 不需要 per-timestep labels 就能训练 sequence model 的 loss；会对 alignments 求边际化 |
| CRNN | “Classic OCR model” | Conv feature extractor + BiLSTM + CTC；2015 baseline 仍用于生产 |
| Donut | “End-to-end OCR” | ViT encoder + text decoder；直接从 image emit JSON |
| Layout parsing | “Find regions” | 检测并标注文档中的 Title/Table/Figure/Paragraph regions |
| Reading order | “Text sequence” | 把 recognised regions 排成句子的顺序；Latin 简单，mixed layouts 不简单 |
| CER / WER | “Error rates” | Character 或 word 粒度的 Levenshtein distance / reference length |
| VLM-OCR | “能读图的 LLM” | 为 OCR tasks 训练或 prompt 的 vision-language model；复杂文档的当前 SOTA |

## Further Reading / 延伸阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717)：原始 CNN+RNN+CTC architecture
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)：CTC 原论文，算法思想密集
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664)：OCR-free document understanding transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：open-source production OCR stack
