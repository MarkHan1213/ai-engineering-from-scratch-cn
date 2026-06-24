# Edge Inference — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson / 边缘推理：Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> Edge 的核心约束是 memory bandwidth，不是 compute。Mobile DRAM 约 50-90 GB/s；datacenter HBM3 超过 2-3 TB/s，差 30-50x。Decode 是 memory-bound，所以差距具有决定性。2026 年格局分成四类。Apple M4/A18 Neural Engine 峰值 38 TOPS，使用 unified memory（无 CPU↔NPU copy）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上运行 Llama 3.1 8B（Q4）约 41 tok/s（原生的 70-80%）；17.6k GitHub stars，OpenAI-compatible API，移动端覆盖约 70-75%。NVIDIA Jetson Orin Nano Super（8GB）可放 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 跑 gpt-oss-20b 约 40 tok/s；Jetson T4000（JetPack 7.1）是 AGX Orin 的 2x。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、chunked prefill，Bosch、ThunderSoft、MediaTek 在 CES 2026 展示过。

**类型：** 学习
**语言：** Python（stdlib, toy bandwidth-bound decode simulator）
**前置知识：** 第 17 阶段 · 04（vLLM Serving Internals）, 第 17 阶段 · 09（Production Quantization）
**时间：** 约 60 分钟

## Learning Objectives / 学习目标

- 解释为什么 mobile LLM inference 是 memory-bandwidth-bound，而 compute 是次要因素。
- 枚举四类 edge targets（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），并匹配 use case。
- 说出 2026 年 WebGPU coverage gap（Firefox Android catching up）和 Safari iOS 26 landing。
- 为每个 target 选择 quantization format（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，browser 用 WebGPU Q4，Jetson Thor 用 NVFP4）。

## The Problem / 问题

客户想要一个 on-device chatbot：voice-first、private-by-default、offline 可用。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 约 55 tok/s，可以。在 iPhone 16 Pro 上，同一模型约 3 tok/s，不可以。在 Snapdragon 8 Gen 3 的中端 Android 上约 7 tok/s。在 Chrome Android v121+ 上通过 WebGPU in browser，依设备不同约 4-8 tok/s。

Throughput variance 不是 porting issue，而是 bandwidth gap × quantization format × NPU 是否可从 user-space 访问。2026 年 Edge inference 是四个不同问题，对应四套不同解法。

## The Concept / 概念

### Bandwidth is the real ceiling / Bandwidth 才是真上限

Decode 每个 token 都读取完整 weights。一个 7B model 的 Q4 weights 是 3.5 GB。以 50 GB/s 读取需要 70 ms，理论上限约 14 tok/s。90 GB/s（高端 mobile DRAM）把上限推到约 25 tok/s。这个数字以下，再多 compute 都没有帮助。

Datacenter HBM3 以 3 TB/s 读取同样 3.5 GB 只需 1.2 ms，上限 830 tok/s。同一模型、同一 weights，不同 memory subsystem。

### Apple Neural Engine (M4 / A18) / Apple Neural Engine（M4 / A18）

- 最高 38 TOPS。Unified memory（CPU 与 ANE 共享同一池）无 copy overhead。
- 通过 Core ML + `.mlmodel` compiled models 访问，或通过 PyTorch 的 Metal Performance Shaders（MPS）。
- Llama.cpp Metal backend 使用 MPS，不直接使用 ANE；原生 ANE 需要 Core ML conversion。
- 2026 年 iOS apps 的最佳实践路径：Core ML with INT4 weights + FP16 activations。

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4) / Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最高 45 TOPS。集成在 SoC 内，与 CPU/GPU 相邻，但 memory domain 分离。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 支持从 PyTorch/ONNX 转换。
- Chat templates、Llama 3.2、Phi-3 都以 first-class artifacts 发布在 AI Hub。

### Intel / AMD NPUs (Lunar Lake, Ryzen AI 300) / Intel / AMD NPUs（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件落后 Apple/Qualcomm；OpenVINO 在改善，但仍偏 niche。
- 适合 Windows ARM copilot apps，以及 AMD/Intel desktops 上的 local-first。

### WebGPU + WebLLM / WebGPU + WebLLM

- 通过 WebGPU compute shaders 在 browser 中运行模型；无需安装。
- M3 Max 上 Llama 3.1 8B Q4 约 41 tok/s，约为同 backend 原生性能的 70-80%。
- WebLLM 17.6k GitHub stars；OpenAI-compatible JS API；Apache 2.0。
- 2026 覆盖：Chrome Android v121+、Safari iOS 26 GA，Firefox Android 仍在追赶。整体移动覆盖约 70-75%。

### NVIDIA Jetson family / NVIDIA Jetson 系列

- Orin Nano Super（8GB）：适合 Llama 3.2 3B、Phi-3，tok/s 表现不错。
- AGX Orin：通过 vLLM 跑 gpt-oss-20b，约 40 tok/s。
- Thor / T4000（JetPack 7.1）：性能为 AGX Orin 的 2x，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 speculative decoding、NVFP4 weights、chunked prefill，把 datacenter optimizations 移植到 edge。

### Quantization choice per target / 每类 target 的 quantization 选择

| Target | Format | Notes |
|--------|--------|-------|
| Apple ANE | INT4 weights + FP16 activations | Core ML conversion path |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub converters |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | Use `mlc_llm convert_weight` + compiled `.wasm`; GGUF is not supported |
| Jetson Orin Nano | Q4 GGUF or TRT-LLM INT4 | Memory-bound |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM path |

### The long-context trap on edge / Edge 上的长上下文陷阱

Llama 3.1 的 128K context 是 datacenter feature。在一台 8 GB RAM 的手机上，4 GB model + 32K tokens 的 2 GB KV cache + OS overhead = OOM。Edge deployments 通常把 context 保持在 4K-8K，除非接受激进 KV quantization（Q4 KV）。

### Voice is the killer app / Voice 是 killer app

Voice agents 对 latency 敏感（first token < 500 ms）。本地推理完全消除 network latency。再结合 speech-to-text（Whisper Turbo variants 可在 edge 上运行），edge inference 就成为 production-quality voice loop。

### Numbers you should remember / 你应该记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 约 41 tok/s。
- AGX Orin：通过 vLLM 跑 gpt-oss-20b 约 40 tok/s。
- Datacenter-edge bandwidth gap：30-50x。
- WebGPU mobile coverage：约 70-75%（Firefox Android lagging）。

## Build It / 动手构建

在 `code/main.py` 里用 bandwidth-bound decode 公式估算各类 edge target 的 tokens/sec ceiling，再与观测 benchmark 对比，判断 runtime 开销。

## Use It / 应用它

`code/main.py` 会根据 bandwidth-bound math，计算各类 edge targets 的 theoretical decode throughput ceilings。它与 observed benchmarks 对比，指出瓶颈是 bandwidth 而不是 compute 的地方。

## Ship It / 交付它

本课产出 `outputs/skill-edge-target-picker.md`。给定 platform（iOS/Android/browser/Jetson）、model 和 latency/memory budget，它会选择 quantization format 和 conversion pipeline。

## Exercises / 练习

1. 运行 `code/main.py`。对 Snapdragon 8 Gen 3（约 77 GB/s bandwidth）上的 7B Q4 model，计算 decode ceiling。与 observed 6-8 tok/s 比较：runtime 是否高效？
2. Android 上 WebGPU 需要 Chrome v121+。为旧浏览器设计 fallback：通过同一个 OpenAI-compatible API 走 server-side。
3. 你的 iOS app 需要 4K-context streaming。哪种 model/format 组合能让 iPhone 16 上 active memory 保持低于 4 GB？
4. Jetson AGX Orin 以 40 tok/s 跑 gpt-oss-20b。Jetson Nano 只能放 3B。如果产品同时面向两者，如何统一 inference stack？
5. 论证 “WebLLM is production-ready in 2026” 是否成立。引用 coverage、performance 和 Firefox Android gap。

## Key Terms / 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ANE | “Apple neural engine” | M-series 和 A-series 中的 on-device NPU；unified memory |
| Hexagon | “Qualcomm NPU” | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | “browser GPU” | W3C-standardized browser GPU API；Chrome/Safari 2026 |
| WebLLM | “browser LLM runtime” | MLC-LLM 项目；Apache 2.0；OpenAI-compatible JS |
| Jetson | “NVIDIA edge” | Orin Nano / AGX / Thor / T4000 family |
| TRT Edge-LLM | “edge TensorRT” | TensorRT-LLM 的 2026 edge port；EAGLE-3 + NVFP4 |
| Unified memory | “shared pool” | CPU 和 NPU 看到同一 RAM；无 copy overhead |
| Bandwidth-bound | “memory limited” | Decode 被读取 weights 的 bytes/sec 限制 |
| Core ML | “Apple conversion” | ANE-native models 的 Apple framework |
| QNN | “Qualcomm stack” | Qualcomm Neural Network SDK |

## Further Reading / 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — landscape 和 benchmarks。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 edge port announcement。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — design 和 benchmarks。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE-native conversion。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon 预转换模型。
