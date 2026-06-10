---
title: "vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models"
method_name: "vla.cpp"
authors: [Khanh D. Nguyen, Hung T. Ho, Chinh T. Nguyen, Thanh Q. Duong, Linh D. Le, Duy M. H. Nguyen, Vien A. Ngo, An T. Le]
year: 2026
venue: arXiv
tags: [vla, inference-runtime, edge-deployment, model-quantization, robot-policy, flow-matching, llama-cpp]
zotero_collection: Robotics/VLA
image_source: mixed
arxiv_html: https://arxiv.org/html/2606.08094
created: 2026-06-10
---

# 论文笔记：vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | VinRobotics (Vietnam); Center for AI Research, VinUniversity; TU Darmstadt / MPRS / Univ. Stuttgart / DFKI (Germany) |
| 日期 | June 2026 |
| 项目主页 | [vla.cpp Project Page](https://fai-modelopt-tech.github.io/vla-cpp.github.io/) |
| 对比基线 | [[llama.cpp]] / [[PyTorch]] reference |
| 链接 | [arXiv](https://arxiv.org/abs/2606.08094) / Code: Coming Soon |

---

## 一句话总结

> vla.cpp 是基于 llama.cpp 扩展的 C++ VLA 推理运行时，统一支持 7 种 VLA 架构，在边缘设备（8GB Jetson Orin）上实现与 PyTorch 相当的精度，并对 BitVLA 达到 4.5× 加速。

---

## 核心贡献

1. **统一推理运行时**: 首个 ggml 级引擎，在[[Flow Matching|流匹配]]与[[Diffusion Policy|扩散]]头的迭代去噪过程中原生支持跨注意力缓存 VLM 前缀（Dynamic Cross-Attention KV Cache）。
2. **Roofline 分析与内存表征**: 系统量化了 VLA 推理的计算/内存瓶颈——VLM prefix 阶段是计算密集型（compute-bound），动作专家阶段是内存密集型（memory-bound），将"容量"与"延迟"两个优化维度分离。
3. **数值一致性框架**: 识别离散操作（如位置嵌入索引选择）在多模态编码器中的精度敏感性，构建编译期"fail-loud"检测门控，防止静默行为漂移。
4. **W2A8 GEMM 优化核**: 为三元量化设计 IMMA 张量核 W2A8 GEMM，使 BitVLA 在 RTX 3060 上获得 4.5× 加速（172.8ms → 37.85ms）。

---

## 问题背景

### 要解决的问题

现代 [[VLA|视觉-语言-动作模型]] 依赖工作站级 GPU（通常需要 16GB+ 显存），而机器人板载硬件（如 NVIDIA Jetson Orin，8GB 内存）远无法满足。如何将 VLA 策略部署到资源受限的边缘设备上，同时保持与 PyTorch 参考实现相同的行为表现？

### 现有方法的局限

- **TensorRT-LLM / ONNX Runtime / MLC-LLM**: 不支持[[Flow Matching|流匹配]]或[[Diffusion Policy|扩散]]动作头，不支持跨步骤的动态 cross-attention KV 缓存，且无法跨平台免重编译部署。
- **llama.cpp AR-VLA**: 仅支持自回归 VLA 变体，无法处理迭代去噪架构。
- 所有现有方案均无法从单一二进制文件跨 RTX / AGX Orin / Orin Nano 平台运行，缺乏可移植的 bundle 格式。

### 本文的动机

VLA 推理与纯 LLM 推理存在四个关键差异，需要针对性工程处理：(1) 需要暴露完整隐藏状态（而非仅 logits）；(2) 需要对前缀使用双向注意力掩码；(3) 需要在多个去噪步骤间维护 cross-attention KV 缓存生命周期；(4) 需要使用模型统计信息进行动作去归一化。基于 [[llama.cpp]] 的 [[ggml]] 后端，可以用少量工程改动填补这四个缺口，同时继承其跨平台、内存高效的优势。

---

## 方法详解

### 系统架构

![[vlacpp_fig1_architecture.png]]

**说明**: vla.cpp 整体架构。C++ 服务端加载 GGUF bundle（含权重、tokenizer、归一化统计）并执行 VLA 推理；Python 客户端通过 [[ZeroMQ]] + [[Protobuf]] 进行观测编码与闭环控制。

vla.cpp 采用 **C++ 服务器 + Python 客户端** 解耦架构：

- **输入**: 相机图像 $I$ + 自然语言指令 $\ell$ + 机器人状态 $s$
- **Backbone**: 根据架构选择（SigLIP / InternViT / Qwen3-VL ViT + SmolLM2 / Gemma-2B / Qwen3 等）
- **核心模块**: [[VLM Prefix|VLM 前缀编码]]（双向注意力）+ [[Flow Matching|流匹配]]/[[Diffusion Policy|扩散]]动作头（cross-attention 至缓存前缀）
- **输出**: [[Action Chunking|动作块]] $\tilde{a}$（去归一化后的动作序列）
- **模型格式**: 单一 GGUF 文件（权重 + tokenizer + 归一化统计量）

### 核心模块

#### 模块1: 两阶段 VLA 推理模式（Two-Stage Forward）

**设计动机**: VLA 推理由性质截然不同的两部分组成，需分别优化：

- **Prefix 阶段（计算密集型）**: VLM 对数百个图像/指令 token 进行双向[[Self-Attention|自注意力]]编码，运算量大但 batch 内可并行。
- **Action 阶段（内存密集型）**: 动作专家对少量 token 执行 $T$ 步去噪，每步需 cross-attend 到缓存的前缀隐藏状态 $h$，内存带宽是瓶颈。

**具体实现**:
- 通过 [[ggml]] 图结构暴露完整隐藏状态（绕过 llama.cpp 仅输出 logits 的限制）
- 在 prefix 计算后缓存 cross-attention KV，跨 $T$ 个去噪步骤复用
- 使用 build-time 编译门控选择 FP32/FP16 精度用于位置嵌入索引计算

#### 模块2: 精度感知索引计算（Precision-Aware Index Gating）

**设计动机**: 视觉塔的位置嵌入索引是离散整数，FP32 与 FP16 的截断方式不同，会选中错误的嵌入行。

**具体实现**:
- 在 SmolVLA 中，FP16 精度导致动作块在机器人空间的最大偏移达 $|\Delta|_{\max} \approx 1.97$，导致抓取失败
- [[Flow Matching|流匹配]]会在 $T$ 步中放大此误差
- 解决方案：编译期检测精度类型，对索引计算强制使用 FP32，其余部分保持 FP16

#### 模块3: W2A8 IMMA 张量核 GEMM

**设计动机**: [[BitVLA]] 使用三元权重（$\{-1,0,1\}$），标准 FP16 GEMM 无法充分利用其低比特优势。

**具体实现**:
- 为 NVIDIA IMMA（Integer Matrix Multiply Accumulate）张量核定制 W2A8 GEMM
- 无需修改权重格式，直接在 ggml 后端注册自定义 kernel
- RTX 3060 上 BitVLA per-step 延迟：172.8ms → 37.85ms（**4.5× 加速**）
- AGX Orin 上：406.6ms → 101.11ms（**4.0× 加速**）

### 支持的七种架构（Table 5）

| 模型 | 视觉主干 | 语言主干 | 动作头 | 参数量 | 去噪步数 $T$ |
|------|---------|---------|-------|--------|------------|
| SmolVLA | SigLIP-So400m | SmolLM2-360M | FM cross-attn expert | 450M | 10 |
| Evo-1 | InternViT-300M | Qwen2.5-0.5B | cross-attn DiT | 770M | 32 |
| BitVLA | BitSigLIP-L | BitNet-2B | MLP regression | 2.4B | — |
| π₀ | SigLIP-So400m | Gemma-2B | FM joint-attn expert | 3B | 10 |
| GR00T-N1.5 | SigLIP2-400M | Qwen3-1.7B | AltVL DiT++self-attn | 3B | 4 |
| GR00T-N1.6 | SigLIP2-400M | Qwen3-1.7B | AltVL DiT | 3B | 4 |
| GR00T-N1.7 | Qwen3-VL ViT | Qwen3-VL | AltVL DiT++self-attn | 3B | 4 |

---

## 关键公式

### 公式1: [[VLA Inference|标准 VLA 推理算法]]

$$
\begin{aligned}
&h \leftarrow \text{VLM-Prefix}(I, \ell, s) \quad \text{[双向注意力，计算并缓存 cross-attn KV]} \\
&a_0 \sim \mathcal{N}(0, I) \\
&\text{for } t = 1, \ldots, T: \\
&\quad a_t \leftarrow \text{ActionHead}(a_{t-1},\, h,\, t) \quad \text{[cross-attend 至缓存的 } h \text{]} \\
&\tilde{a} \leftarrow \text{Denormalize}(a_T) \\
&\text{return } \tilde{a}
\end{aligned}
$$

**含义**: 两阶段 VLA 推理范式——先用 VLM 编码多模态前缀，再用动作头迭代去噪得到动作块。

**符号说明**:
- $I$: 相机图像输入
- $\ell$: 自然语言指令
- $s$: 机器人当前状态
- $h$: VLM 前缀隐藏状态（被缓存以供动作头 cross-attend）
- $a_0 \sim \mathcal{N}(0, I)$: 从标准高斯噪声初始化动作
- $T$: 去噪步骤数（依架构不同：4、10、32）
- $\tilde{a}$: 去归一化后输出的[[Action Chunking|动作块]]

### 公式2: [[Roofline Model|Roofline 算术强度]]

$$
I = \frac{\text{FLOPs}}{\text{Bytes}}
$$

**含义**: 操作的算术强度，决定计算阶段位于 roofline 的计算密集区（$I > I_{\text{ridge}}$）还是内存密集区（$I < I_{\text{ridge}}$）。

**符号说明**:
- $I$: 算术强度（FLOPs/Byte）
- $I_{\text{ridge}}$: 屋脊点强度（峰值算力 / 内存带宽）
- VLM prefix 阶段: $I \gg I_{\text{ridge}}$（计算密集型）
- 动作专家阶段: $I < I_{\text{ridge}}$（内存密集型）

### 公式3: [[Position Embedding|位置嵌入精度误差]]

$$
|\Delta|_{\max} \approx 1.97 \quad \text{（FP32 vs FP16 索引截断导致的动作空间最大偏移）}
$$

**含义**: FP16 精度下视觉位置嵌入索引选错行，造成动作块在机器人空间产生近 2 个单位的最大偏移，经过[[Flow Matching|流匹配]]的 $T$ 步放大后会导致抓取失败。

---

## 关键图表

### Figure 1: 系统架构图

![[vlacpp_fig1_architecture.png]]

**说明**: vla.cpp 架构全图。左侧为 C++ 服务端（单次加载 GGUF，执行 VLA 推理），右侧为 Python 客户端（观测编码、闭环控制逻辑），两者通过 ZeroMQ + Protobuf 通信。

### Figure 2: Roofline 分析图

![[vlacpp_fig2_roofline.png]]

**说明**: RTX 3060 和 AGX Orin 上的 [[Roofline Model|Roofline]] 分析。VLM prefix 阶段（计算密集，点在带宽线以上），动作专家阶段（内存密集，点在带宽线以下）。三元量化降低 BitVLA 内存占用但不改善延迟，W2A8 GEMM 可同时降低两者。

### Figure 3 (Appendix): Replan 间隔研究

*(图片暂不可用，见论文 Appendix G)*

**说明**: SmolVLA 在不同 replan 间隔（1–50 步）下的成功率。1–32 步时成功率稳定（80–98%），全开环（50 步）时骤降至 38%，说明闭环控制对 VLA 策略至关重要。

### Table 1: 推理栈能力对比

| 推理栈 | Portable Bundle | Dynamic Cross-Attn KV | Flow/Diffusion Head | 官方 Orin 支持 | 跨层无需重编译 |
|-------|:-:|:-:|:-:|:-:|:-:|
| TensorRT-LLM | ✗ | ∘ | ✗ | ∘ | ✗ |
| TensorRT Edge-LLM | ✗ | ∘ | ∘ | ∘ | ✗ |
| ONNX Runtime / Torch-TensorRT | ✗ | ✗ | ✗ | ∘ | ✗ |
| MLC-LLM | ∘ | ✗ | ✗ | ∘ | ✗ |
| llama.cpp AR-VLA | ✓ | ✗ | ✗ | ✓ | ✓ |
| **vla.cpp（ours）** | **✓** | **✓** | **✓** | **✓** | **✓** |

**说明**: vla.cpp 是唯一同时满足五项关键能力的方案，现有竞品均存在不同程度的功能缺失。

### Table 2: RTX 3060 行为保真度结果

| 模型 | 主干 | 去噪步数 $n_a$ | 成功率 (%) | 单步延迟 (ms) | 推理延迟 (ms) | VRAM (MiB) |
|------|-----|--------------|-----------|-------------|-------------|-----------|
| SmolVLA | SmolVLM2 | 4 | 90.5 [86, 94] | 28.16 | 54.8 | 1410 |
| π₀ | PaliGemma | 32 | 87.5 [82, 91] | 9.74 | 207.2 | 5548 |
| BitVLA | BitNet–SigLIP | 8 | **100.0** [98, 100] | 37.85 | 235.9 | 1312 |
| Evo-1 | InternVL3 | 8 | 94.5 [90, 97] | 63.60 | 131.0 | 1564 |
| GR00T-N1.5 | Eagle/Cosmos | 16 | 96.0 [92, 98] | 14.17 | 147.0 | 4866 |
| GR00T-N1.6 | Eagle/Cosmos | 16 | 86.5 [81, 91] | 10.29 | 83.6 | 6048 |
| GR00T-N1.7 | Eagle/Cosmos | 16 | **98.0** [95, 99] | 10.26 | 84.1 | 6302 |

**说明**: vla.cpp 跨全部 7 种架构达到与 PyTorch 参考实现相当的成功率，BitVLA 达到 100% 完美复现，且仅占用 1.3 GiB 内存。

### Table 3: 跨硬件层级延迟与内存

| 模型 | RTX 3060 (ms) | AGX Orin (ms) | Orin Nano (ms) | Nano Peak RSS (MiB) |
|------|-------------|--------------|---------------|-------------------|
| SmolVLA | 28.16 | 65.41 | 141.81 | 2031 |
| BitVLA | 37.85 | 101.11 | 355.65 | 2199 |
| Evo-1 | 63.60 | 131.01 | 458.84 | 2135 |
| GR00T-N1.5 | 14.17 | 28.78 | 84.76† | 5975 |
| π₀ | 9.74 | 27.90 | 39.10† | 6068 |
| GR00T-N1.6 | 10.29 | 26.70 | * | — |
| GR00T-N1.7 | 10.26 | 26.84 | * | — |

†需 simulator offload；*超出 8GB 内存池，无法加载

**说明**: 向 Nano 方向延迟增加 3–10×；GR00T-N1.6/N1.7 驻留权重超 6GiB，无法在 Orin Nano 上运行。

### Table 4: ALOHA 真实机器人闭环测试

| 任务 | 引擎 | 成功率 | 任务耗时 | 推理/块 |
|------|-----|-------|---------|--------|
| Task 1: 积木→盒子 | PyTorch | 3/20 (15%) | 71 s | ~620 ms |
| Task 1: 积木→盒子 | vla.cpp | **18/20 (90%)** | **49 s** | ~470 ms |
| Task 2: 垃圾+香蕉 | PyTorch | 13/20 (65%) | 30 s | ~620 ms |
| Task 2: 垃圾+香蕉 | vla.cpp | **17/20 (85%)** | **28 s** | ~470 ms |

**说明**: vla.cpp 凭借约 150ms 更新鲜的观测（推理延迟 470ms vs 620ms），在长视野任务中获得巨大优势（Task 1 成功率从 15% 提升至 90%）。低延迟在多块动作序列中的累积效应是关键。

### Table 6: LIBERO 套件完整结果（95% Wilson 置信区间）

| 套件 | BitVLA vla.cpp | BitVLA ref | GR00T-N1.7 vla.cpp | GR00T-N1.7 ref |
|-----|--------------|------------|-------------------|---------------|
| Spatial | 94.5 [90.4, 96.9] | 93.6 [91.1, 95.4] | 96.0 [92.3, 98.0] | 97.3 [95.7, 98.4] |
| Object | **100.0** [98.1, 100.0] | 99.6 [98.6, 99.9] | 97.0 [93.6, 98.6] | 99.8 [99.1, 100.0] |
| Goal | 92.5 [88.0, 95.4] | 91.6 [88.8, 93.7] | 97.5 [94.3, 98.9] | 98.0 [96.5, 98.9] |
| Long | 83.5 [77.7, 88.0] | 85.8 [82.5, 88.6] | 89.0 [83.9, 92.6] | 91.7 [89.2, 93.6] |

**说明**: vla.cpp 在全部 4 个 LIBERO 子套件上与 PyTorch 参考实现统计上等价（置信区间高度重叠），验证了行为保真度。

---

## 实验

### 数据集 / Benchmark

| Benchmark | 规模 | 特点 | 用途 |
|-----------|------|------|------|
| [[LIBERO]]-Object | 200 episodes/架构 | 操作任务，4 个子套件 | 主要行为保真度评测 |
| [[SimplerEnv]] WidowX | — | 真实机器人仿真 | 补充验证 |
| ALOHA 真实机器人 | 20 trials/任务 | 双臂操作，积木+物体 | 真实机器人压力测试 |

### 实现细节

- **后端**: [[ggml]]（llama.cpp 的 C++ 张量计算库）
- **通信**: [[ZeroMQ]] + [[Protobuf]]
- **模型格式**: GGUF（权重 + tokenizer + 归一化统计量单文件打包）
- **硬件**: RTX 3060（12GB）、NVIDIA AGX Orin（32GB）、Jetson Orin Nano（8GB）
- **关键优化**: W2A8 IMMA GEMM（BitVLA 专用）、FP32 索引精度门控

### 引擎加速（SmolVLA 基线）

| 模式 | 单步延迟 (ms) | 峰值 VRAM |
|------|------------|---------|
| PyTorch eager | 223.96 | 同 vla.cpp |
| vla.cpp | 28.16 | 同上 |
| **加速比** | **7.95×** | — |

---

## 批判性思考

### 优点

1. **完整工程交付**: 从问题定义到系统实现、基准测试、真实机器人验证，链条完整，工程价值高。
2. **Roofline 洞察深刻**: 明确区分 prefix（计算密集）与 action expert（内存密集），为未来 VLA 优化提供了清晰的方向框架。
3. **数值保真度严格**: 识别到位置嵌入精度陷阱并给出系统性解决方案，防止了在实际部署中难以发现的静默错误。
4. **覆盖面广**: 7 种主流 VLA 架构，4 种动作头类型，3 个硬件层级，结果可信度高。

### 局限性

1. **无批处理加速**: 论文仅测单请求延迟，未探索批处理（batching）对吞吐量的提升，实际部署场景可能受限。
2. **代码尚未开源**: 论文提交时代码 "coming soon"，可复现性暂无法验证。
3. **量化策略有限**: 除 BitVLA 外，其他架构未展示量化收益，FP16/INT8 量化路径不明确。
4. **仅单摄像头/单任务场景**: ALOHA 测试仅 2 个任务，实际部署中多摄像头、长视野复杂任务的表现待验证。

### 潜在改进方向

1. **动态批处理**: 利用 prefix 阶段的并行性实现多请求批处理，提升服务器吞吐量。
2. **量化扩展**: 将 W2A8 思路推广到非三元架构（INT4/INT8），进一步降低内存占用。
3. **在线 KV Cache 压缩**: 对 prefix 隐藏状态做稀疏化或低秩压缩，缩小 cross-attention 内存瓶颈。

### 可复现性评估

- [ ] 代码开源（声明 coming soon）
- [ ] 预训练模型（声明 coming soon）
- [x] 训练细节完整（benchmark scaffold 已提供框架）
- [x] 数据集可获取（LIBERO / SimplerEnv 均公开）

---

## 关联笔记

### 基于

- [[llama.cpp]]: 底层 ggml C++ 推理引擎，vla.cpp 在其上扩展
- [[ggml]]: llama.cpp 使用的 C 张量计算库
- [[Flow Matching]]: 主要支持的动作头范式之一

### 对比

- [[SmolVLA]]: 450M 轻量 VLA，作为主要验证架构之一
- [[pi0|π₀]]: PaliGemma backbone VLA，3B 参数
- [[BitVLA]]: 三元量化 VLA，vla.cpp 针对其实现 W2A8 GEMM
- [[GR00T]]: NVIDIA GR00T-N1.x 系列，多版本支持

### 方法相关

- [[ZeroMQ]]: 服务端与客户端通信中间件
- [[Protobuf]]: 序列化协议
- [[GGUF]]: 模型 bundle 格式
- [[Action Chunking]]: VLA 输出的动作块形式
- [[Roofline Model]]: 系统性能分析模型

### 硬件/数据相关

- [[LIBERO]]: 主要评测 benchmark
- [[ALOHA]]: 真实机器人实验平台（双臂操作）
- [[Jetson Orin]]: 目标部署边缘设备

---

## 速查卡片

> [!summary] vla.cpp: A Unified Inference Runtime for VLA Models
> - **核心**: 基于 llama.cpp 的 C++ VLA 推理运行时，统一支持 7 种架构，面向 8GB Jetson Orin 边缘部署
> - **方法**: 两阶段推理（prefix 计算密集 + action expert 内存密集）+ cross-attn KV 缓存 + W2A8 IMMA GEMM + 精度感知索引门控
> - **结果**: BitVLA 100% 成功率（1.3 GiB 内存）；SmolVLA 7.95× 加速；ALOHA Task 1 成功率从 15% → 90%
> - **代码**: Coming Soon（[项目主页](https://fai-modelopt-tech.github.io/vla-cpp.github.io/)）

---

*笔记创建时间: 2026-06-10*
