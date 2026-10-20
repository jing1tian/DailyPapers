---
title: "OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism"
method_name: "OxyGen"
authors: [Xiangyu Li, Huaizhi Tang, Xin Ding, Weijun Wang, Ting Cao, Yunxin Liu]
year: 2026
venue: NeurIPS 2026
tags: [vla-inference, kv-cache, multi-task-parallelism, system-optimization, embodied-ai, mixture-of-transformers, robotics]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2603.14371v5
created: 2026-10-20
---

# 论文笔记：OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | AIR & Dept. of Electronic Engineering, Tsinghua University; USTC |
| 日期 | March 2026 |
| 项目主页 | [air-embodied-brain.github.io/OxyGen](https://air-embodied-brain.github.io/OxyGen) |
| 对比基线 | [[π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.14371) / [Code](https://github.com/air-embodied-brain/OxyGen) |

---

## 一句话总结

> OxyGen 通过统一的 [[KV Cache]] 管理器，在 [[Mixture-of-Transformers]] VLA 推理中实现跨任务 KV 共享与跨帧连续批处理，在 RTX 4090 上同时达到 70 Hz 动作频率和 200+ tokens/s 语言吞吐，最高加速 3.7×。

---

## 核心贡献

1. **问题定位**：首次将"孤立 KV Cache 管理"识别为 [[Mixture-of-Transformers|MoT]] VLA 多任务并行推理中计算浪费与资源竞争的根本原因。
2. **跨任务 KV 共享（Cross-Task KV Sharing）**：一次 [[Prefill]] 生成的 [[KV Cache]] $\mathcal{K}_t$ 同时供动作专家和语言专家使用，消除重复 prefill 开销。
3. **跨帧连续批处理（Cross-Frame Continuous Batching）**：将来自不同帧的活跃语言请求打包成一个 batch，在每帧的 per-frame 解码预算内推进 $k$ 步，显著提升语言吞吐。

---

## 问题背景

### 要解决的问题

具身 AI 智能体需要从同一观测出发，**同时**执行操控动作生成、对话语言生成和记忆构建等异构任务，且各任务具有不同的时间约束（动作任务有严格的实时截止期，语言任务跨多帧延伸）。如何在有限的计算资源（边缘 GPU/Jetson）上高效服务这些任务是核心挑战。

### 现有方法的局限

现有 VLA 推理系统对每个任务**孤立管理** [[KV Cache]]：
- **冗余 Prefill**：动作专家和语言专家分别对相同的观测做一遍 prefill，重复计算 $L$ 层的 Key/Value。
- **资源竞争**：不同任务争夺 GPU 内存和计算核心，导致动作频率下降或语言请求积压。
- **批处理缺失**：跨帧的语言请求无法被批处理，GPU 利用率低。

### 本文的动机

[[Mixture-of-Transformers]] VLA 的共享 VLM backbone 天然产生了对同一观测的多份 KV 计算。若将 [[KV Cache]] 作为**跨任务共享的一等公民（first-class shared resource）**，则可以：
1. 只做一次 prefill，将 $\mathcal{K}_t$ 分叉给所有专家；
2. 将跨帧积压的语言请求打包，在动作帧间隙中统一推进解码。

---

## 方法详解

### 系统架构概览

OxyGen 围绕一个**统一 KV Cache 管理器**（Unified KV Cache Manager $\mathcal{M}$）构建，支持 [[Mixture-of-Transformers|MoT]] VLA 在多任务并行下的推理：

- **输入**：当前帧观测 $o_t$（图像 + 状态 + 语言指令）
- **Backbone**：共享 VLM（如 Qwen3VL）+ 动作专家 + 语言专家
- **核心机制**：[[KV Cache]] 跨任务共享 + 跨帧 [[Continuous Batching|连续批处理]]
- **输出**：每帧机器人动作 $A_t$（高频）+ 跨帧语言 token $y_t$（低频）

### 核心模块

#### 模块 1：统一 KV Cache 管理器（$\mathcal{M}$）

**设计动机**：用四个原语统一管理所有在途请求的生成状态，实现跨任务和跨帧的状态共享与恢复。

**管理器接口**：

| 操作 | 语义 |
|------|------|
| **Store**$(r, \sigma)$ | 保存请求 $r$ 的生成状态 $\sigma_t$ |
| **Retrieve**$(r)$ | 取回请求 $r$ 的状态，恢复解码 |
| **Update**$(r, \sigma')$ | 推进 $k$ 步解码后更新状态 |
| **Remove**$(r)$ | 请求完成后释放状态 |

其中**可恢复生成状态（Resumable Generation State）**定义为：

$$\sigma_t = (\mathcal{K}_t,\; y_t,\; \delta_t)$$

- $\mathcal{K}_t$：当前帧的 KV Cache（含 prefill KV + 解码时累积的 KV）
- $y_t$：已生成 token 序列
- $\delta_t$：终止标志（EOS 是否出现）

#### 模块 2：跨任务 KV 共享（Cross-Task KV Sharing）

**设计动机**：动作专家和语言专家共享同一 VLM backbone，对 $o_t$ 做的 prefill 结果完全相同，无需重复计算。

**具体实现**：
- [[Prefill]] 一次：$\{(h_{t,l}, K_{t,l}, V_{t,l})\}_{l=1}^{L} = \Theta_{VLM}(o_t)$
- 将 $\mathcal{K}_t = \{(K_{t,l}, V_{t,l})\}_{l=1}^{L}$ 存入管理器
- 动作专家读 $\mathcal{K}_t$ → 去噪采样 $A_t$；语言专家读 $\mathcal{K}_t$ → 自回归解码 $y_t$
- 节省约 **1.4×** 帧延迟（ablation 实测）

#### 模块 3：跨帧连续批处理（Cross-Frame Continuous Batching）

**设计动机**：语言任务跨多帧才能完成（$N \gg k$），每帧推进 $k$ 步，多帧请求可以打包成一个 batch 并行解码。

**具体实现**：
- 每帧预算：$k$ 步解码（超参数，控制动作频率与语言吞吐的 tradeoff）
- 跨帧批状态：$\hat{\sigma} = (\hat{\mathcal{K}},\; \hat{y},\; \hat{\delta})$，聚合 $B$ 个活跃请求
- 每帧解码后：完成的请求调用 Remove，未完成的调用 Update 持久化

#### 每帧执行流（Per-Frame Flow）

1. **Prefill**：对 $o_t$ 做一次 prefill，KV 存入管理器
2. **Fan-out**：将 $\mathcal{K}_t$ 分叉给动作请求（新建）和语言请求（新建，Store）
3. **Decode**：将所有活跃语言请求合并成 batch $\hat{\sigma}$，推进 $k$ 步（Update）
4. **Action**：并行/后续执行动作去噪 $\text{Denoise}^{(S)}(\varepsilon, \mathcal{K}_t)$
5. **Evict**：Remove 已完成请求（$\delta_t = 1$），持久化未完成请求

---

## 关键公式

### 公式 1：[[Prefill|VLM Prefill]]

$$
\{(h_{t,l},\, K_{t,l},\, V_{t,l})\}_{l=1}^{L} = \Theta_{VLM}(o_t)
$$

$$
\mathcal{K}_t = \{(K_{t,l},\, V_{t,l})\}_{l=1}^{L}
$$

**含义**：将观测 $o_t$ 输入共享 VLM，得到 $L$ 层 Key/Value 矩阵组成的 KV Cache $\mathcal{K}_t$。

**符号说明**：
- $\Theta_{VLM}$：共享 VLM backbone 参数
- $o_t$：第 $t$ 帧的多模态观测（图像、状态、语言指令）
- $L$：Transformer 层数
- $h_{t,l}$：第 $l$ 层的隐状态
- $K_{t,l}, V_{t,l}$：第 $l$ 层的 Key/Value 矩阵

### 公式 2：[[Mixture-of-Transformers|专家条件分布]]

$$
p(A_t \mid \mathcal{K}_t) \quad \text{（动作专家）}, \qquad p(y_t \mid \mathcal{K}_t) \quad \text{（语言专家）}
$$

**含义**：两类专家均以相同的 $\mathcal{K}_t$ 为条件，因此只需做一次 prefill 即可同时为两者服务。

### 公式 3：[[Multi-Task Parallelism|优化目标]]

$$
\text{maximize} \quad \tau = \frac{Bk}{T} \quad \text{s.t.} \quad f = \frac{H}{T} \geq f_{\min}
$$

**含义**：在保证动作频率 $f$ 不低于最低要求 $f_{\min}$ 的约束下，最大化语言吞吐 $\tau$（tokens/s）。

**符号说明**：
- $B$：当前帧的活跃语言请求数（batch size）
- $k$：每帧允许的解码步数（每帧解码预算）
- $T$：每帧总时间（ms）
- $H$：控制帧率分子（$H=1000$ 时 $f$ 单位 Hz）
- $f_{\min}$：最低动作频率要求

### 公式 4：[[Resumable Generation State|可恢复生成状态]]

$$
\sigma_t = (\mathcal{K}_t,\; y_t,\; \delta_t)
$$

**含义**：将一个语言请求的完整解码状态封装为三元组，支持跨帧持久化和恢复。

**符号说明**：
- $\mathcal{K}_t$：prefill KV + 解码累积 KV
- $y_t$：已生成 token 序列 $y_{t,1:j}$
- $\delta_t \in \{0,1\}$：终止标志，$\delta_t=1$ 表示生成已结束

### 公式 5：[[Autoregressive Generation|自回归语言似然]]

$$
p(y_t \mid \mathcal{K}_t) = \prod_{j=1}^{N} p\!\left(y_{t,j} \mid y_{t,1:j-1},\, \mathcal{K}_t\right)
$$

**含义**：语言专家以 $\mathcal{K}_t$ 为 context，逐 token 自回归解码，共生成 $N$ 个 token。

### 公式 6：[[Denoising Diffusion|动作去噪采样]]

$$
A_t = \text{Denoise}^{(S)}(\varepsilon,\, \mathcal{K}_t), \quad \varepsilon \sim \mathcal{N}(0,\, I)
$$

**含义**：动作专家以 $\mathcal{K}_t$ 为条件，对高斯噪声做 $S$ 步去噪，生成当前帧的动作序列 $A_t$。

**符号说明**：
- $S$：去噪步数（超参数，影响动作质量与帧延迟）
- $\varepsilon$：初始噪声
- $A_t$：当前帧输出动作块

### 公式 7：[[Continuous Batching|跨帧批状态]]

$$
\hat{\sigma} = (\hat{\mathcal{K}},\; \hat{y},\; \hat{\delta})
$$

**含义**：将 $B$ 个活跃请求的状态拼接为批次状态，在 GPU 上做一次批推理，推进所有请求各 $k$ 步。

---

## 关键图表

### Figure 1：系统概览与动机

![Figure 1: OxyGen Teaser](https://arxiv.org/html/2603.14371v5/teaser.png)

**说明**：左侧展示 [[Mixture-of-Transformers|MoT]] VLA 同时产生高频机器人动作和跨帧语言记忆的多任务并行场景；右侧对比孤立执行（两次 prefill，无跨帧批处理）与 OxyGen 统一 KV 管理（一次 prefill，跨帧 batch decode），最高加速 **3.7×**。

### Figure 2：KV-centric 数据流

![Figure 2: KV Dataflow](https://arxiv.org/html/2603.14371v5/dataflow.png)

**说明**：第 $t$ 帧在统一管理器下的完整数据流。一次 [[Prefill]] 生成 $\mathcal{K}_t$，通过 Store/Fan-out 供动作专家（Denoise）和语言专家（Decode）共享；解码时新增的 token KV 增量地追加到 $\mathcal{K}_t$；解码完毕后 Remove 释放内存。

### Figure 3：时间线对比（Timeline）

**说明**（无图 URL，见论文原图）：以 $N=12, B=3, k=4$ 为例，孤立执行每帧需要 $T+\Delta T$（额外 prefill 开销），OxyGen 将帧延迟降至 $T$，每个语言请求跨三帧完成。直观展示跨帧批处理将 GPU 利用率从碎片化变为连续填充。

### Figure 4：动作频率 vs. 语言吞吐 Tradeoff

**说明**（无图 URL，见论文原图）：在 LIBERO 上扫描 $k \in \{1, 5, 10\}$ 和 $N \in \{5, 10, 15, 20, 30\}$，展示动作频率与语言吞吐的 Pareto 前沿。增大 $k$ 和 $N$ 均可提升吞吐，但较大 $k$ 会挤占动作帧时间。

### Figure 5：Speedup Heatmap（LIBERO / ALOHA / DROID）

![Figure 5: Speedup Heatmap](https://arxiv.org/html/2603.14371v5/e2e_speedup_heatmap_vs_steps_per_frame.png)

**说明**：在 LIBERO、ALOHA、DROID 三个 benchmark 上，对去噪步数 $S$ 和每帧解码步数 $k$ 做二维扫描（$N=30$）的动作频率 Speedup 热力图。$S$ 越小（去噪越少）、$k$ 越小（保留更多时间给动作）时 Speedup 越高，最高达 **3.7×**。

### Figure 6：消融实验（动作频率）

**说明**（无图 URL，见论文原图）：在 RTX 4090 和 Jetson AGX Thor 上，对比五组配置在 LIBERO（5 steps/frame）上的动作频率：
- **Baseline**：孤立执行（两次 prefill）
- **Parallel**：并行化但不共享 KV
- **KV Sharing Only**：仅跨任务 KV 共享（~1.4×）
- **OxyGen (Full)**：KV 共享 + 跨帧批处理（~2.7× 额外提升）
- **Single-Frame Decode (Upper Bound)**：单帧内完成所有解码的理论上界

### Figure 7：不同负载模式下的性能

**说明**（无图 URL，见论文原图）：在 Uniform、Poisson 和混合长度请求三种到达模式下，OxyGen 的动作频率和平均语言 batch size 均保持稳定，说明跨帧连续批处理对 stochastic workload 具有鲁棒性。

### Figure 8：人形机器人部署（Jetson AGX Thor）

![Figure 8: Humanoid Robot](https://arxiv.org/html/2603.14371v5/figures/robot_jetson.jpg)

**说明**：搭载 Jetson AGX Thor 的人形机器人（左）和各阶段延迟测量（右）。OxyGen 将语言路径延迟从 **822.3 ms → 195.4 ms**，端到端总延迟从 1030 ms → **393 ms**（~2.6×）。

### Table 1：跨骨干网络与专家数的迁移性（RTX 4090，$N=30$）

| 模型 | 专家数 $E$ | 基线延迟（ms） | Speedup ($k=1$) | Speedup ($k=5$) |
|------|-----------|--------------|----------------|----------------|
| Xiaomi-Robotics-0 | 2 | 562.8 | 5.83× | 3.23× |
| Qwen3VL-PI_v3 (StarVLA) | 2 | 478.0 | 6.64× | 3.60× |
| + GR00T | 3 | 515.8 | 6.05× | 3.53× |
| + GR00T + OFT | 4 | 541.2 | 6.30× | 3.69× |

**关键发现**：OxyGen 在不同骨干（Qwen3VL、Xiaomi-Robotics-0）和不同专家数（2-4）上均有效，加速比随专家数增加而略有变化。

### Table 2：动作质量验证（LIBERO 成功率，%）

| 方法 | 空间 | 物体 | 目标 | LIBERO-10 | 均值（95% CI） |
|------|------|------|------|-----------|----------------|
| openpi（参考） | — | — | — | — | 96.85 |
| **OxyGen** | 98.42 | 98.52 | 97.24 | 92.86 | **96.76 (96.55–96.97)** |

**关键发现**：OxyGen 的动作质量（96.76%）与无系统优化的基线（96.85%）几乎持平，系统优化不损失任务成功率。

### Table 3：内存与能耗对比（RTX 4090）

| 配置 | 峰值内存（GiB） | 能耗/请求（J） | 相对基线能耗变化 |
|------|----------------|--------------|----------------|
| Baseline | 6.43 | 117.4 | — |
| Parallel（孤立并行） | 12.49 | 120.9 | +3.0% |
| OxyGen w/o Batching | — | 97.8 | −16.7% |
| **OxyGen (batch=2)** | — | 39.1 | **−66.7%** |
| **OxyGen (batch=4)** | — | 25.8 | **−78.0%** |

**关键发现**：跨帧批处理大幅降低能耗，batch=4 时能耗降低 78%，显著优于孤立并行方案。

### Table 4：跨模型配置加速比（RTX 4090）

**关键发现**：StarVLA（$N=30, k=1$）获最高 6.64× 加速；当 $N=k$ 时（如 $k=10, N=10$）无法跨帧批处理，略低于基线（0.96×），符合预期。

### Table 5：不同模型规模的加速比（Qwen3.5，0.8B–9B，$N=30, k=5$）

| 模型规模 | 基线延迟（ms） | Speedup |
|---------|--------------|---------|
| 0.8B | 429.0 | 3.58× |
| 3B | — | ~3.6× |
| 7B | — | ~3.7× |
| 9B | 731.9 | 3.79× |

**关键发现**：加速比随模型规模增大略有提升（3.58×→3.79×），说明 OxyGen 对大模型同样有效。

### Table 6：延迟分解（$S=10$）

| 阶段 | 延迟（ms） | OxyGen 贡献分析 |
|------|-----------|----------------|
| Prefill | ~44 | 从 2× 降至 1×（跨任务共享核心收益） |
| 去噪（每步）| ~20.4 | 与基线相同 |
| 无批处理节省 | 36.4%（N=1）→ 9.4%（N=30） | N 大时批处理贡献更重要 |

### Table 7：KV 管理器开销（%占帧时间）

| batch size | JAX 开销 | PyTorch 开销 |
|-----------|---------|-------------|
| 6 | 3.40% | 1.24% |
| 30 | 17.48% | 6.63% |

**关键发现**：Cache splitting 操作是开销主要来源；PyTorch 实现比 JAX 开销低约 2.7×；batch=30 时 JAX 开销较高但 PyTorch 仍可接受（<7%）。

---

## 实验

### 数据集 / Benchmarks

| 数据集 | 特点 | 用途 |
|--------|------|------|
| LIBERO | 4 suite，桌面操控 | 主要 E2E 评测 |
| ALOHA | 双臂操控 | 泛化评测 |
| DROID | 大规模真实场景 | 泛化评测 |
| Textual Memory（自建）| Zotero + 人工标注 | 记忆构建应用 |

### 实现细节

- **基础模型**：[[π0.5]]（Qwen3VL-PI + 动作专家）；另评测 Xiaomi-Robotics-0、StarVLA（Qwen3VL-PI_v3 + GR00T）
- **推理框架**：基于 JAX（openpi 原生）；Table 7 中 PyTorch 对比
- **硬件**：NVIDIA RTX 4090（云端/PC）；NVIDIA Jetson AGX Thor（边缘/人形机器人）
- **超参数扫描**：$S \in \{5,10,15\}$，$k \in \{1,5,10\}$，$N \in \{5,10,15,20,30\}$
- **基线**：孤立执行（Isolated）；Parallel（任务并行但不共享 KV）

### 实验结果摘要

- **最高加速**：RTX 4090 上 3.7×（$N=30, k=1$），Jetson 上同样有效
- **动作+语言同时**：70 Hz 动作频率 + 200+ tokens/s 语言吞吐（RTX 4090）
- **机器人部署**：端到端延迟 1030 ms → 393 ms（人形，Jetson AGX Thor）

---

## 批判性思考

### 优点

1. **问题定位精准**：将 KV 管理识别为多任务 VLA 推理的瓶颈，切入点新颖且有充分 profiling 数据支撑
2. **与模型解耦**：统一 KV 管理器是系统层优化，不改变模型权重和结构，可直接应用于现有 MoT VLA
3. **全面实验**：覆盖多 benchmark（LIBERO/ALOHA/DROID）、多骨干、多硬件（RTX 4090 + Jetson），泛化性充分验证
4. **实际部署验证**：在真实人形机器人上完成 end-to-end 验证，工程落地价值高
5. **能耗大幅下降**：批处理使能耗降低 78%，对边缘设备意义重大

### 局限性

1. **仅支持语言+动作双任务并行**：作者自述当前仅验证了操控+语言，视频生成等其他模态未覆盖
2. **缺乏与异步控制的集成**：尚未与 asynchronous control 结合，实时性进一步提升的潜力未开发
3. **JAX 框架开销较高**：batch=30 时 JAX 开销达 17.48%，限制了超大批次场景的可用性
4. **Jetson 内存限制**：边缘设备上活跃请求数 $B$ 受内存约束，跨帧 batch 受限

### 潜在改进方向

1. 扩展到视频生成、触觉等额外模态的多专家并行
2. 与模型量化（INT4/INT8）结合进一步降低 Jetson 上的内存压力
3. 引入动态 $k$ 调度（根据当前 batch size 自适应调整每帧解码步数）

### 可复现性评估

- [x] 代码开源（[GitHub](https://github.com/air-embodied-brain/OxyGen)）
- [x] LoRA adapter 开源（HuggingFace）
- [x] 训练细节完整（含 ablation 配置）
- [x] 数据集可获取（LIBERO/ALOHA/DROID 公开；textual-memory 标注已发布）

---

## 关联笔记

### 基于

- [[π0.5]]: 主要评测对象，基于 Qwen3VL 的 MoT VLA
- [[Mixture-of-Transformers]]: OxyGen 所针对的模型架构范式
- [[Continuous Batching]]: 跨帧连续批处理的核心思想来源于 LLM serving 中的 continuous batching（vLLM 等）

### 对比

- [[vla.cpp]]: 另一 VLA 推理优化方向（C++ 实现，主要优化单任务吞吐）
- [[StarVLA]]: 被 OxyGen 评测的多专家 VLA（GR00T + Qwen3VL-PI_v3）
- [[Xiaomi-Robotics-0]]: 被 OxyGen 评测的另一骨干 VLA

### 方法相关

- [[KV Cache]]: OxyGen 的核心优化对象
- [[Prefill]]: 跨任务 KV 共享消除的冗余计算
- [[Continuous Batching]]: 跨帧批处理的技术基础
- [[Autoregressive Generation]]: 语言专家的解码方式
- [[Denoising Diffusion]]: 动作专家的去噪采样机制

### 硬件/数据相关

- [[NVIDIA Jetson AGX Thor]]: 边缘部署目标硬件
- [[LIBERO]]: 主要操控 benchmark

---

## 速查卡片

> [!summary] OxyGen (NeurIPS 2026)
> - **核心**: 统一 KV Cache 管理解决 MoT VLA 多任务并行推理的资源浪费
> - **方法**: 跨任务 KV 共享（消除冗余 prefill）+ 跨帧连续批处理（提升语言吞吐）
> - **结果**: RTX 4090 上 3.7× 加速，70 Hz 动作 + 200+ tokens/s 语言同时达到
> - **代码**: [github.com/air-embodied-brain/OxyGen](https://github.com/air-embodied-brain/OxyGen)

---

*笔记创建时间: 2026-10-20*
