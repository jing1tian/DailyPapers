---
title: "Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation"
method_name: "ReflexVLA"
authors: [Yuxuan Chen, Wanruo Zhang, Xiao Li]
year: 2026
venue: arXiv
tags: [vla, robot-manipulation, temporal-fusion, latent-prediction, latency-optimization, benchmark, reaction-critical]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.14379
created: 2026-08-18
---

# 论文笔记：Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未注明（arXiv 预印本） |
| 日期 | August 2026 |
| 项目主页 | [reflexvla.github.io](https://reflexvla.github.io) |
| 对比基线 | [[VLA-Adapter]]、[[SmolVLA]]、[[PUMA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.14379) / Code: 未公开 |

---

## 一句话总结

> 提出 ReflexBench（6 个动态操控任务基准）与 ReflexVLA（集成潜空间未来预测、多帧时序融合与推理延迟优化），以 1B 参数在反应敏感型操控中超越 4B 的 PUMA，延迟从 125ms 降至 65ms。

---

## 核心贡献

1. **ReflexBench**：首个专注于反应敏感型操控（reaction-critical manipulation）的基准，包含 6 个需要快速响应的动态任务，并提供支持可配置延迟的延迟感知评测框架。
2. **ReflexVLA**：在 [[VLA-Adapter]] 基础上整合三项技术——冻结 [[DINOv2|DINOv3]] 的潜空间未来预测、[[Causal Temporal Attention|因果时序注意力]] 的多帧融合、批量视觉编码与 [[CUDA Graph]] 回放——实现快速且具前瞻性的决策。
3. **系统级延迟优化**：通过批量视觉编码与 CUDA Graph 静态图回放，将端到端推理延迟从 125ms 压缩至 65ms，在 [[ReflexBench]] 上验证了延迟对动态任务成功率的显著影响。

---

## 问题背景

### 要解决的问题

现有 [[Vision-Language-Action Model|VLA]] 在静态操控（如 [[LIBERO]]）上表现优秀，但面对动态变化的环境（移动传送带、飞行球体、旋转插孔）时性能急剧下降。原因有两个：（1）缺乏对未来状态的预见性，（2）推理延迟导致感知到动作执行之间存在时间差，而这在动态任务中是致命的。

### 现有方法的局限

- 大参数 VLA（[[OpenVLA-OFT]] 7B、[[PUMA]] 4B、[[π₀.₅]] 4B）推理延迟大，不适合高频控制；
- 现有评测框架（如 LIBERO）假设静态场景，不能反映延迟对任务成功率的真实影响；
- 小模型（[[SmolVLA]] 0.5B）虽然轻量但缺少时序推理能力，动态任务成功率低。

### 本文的动机

通过三个正交方向同时改进：让模型"看得更久"（时序融合）、"想得更远"（未来预测）、"动得更快"（延迟优化）。三者结合后，1B 参数的 ReflexVLA 可在动态任务中超越 4B 参数的基线，同时在静态基准上不退步。

---

## 方法详解

### 模型架构

ReflexVLA 采用 **编码器-语言模型-动作头** 架构：

- **输入**：语言指令 $l$ + 多帧视觉观测 $\{o_{t-T+1}, \ldots, o_t\}$ + 关节状态 $s_t$
- **视觉骨干**：融合 [[DINOv2]] 与 [[SigLIP]] 特征的 ViT，分辨率 224×224
- **语言骨干**：[[Qwen2.5]]-0.5B 因果 Transformer
- **核心模块**：[[Latent Future Prediction|潜空间未来预测]] + [[Causal Temporal Attention|因果时序注意力]] 融合
- **动作头**：连续回归头，输出 [[Action Chunking|动作块]] $a_{t:t+H-1}$
- **总参数**：1B

![Figure 1: Overview](https://arxiv.org/html/2608.14379v1/figures/reflex_f1.png)

**说明**：ReflexVLA 整体框架。左侧为 ReflexBench 的 6 个动态任务，右侧展示 ReflexVLA 三大组件：潜空间未来预测（绿色路径）、多帧时序融合（蓝色路径）、推理加速（橙色路径）。

---

### 核心模块

#### 模块一：潜空间未来预测（Latent Future Prediction）

**设计动机**：让模型在当前帧预测未来帧的视觉表示，迫使网络学习动态场景的演化规律，而不仅仅是当前状态。使用冻结的 [[DINOv2|DINOv3]] 特征作为预测目标，避免了对高维像素空间建模的计算开销。

**具体实现**：
- 从语言模型隐层提取"未来推理 token" $\mathbf{h}^{future}_i$，对应预测第 $i$ 步未来
- 通过轻量预测头 $f_{pred}$ 映射到 DINOv3 特征维度（$\mathbb{R}^{1024}$）
- 使用掩码余弦相似度损失监督，屏蔽无效时间步
- **关键发现**：使用可训练的未来特征目标（而非冻结）会导致性能崩溃（-31.9%），冻结 DINOv3 是稳定训练的关键

#### 模块二：多帧时序融合（Multi-Frame Temporal Fusion）

**设计动机**：单帧输入无法感知物体运动，通过在视觉骨干内部（而非 LLM 层）对多帧历史进行融合，可以在不增加 LLM token 数量的前提下引入时序运动感知。

**具体实现**：
- 从 ViT 中间层与最终层分别提取特征 $\mathbf{X}_{mid}, \mathbf{X}_{final} \in \mathbb{R}^{B \times V \times T \times P \times D}$
- 对中间层特征做 LayerNorm + 降维投影，加入时序位置编码
- 通过 [[Causal Temporal Attention|因果多头注意力]] 计算帧间注意力差分
- 将差分特征残差加回到当前帧最终层特征
- 消融实验表明中间层（而非最终层）插入时序融合效果最优（+34.9%）

#### 模块三：推理延迟优化

**设计动机**：高频控制要求每步推理 <33ms（30Hz），原始 VLA 125ms 的延迟在动态任务中会错失最优动作窗口。

**具体实现**：
- **批量视觉编码**：将多帧图像拼批处理，充分利用 GPU 并行性
- **[[CUDA Graph]] 回放**：将语言模型推理路径静态化，消除重复的 CUDA kernel 启动开销
- 结合两者：从 125ms → **65ms**，下降 48%

---

## 关键公式

### 公式 1：[[Real-Time Factor|实时因子（RTF）]]

$$
\text{RTF} = \frac{t_{\text{sim}}}{t_{\text{wall}}}
$$

**含义**：衡量仿真器与机器人控制是否解耦，RTF=1 时仿真与真实时间同步，RTF<1 时仿真慢于实时。

**符号说明**：
- $t_{\text{sim}}$：仿真器内部时间步长
- $t_{\text{wall}}$：实际挂钟时间

---

### 公式 2：[[DINOv2|DINOv3]] 目标特征编码

$$
\mathbf{y}_{t+i} = \phi_{\text{DINOv3}}(\mathbf{o}_{t+i}), \quad \mathbf{y}_{t+i} \in \mathbb{R}^{1024}
$$

**含义**：用冻结的 DINOv3 编码器提取未来帧 $o_{t+i}$ 的语义特征作为预测目标，避免了可学习目标导致的训练崩溃。

**符号说明**：
- $\phi_{\text{DINOv3}}$：冻结的 DINOv3 编码器（参数不更新）
- $\mathbf{o}_{t+i}$：第 $t+i$ 时刻的视觉观测
- $i = 1, \ldots, H$：未来预测步数

---

### 公式 3：[[Latent Future Prediction|未来特征预测]]

$$
\hat{\mathbf{y}}_{t+i} = f_{\text{pred}}(\mathbf{h}^{\text{future}}_{i}), \quad i = 1, \ldots, H
$$

**含义**：从语言模型未来推理 token 出发，通过轻量预测头预测未来帧的潜在视觉表示。

**符号说明**：
- $f_{\text{pred}}$：轻量预测头（MLP）
- $\mathbf{h}^{\text{future}}_i$：语言模型中对应第 $i$ 个未来步的隐层状态
- $\hat{\mathbf{y}}_{t+i}$：预测的未来视觉特征

---

### 公式 4：[[Cosine Similarity Loss|掩码余弦相似度损失]]

$$
\mathcal{L}_{\text{future}} = \frac{\sum_{i=1}^{H} m_i \left[1 - \cos\left(\hat{\mathbf{y}}_{t+i}, \mathbf{y}_{t+i}\right)\right]}{\sum_{i=1}^{H} m_i}
$$

**含义**：在语义特征空间中用余弦相似度监督未来预测，$m_i$ 屏蔽无效（填充）时间步，确保损失只对有效帧计算。

**符号说明**：
- $m_i \in \{0, 1\}$：掩码，1 表示有效时间步
- $\cos(\cdot, \cdot)$：余弦相似度
- $H$：未来预测步数

---

### 公式 5：[[Joint Training|联合训练损失]]

$$
\mathcal{L} = \mathcal{L}_{\text{act}} + \lambda_{\text{future}} \mathcal{L}_{\text{future}}
$$

**含义**：动作预测损失与未来预测损失的加权组合，通过 $\lambda_{\text{future}}$ 控制未来预测的重要性。

**符号说明**：
- $\mathcal{L}_{\text{act}}$：标准动作回归损失
- $\lambda_{\text{future}}$：未来预测损失权重（超参数）

---

### 公式 6–7：时序融合特征归一化与投影

$$
\mathbf{z}_{v,p,1:T} = \text{Down}\!\left(\text{LN}\!\left(\mathbf{x}^{mid}_{v,p,1:T}\right)\right) + \mathbf{e}_{1:T}
$$

**含义**：对 ViT 中间层特征做 LayerNorm + 降维线性投影，并加入时序位置嵌入，将多帧信息压缩至低维空间进行注意力计算。

**符号说明**：
- $\mathbf{x}^{mid}_{v,p,1:T}$：第 $v$ 个视角、第 $p$ 个空间 patch 的 $T$ 帧中间层特征
- $\text{Down}$：降维线性层
- $\text{LN}$：Layer Normalization
- $\mathbf{e}_{1:T}$：时序位置编码

---

### 公式 8：[[Causal Temporal Attention|因果时序多头注意力]]

$$
\Delta\mathbf{x}_{v,p,t} = \text{Up}\!\left(\text{MHA}_{\text{causal}}\!\left(\mathbf{z}_{v,p,1:T}\right)_t\right)
$$

**含义**：对每个 patch 的 $T$ 帧历史特征施加因果（单向）多头注意力，提取帧间差分信息，升维后得到当前帧的时序增量。

**符号说明**：
- $\text{MHA}_{\text{causal}}$：因果掩码多头注意力（未来帧信息不泄露）
- $(\cdot)_t$：取第 $t$ 帧的输出
- $\text{Up}$：升维线性层

---

### 公式 9–10：时序融合特征整合

$$
\tilde{\mathbf{x}}_{v,p,t} = \mathbf{x}^{final}_{v,p,t} + \Delta\mathbf{x}_{v,p,t}
$$

$$
\tilde{\mathbf{X}}_t \in \mathbb{R}^{B \times VP \times D}
$$

**含义**：将因果时序增量残差加回到 ViT 最终层特征上，再展平为标准的视觉 token 序列，送入语言模型。

**符号说明**：
- $\mathbf{x}^{final}_{v,p,t}$：ViT 最终层当前帧特征
- $B$：批大小，$V$：视角数，$P$：patch 数，$D$：特征维度

---

### 公式 11：[[CUDA Graph]] 推理

$$
\hat{\mathbf{a}}_{t:t+H-1} = \mathcal{G}\!\left(\mathbf{x}_{\text{text}},\, \mathbf{x}_{\text{vision}},\, \mathbf{x}_{\text{state}}\right)
$$

**含义**：将语言模型推理封装为静态 CUDA Graph $\mathcal{G}$，每次仅更新输入缓冲区，消除重复的 kernel 启动开销，实现显著的延迟下降。

**符号说明**：
- $\mathcal{G}$：预编译的 CUDA 静态计算图
- $\mathbf{x}_{\text{text}}, \mathbf{x}_{\text{vision}}, \mathbf{x}_{\text{state}}$：文字、视觉、状态 token
- $\hat{\mathbf{a}}_{t:t+H-1}$：输出的动作块

---

## 关键图表

### Figure 1：总体概览

![Figure 1](https://arxiv.org/html/2608.14379v1/figures/reflex_f1.png)

**说明**：ReflexBench 的 6 类动态任务与 ReflexVLA 的三大技术组件概览。

---

### Figure 2：延迟感知推理机制

![Figure 2](https://arxiv.org/html/2608.14379v1/figures/infer_bench.png)

**说明**：ReflexBench 的四种推理机制示意图，通过解耦仿真步进与机器人控制周期，可配置化地模拟不同推理延迟对任务执行的影响。

---

### Figure 3：ReflexVLA 架构图

![Figure 3](https://arxiv.org/html/2608.14379v1/figures/reflexvla.png)

**说明**：完整的 ReflexVLA 架构。绿色路径：潜空间未来预测；蓝色路径：多帧时序融合（插入 ViT 中间层）；橙色路径：批量视觉编码 + CUDA Graph 加速。

---

### Figure 4：同步 vs 异步推理对比

![Figure 4](https://arxiv.org/html/2608.14379v1/chunk.png)

**说明**：SmolVLA 在不同推理频率下同步与异步推理的成功率对比。异步推理（允许 action chunk 在下一帧观测到达前执行）在低频时显著优于同步推理，说明延迟是动态任务的核心瓶颈。

---

### Figure 5：Action Chunk 大小与执行步数矩阵

![Figure 5](https://arxiv.org/html/2608.14379v1/figures/reflex_real.png)

**说明**：SmolVLA 在不同 chunk size（训练超参）与 action horizon（推理时实际执行步数）组合下的成功率矩阵，推理频率固定 30Hz。揭示了 chunk size 与 horizon 之间的权衡关系。

---

### Figure 6：真实世界部署示例

![Figure 6](https://arxiv.org/html/2608.14379v1/figures/reflex_real.png)

**说明**：AgileX Piper 机械臂上的真实实验展示，从上到下分别为：传送带拾取放置（Conveyor Belt Pick-and-Place）、按键操作（PressButtons）、接球（CatchBalls）。

---

### Table I：ReflexBench 性能对比

| 模型 | 参数 | 传送带拾取 | 接球 | 打地鼠 | 滚球拦截 | 投球 | 旋转插孔 | **平均** |
|------|------|-----------|------|--------|---------|------|---------|---------|
| [[OpenVLA-OFT]] | 7B | 58.0±1.2 | 5.3±1.2 | 100.0±0.0 | 41.4±4.2 | 10.0±3.3 | 1.3±0.7 | 36.0 |
| [[π₀.₅]] | 4B | 39.1±1.0 | 6.0±0.7 | 98.9±0.3 | 36.8±1.8 | 34.0±2.7 | 6.7±1.4 | 36.9 |
| [[PUMA]] | 4B | 67.4±1.7 | 4.0±0.7 | 100.0±0.0 | 85.1±3.4 | 33.8±1.0 | 11.1±1.7 | 50.2 |
| DynamicVLA | 0.5B | 30.6±2.4 | 4.4±1.0 | 90.2±1.7 | 45.8±1.7 | 36.2±4.7 | 16.7±4.2 | 37.3 |
| [[SmolVLA]] | 0.5B | 19.3±2.0 | 2.9±0.8 | 100.0±0.0 | 70.2±2.0 | 27.1±1.0 | 10.0±4.6 | 38.3 |
| VLA-Adapter（基线） | 1B | 36.8±2.4 | 6.0±2.0 | 68.4±3.9 | 23.1±7.7 | 29.1±5.2 | 18.4±2.5 | 30.3 |
| **ReflexVLA（Ours）** | **1B** | **73.8±4.1** | **7.3±0.7** | **100.0±0.0** | **77.1±7.1** | 31.7±1.0 | 12.4±1.5 | **50.4** |

**说明**：ReflexVLA 以 1B 参数达到 50.4% 平均成功率，超越 4B 的 PUMA（50.2%），尤其在传送带拾取（+6.4pp）和滚球拦截（-8pp，稍弱）任务上差异显著。接球任务所有模型均极低（<8%），揭示该任务的极端难度。

---

### Table II：LIBERO 静态基准（泛化性）

| 模型 | Spatial | Object | Goal | Long | **平均** |
|------|---------|--------|------|------|---------|
| [[OpenVLA-OFT]] | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| [[π₀.₅]] | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| VLA-Adapter | 97.8 | 99.2 | 97.2 | 95.0 | 97.3 |
| **ReflexVLA（Ours）** | 98.2 | 99.2 | 98.0 | 93.6 | **97.2** |

**说明**：ReflexVLA 在 LIBERO 上保持与基线相当的 97.2%，证明动态任务优化不以牺牲静态任务为代价。

---

### Table III：消融实验（渐进式）

| 方法 | 变体 | 成功率 (%) | 延迟 (ms) |
|------|------|-----------|----------|
| 基线（VLA-Adapter） | — | 36.8 | 81.5 |
| + 未来预测 | 可训练目标 | 4.9 (−31.9) | 82.5 |
| + 未来预测 | **冻结目标** | 62.8 (+26.0) | 82.5 |
| + 时序融合 | 交叉注意力 | 66.1 (+29.3) | 127.3 |
| + 时序融合 | MHA | 68.2 (+31.4) | 125.8 |
| + 时序融合 | **MHA（中间层）** | 71.7 (+34.9) | 125.1 |
| + 延迟优化 | **批量编码 + CUDA Graph** | **73.8 (+37.0)** | **65.0** |

**关键发现**：
1. 未来预测目标必须冻结，否则负向影响达 -31.9%
2. 时序融合插入中间层优于最终层
3. 延迟优化不仅降低延迟（−48%），还因更高推理频率提升了成功率（+2.1pp）

---

### Table IV：真实机器人实验结果

| 模型 | 传送带拾取 | PressButtons | 接球 |
|------|-----------|-------------|------|
| [[SmolVLA]] | 2/20 | 0.9 | 3.8 |
| [[PUMA]] | 13/20 | 20.8 | 5.4 |
| **ReflexVLA（Ours）** | **16/20** | **22.5** | **6.7** |

**说明**：在 AgileX Piper 机械臂上，ReflexVLA 以 1/4 参数量（1B vs 4B PUMA）在所有任务上实现相当或更优的表现。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[ReflexBench]]（仿真） | 6 任务 × 200 episodes | 动态反应任务 | 训练 + 测试 |
| [[LIBERO]] | 4 子任务 | 静态操控基准 | 泛化性测试 |
| 真实机器人数据 | 3 任务 | AgileX Piper 机械臂 | 真实验证 |

### 实现细节

- **视觉骨干**：DINOv2 + SigLIP 融合 ViT，224×224 输入
- **语言模型**：Qwen2.5-0.5B（参数不冻结）
- **未来预测目标**：冻结 DINOv3（$\mathbb{R}^{1024}$ 特征）
- **历史帧数**：$T = 3$（含当前帧）
- **动作块**：chunk size = 8，action horizon = 2（仿真评测）
- **推理模式**：异步推理（评测时标准配置）
- **硬件**：单张 NVIDIA RTX 5880 Ada GPU（评测）
- **优化器**：论文未明确说明

### 可视化结果

真实机器人实验中，ReflexVLA 在传送带拾取任务（80% vs 65% for PUMA）、PressButtons（22.5 vs 20.8）、CatchBalls（6.7 vs 5.4）均小幅超越 PUMA，且参数量仅为其 1/4，展示了效率优势。

---

## 批判性思考

### 优点

1. **效率与性能双优**：1B 参数媲美甚至超越 4B 基线，在参数效率上具有显著优势。
2. **系统性设计**：三个组件（预测、融合、加速）相互独立、渐进可加，消融实验设计清晰。
3. **评测框架创新**：ReflexBench 填补了动态操控评测的空白，延迟感知框架具有方法论价值。
4. **冻结目标的关键洞察**：发现可训练预测目标会崩溃、冻结才稳定，这是工程上的重要经验。

### 局限性

1. **接球任务全线失败**：所有模型在 Ball Catching 上成功率均 <8%，说明该任务难度远超当前 VLA 能力边界，论文未深入分析根本原因。
2. **数据规模极小**：每任务仅 200 episodes 训练，实际应用场景的泛化能力存疑。
3. **未公开代码与模型**：可复现性受限，ReflexBench 是否开源也不明确。
4. **延迟优化有局限**：CUDA Graph 要求输入形状固定，在动态输入（如变长语言指令）场景下需要额外工程处理。

### 潜在改进方向

1. **更丰富的动态任务**：ReflexBench 目前任务类型仍有限，缺乏双臂协作、人机交互等场景。
2. **自适应 action horizon**：当前 chunk size 与 horizon 需手动调参，自适应选择可进一步提升性能。

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节基本完整（部分缺失优化器配置）
- [ ] 数据集（ReflexBench）可获取

---

## 关联笔记

### 基于

- [[VLA-Adapter]]：直接基础架构，ReflexVLA 在其上添加三个模块
- [[DINOv2]]：视觉特征提取骨干（融合 SigLIP）
- [[Qwen2.5]]：语言模型骨干

### 对比

- [[SmolVLA]]：0.5B 轻量 VLA，无时序融合
- [[PUMA]]：4B 参数，高延迟但静态任务强
- [[OpenVLA-OFT]]：7B 参数开放 VLA
- [[π₀.₅]]：4B 流匹配 VLA

### 方法相关

- [[Latent Future Prediction]]：核心预测机制
- [[Causal Temporal Attention]]：时序融合关键技术
- [[CUDA Graph]]：推理加速组件
- [[Action Chunking]]：动作输出形式

### 数据集 / 基准

- [[ReflexBench]]：本文提出的动态操控基准
- [[LIBERO]]：静态操控泛化性基准

---

## 速查卡片

> [!summary] Reflex / ReflexVLA
> - **核心**：为反应敏感型操控任务设计的快速、前瞻性 VLA
> - **方法**：潜空间未来预测（冻结 DINOv3 目标）+ 中间层因果时序融合 + CUDA Graph 推理加速
> - **结果**：1B 参数，ReflexBench 50.4%（≈ 4B PUMA），延迟 65ms，LIBERO 97.2%
> - **代码**：未开源（项目页：reflexvla.github.io）

---

*笔记创建时间：2026-08-18*
