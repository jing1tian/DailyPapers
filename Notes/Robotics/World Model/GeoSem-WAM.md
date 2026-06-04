---
title: "GeoSem-WAM: Geometry- and Semantic-Aware World Action Models"
method_name: "GeoSem-WAM"
authors: [Fulong Ma, Daojie Peng, Wenjun Yue, Jiahang Cao, Bintao Wang, Qiang Zhang, Jun Ma]
year: 2026
venue: arXiv
tags: [world-action-model, robot-manipulation, geometric-learning, semantic-learning, imitation-learning, diffusion-transformer, multi-task-learning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.03188
created: 2026-06-04
---

# 论文笔记：GeoSem-WAM: Geometry- and Semantic-Aware World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | HKUST(GZ), HKU, USTC, OC, SDU, X-Humanoid |
| 日期 | June 2026 |
| 项目主页 | 暂未公开 |
| 对比基线 | [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.03188) / Code: 暂未公开 |

---

## 一句话总结

> 通过在训练阶段引入几何与语义辅助监督分支，GeoSem-WAM 在不增加推理开销的前提下显著提升了 [[World Action Model]] 的动作预测精度与鲁棒性。

---

## 核心贡献

1. **机理阐明**: 明确指出 [[World Action Model|WAM]] 的核心优势来自训练阶段的表示学习（预测性监督），而非推理时的显式未来想象
2. **结构化多模态监督**: 提出将 RGB、几何深度、语义分割三路信号统一纳入训练目标，通过 [[DPT|DPT 风格辅助头]] 实现密集结构化监督
3. **高效推理不变**: 推理时仅做单次前向传播，不生成视频帧，不使用辅助分支，计算成本与基线相当

---

## 问题背景

### 要解决的问题

当前 [[World Action Model]] 方法依赖未来 RGB 视频预测作为自监督信号来学习动作相关表示，但纯 RGB 预测忽略了场景的几何结构与语义信息，导致模型在空间感知和语义推理方面存在不足。

### 现有方法的局限

- [[Fast-WAM]]、[[WorldVLA]]、[[Motus]] 等方法均以 RGB 图像预测为唯一世界建模目标
- RGB 像素级重建对几何结构（深度、空间布局）和语义类别（物体类别、任务状态）不敏感
- 部分方法在推理阶段仍需显式展开未来轨迹，计算代价高

### 本文的动机

WAM 的有效性本质上来自训练时的**表示学习**：预测未来迫使模型编码对决策有用的特征。若将预测目标从单一 RGB 扩展到几何和语义信号，模型将学到更丰富的场景表示，从而提升动作预测质量，同时不影响推理效率。

---

## 方法详解

### 模型架构

![Figure 1: GeoSem-WAM 整体架构](https://arxiv.org/html/2606.03188v1/figures/architecture.png)

**说明**: 整体图展示训练阶段的三路预测分支（RGB、几何、语义）与动作分支；虚线框内为推理阶段（仅保留动作分支，单次前向传播）。

GeoSem-WAM 采用 [[Mixture-of-Transformers|MoT（Mixture-of-Transformers）]] 架构，以 [[Wan 2.2-5B]] [[Video Diffusion Transformer]] 为主干：

- **输入**: [[T5]] 文本编码器产生的任务指令嵌入 $c$ + [[VAE]] 编码的观测视频潜变量 $z_t$
- **Video DiT 分支**: 预测未来 RGB 视频潜变量（训练时使用，推理时跳过）
- **Action DiT 分支**: 生成动作块 $a_{t:t+H-1}$（训练+推理均使用）
- **DPT 辅助头**: 几何预测头（L1 损失）+ 语义分割头（交叉熵损失），训练时对视频 token 进行密集监督
- **两分支共享注意力**: Video DiT 与 Action DiT 通过 [[Shared Attention|共享注意力]] 机制交互
- **总参数**: ~5B（基于 Wan 2.2-5B 骨干）

### 核心模块

#### 模块 1: DPT 风格辅助预测头

![Figure 2: DPT 辅助头架构](https://arxiv.org/html/2606.03188v1/figures/dpt.png)

**说明**: DPT 辅助头采用 3D Reassemble 模块提取多尺度视频 token，经 Fusion 模块逐级上采样，最终输出与输入分辨率对齐的密集预测图（深度图 / 语义分割图）。

**设计动机**: 利用 [[Dense Prediction Transformer|DPT（Dense Prediction Transformer）]] 的多尺度特征融合能力，对视频潜变量 token 施加像素级密集监督，迫使模型在 latent space 中编码几何和语义信息。

**具体实现**:
- 3D Reassemble 模块：从 Video DiT 不同层提取 token，重整为 3D 特征图
- Fusion 模块：逐级上采样融合多尺度特征
- 几何头输出：单通道深度预测图 $\hat{o}^{\text{geo}}_\tau$
- 语义头输出：多类别 logits $\hat{o}^{\text{sem}}_\tau$

#### 模块 2: 视频-动作共享注意力 (MoT)

**设计动机**: 利用 [[Mixture-of-Experts|MoE]] 思想，让视频分支与动作分支在 [[Transformer]] 的注意力层共享场景上下文，使动作 token 能充分感知世界模型学到的几何与语义特征。

**具体实现**:
- Video DiT 与 Action DiT 的 [[Self-Attention]] 层共享 Key/Value
- 动作 token 以视频 token 为条件进行 [[Cross-Attention|交叉注意力]] 融合
- 推理时 Video DiT 分支完全关闭，仅保留 Action DiT 的单次前向传播

---

## 关键公式

### 公式 1: [[Flow Matching|潜变量世界建模（RGB 分支）]]

$$
\hat{z}^{\sigma}_{t:t+K} = (1-\sigma)z^{\text{gt}}_{t:t+K} + \sigma \epsilon_z
$$

**含义**: 对目标视频 VAE 潜变量 $z^{\text{gt}}$ 施加噪声，得到带噪潜变量 $\hat{z}^{\sigma}$，用于 [[Flow Matching]] 训练。

**符号说明**:
- $z^{\text{gt}}_{t:t+K}$: 未来 $K$ 帧的 GT VAE 潜变量
- $\sigma \in [0, 1]$: 噪声强度（flow time）
- $\epsilon_z$: 标准高斯噪声

$$
\hat{v}_z = f^{\text{rgb}}_{\theta}(\hat{z}^{\sigma}_{t:t+K},\, \sigma,\, c), \quad v_z = \epsilon_z - z^{\text{gt}}_{t:t+K}
$$

**含义**: Video DiT $f^{\text{rgb}}_\theta$ 预测 flow target $v_z$（噪声与干净潜变量之差）。

$$
\mathcal{L}_{\text{rgb}} = \left\| \hat{v}_z - v_z \right\|_2^2
$$

**含义**: RGB 分支的 flow matching 回归损失（MSE）。

---

### 公式 2: [[Action Diffusion|动作建模损失]]

$$
a^{\sigma}_{t:t+H-1} = (1-\sigma)a^{\text{gt}}_{t:t+H-1} + \sigma \epsilon_a
$$

$$
\hat{v}_a = f^{\text{act}}_{\phi}(a^{\sigma}_{t:t+H-1},\, \sigma,\, z_t), \quad v_a = \epsilon_a - a^{\text{gt}}_{t:t+H-1}
$$

$$
\mathcal{L}_{\text{act}} = \left\| \hat{v}_a - v_a \right\|_2^2
$$

**含义**: 动作块的 flow matching 损失。Action DiT $f^{\text{act}}_\phi$ 以当前观测潜变量 $z_t$ 为条件，预测动作 flow。

**符号说明**:
- $a^{\text{gt}}_{t:t+H-1}$: 未来 $H$ 步的 GT 动作序列（[[Action Chunking|动作块]]）
- $z_t$: 当前帧的 VAE 潜变量（推理时的观测条件）
- $\phi$: Action DiT 参数

---

### 公式 3: [[Dense Prediction|几何监督损失]]

$$
\mathcal{L}_{\text{geo}} = \frac{1}{K} \sum_{\tau=t}^{t+K} \left\| \hat{o}^{\text{geo}}_\tau - o^{\text{geo}}_\tau \right\|_1
$$

**含义**: 对未来 $K$ 帧的预测深度图与 GT 深度图施加 L1 损失，监督几何分支。

**符号说明**:
- $\hat{o}^{\text{geo}}_\tau$: DPT 几何头预测的第 $\tau$ 帧深度图
- $o^{\text{geo}}_\tau$: GT 深度图（来自传感器或深度相机）
- $K$: 预测帧数

---

### 公式 4: [[Semantic Segmentation|语义监督损失]]

$$
\mathcal{L}_{\text{sem}} = \frac{1}{K} \sum_{\tau=t}^{t+K} \text{CE}\!\left(\hat{o}^{\text{sem}}_\tau,\, o^{\text{sem}}_\tau\right)
$$

**含义**: 对未来 $K$ 帧的语义分割预测与 GT 标签施加交叉熵损失，监督语义分支。

**符号说明**:
- $\hat{o}^{\text{sem}}_\tau$: DPT 语义头预测的第 $\tau$ 帧语义 logits
- $o^{\text{sem}}_\tau$: GT 语义分割标签（像素级类别）
- $\text{CE}$: 像素级交叉熵

---

### 公式 5: [[Multi-Task Learning|统一训练目标]]

$$
\mathcal{L} = \lambda_{\text{rgb}} \cdot \mathcal{L}_{\text{rgb}} + \lambda_{\text{geo}} \cdot \mathcal{L}_{\text{geo}} + \lambda_{\text{sem}} \cdot \mathcal{L}_{\text{sem}} + \lambda_{\text{act}} \cdot \mathcal{L}_{\text{act}}
$$

**含义**: 四路损失的加权求和，在单次前向传播中联合优化 RGB 预测、几何预测、语义预测和动作生成。

**符号说明**:
- $\lambda_{\text{rgb}}, \lambda_{\text{geo}}, \lambda_{\text{sem}}, \lambda_{\text{act}}$: 各分支的损失权重系数（论文未公开具体数值）
- $\mathcal{L}_{\text{rgb}}$: RGB flow matching 损失
- $\mathcal{L}_{\text{geo}}$: 几何（深度）L1 损失
- $\mathcal{L}_{\text{sem}}$: 语义分割交叉熵损失
- $\mathcal{L}_{\text{act}}$: 动作 flow matching 损失

---

## 关键图表

### Figure 1: 整体架构概览

![GeoSem-WAM 整体架构](https://arxiv.org/html/2606.03188v1/figures/architecture.png)

**说明**: 训练阶段（完整图）包含 Video DiT（RGB预测）、Action DiT（动作生成）、DPT 几何头和语义头四路并行输出；推理阶段（虚线框）仅保留 Action DiT，从当前观测直接生成动作块，无需生成未来视频。

---

### Figure 2: DPT 辅助头架构

![DPT 辅助头](https://arxiv.org/html/2606.03188v1/figures/dpt.png)

**说明**: 3D Reassemble 模块从 Video DiT 中间层抽取多尺度时空 token，Fusion 模块逐级融合上采样，分别输出深度预测和语义分割图。

---

### Figure 3: 表示质量分析

![语义聚类与几何深度探针](https://arxiv.org/html/2606.03188v1/figures/dep_sem_compare.png)

**说明**: (a)(b) t-SNE 可视化语义 token 嵌入，GeoSem-WAM 的 token 呈现更清晰的类别聚类，说明几何语义监督确实改善了潜变量的语义区分度；(c) 冻结骨干的深度探测（frozen-backbone depth probing）实验证明 GeoSem-WAM 的 latent 包含更丰富的几何信息。

---

### Figure 4: 真实机器人实验

![Franka 真实机器人实验](https://arxiv.org/html/2606.03188v1/figures/real_franka.png)

**说明**: 4 类核心任务（Easy-Pick, Multi-Pick, Multi-Goal, Pick-Pour）+ 背景/高度泛化测试，每项 50 次实验，GeoSem-WAM 平均成功率 95.4%，较 Fast-WAM（88.9%）提升 6.6pp。

---

### Figure 5: 仿真环境与多模态观测

![仿真环境总览](https://arxiv.org/html/2606.03188v1/figures/sim_libero.png)

**说明**: 展示 LIBERO 和 RoboTwin 2.0 仿真环境中的多模态输入（RGB 图、深度图、语义分割图），说明训练数据同时包含 RGB 和结构化标注。

---

### Figure 6: Pick-Pour 任务执行详解

![Pick-Pour 执行过程](https://arxiv.org/html/2606.03188v1/figures/real_franka_pour.png)

**说明**: Franka 机械臂执行 Pick-Pour 任务的完整帧序列，包含多视角（腕部相机 + 第三人称相机）同步观测，验证了模型对多步操作任务的规划能力。

---

### Table 1: LIBERO 基准测试结果

| 方法 | 范式 | Spatial (%) | Object (%) | Goal (%) | Long (%) | Average (%) |
|------|------|-------------|-----------|----------|----------|-------------|
| Fast-WAM | WAM | 97.2 | 100.0 | 97.0 | 95.2 | 97.60 |
| **GeoSem-WAM** | **WAM** | **99.0** | **100.0** | **98.2** | **97.0** | **98.55** |

**关键发现**: GeoSem-WAM 在所有 LIBERO 子集上均超过 Fast-WAM，特别是在需要空间理解的 Spatial (+1.8pp) 和长程规划的 Long (+1.8pp) 子集上提升最显著。

---

### Table 2: RoboTwin 2.0 基准测试结果

| 方法 | 范式 | Embodied PT. | Clean SR (%) | Random SR (%) | Average SR (%) |
|------|------|--------------|--------------|---------------|----------------|
| Fast-WAM | WAM | ✗ | 91.88 | 91.78 | 91.80 |
| LingBot-VA | WAM | ✓ | 92.90 | 91.50 | 92.20 |
| **GeoSem-WAM** | **WAM** | **✗** | **92.94** | **92.14** | **92.52** |

**关键发现**: GeoSem-WAM 无需 embodied pre-training 即可超越使用了预训练的 LingBot-VA，说明结构化监督的质量可以弥补数据量的不足。

---

### Table 3: 消融实验（LIBERO）

| 配置 | RGB | Geometry | Semantic | Avg SR (%) | Δ SR (%) |
|------|-----|----------|----------|-----------|----------|
| RGB-only (Fast-WAM) | ✓ | ✗ | ✗ | 97.6 | — |
| +Geometry | ✓ | ✓ | ✗ | 98.2 | +0.61 |
| +Semantic | ✓ | ✗ | ✓ | 98.1 | +0.51 |
| +Both (GeoSem-WAM) | ✓ | ✓ | ✓ | **98.6** | **+1.02** |

**关键发现**: 几何和语义监督的贡献具有互补性（0.61 + 0.51 ≈ 1.02），两者叠加带来超线性增益，说明深度信息与语义信息编码了互不重叠的有用特征。

---

### Table 4: 真实 Franka 机器人结果

| 模型 | Easy-Pick (%) | Easy-Pick-B1 (%) | Easy-Pick-B2 (%) | Easy-Pick-D (%) | Multi-Pick (%) | Multi-Goal (%) | Pick-Pour (%) | Avg (%) |
|------|--------------|------------------|------------------|-----------------|----------------|----------------|----------------|---------|
| Fast-WAM | 100 | 86 | 86 | 80 | 92 | 90 | 88 | 88.9 |
| **GeoSem-WAM** | **100** | **96** | **94** | **92** | **98** | **96** | **92** | **95.4** |

**关键发现**: 在背景变化（B1/B2）和高度变化（D）等泛化测试中提升最为显著（+8~+12pp），说明几何语义感知对视觉干扰的鲁棒性有直接贡献。

---

## 实验结果

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 4 个任务套件（Spatial/Object/Goal/Long）| 仿真桌面操作，多任务 | 训练 + 测试 |
| RoboTwin 2.0 | Clean / Random 两种设置 | 仿真，物体姿态随机化 | 训练 + 测试 |
| Real Franka | 50 次/任务，8 类任务 | Franka Emika Panda，真实场景 | 测试 |

### 实现细节

- **Backbone**: Wan 2.2-5B [[Video Diffusion Transformer]]（预训练 VAE + T5 编码器）
- **架构**: [[Mixture-of-Transformers|MoT]]，Video DiT + Action DiT 共享注意力
- **训练硬件**: 2 × NVIDIA H800 GPU
- **推理硬件**: 1 × NVIDIA RTX 4090 GPU（单卡部署）
- **多模态输入**: 第三人称 RGB + 深度图 + 语义分割（LIBERO/RoboTwin 数据包含三模态标注）
- **损失权重**: $\lambda_{\text{rgb}}, \lambda_{\text{geo}}, \lambda_{\text{sem}}, \lambda_{\text{act}}$（论文未公开具体数值）
- **推理**: 单次前向传播，不生成视频，不使用辅助头

### 可视化结果

- t-SNE 可视化（Figure 3）直观展示 GeoSem-WAM latent token 的语义聚类质量优于 Fast-WAM
- 冻结骨干深度探测实验定量验证了几何信息被有效编码进 latent space
- 真实机器人视频（Figure 4/6）展示了任务执行的稳定性和多场景泛化能力

---

## 批判性思考

### 优点

1. **训练-推理解耦设计优雅**: 辅助监督仅在训练时施加，推理效率与单路方法完全一致，工程实用性强
2. **消融实验充分**: 分别验证了几何和语义监督的独立贡献，以及两者的互补性，说服力强
3. **真实机器人验证**: 在 8 类任务、400 次真实实验中验证，泛化性能（背景/高度变化）的提升尤为有说服力

### 局限性

1. **标注依赖**: DPT 头需要像素级深度和语义标注，数据采集成本高，限制了方法在无结构化标注场景下的适用性
2. **梯度冲突风险**: 四路异质损失联合优化存在 [[Gradient Conflict|梯度冲突]] 风险，论文未提供具体的损失平衡策略（$\lambda$ 值未公开）
3. **基线对比有限**: 主要与 Fast-WAM 对比，未与更广泛的 VLA 方法（如 $\pi_0$、OpenVLA）进行系统比较

### 潜在改进方向

1. **自监督几何/语义特征**: 利用 [[DINO]]、[[Depth Anything]] 等基础模型提取无标注的伪标签，降低数据依赖
2. **梯度外科手术**: 引入 [[Gradient Surgery|Gradient Surgery]] 或 [[PCGrad]] 缓解多任务梯度冲突
3. **扩展到更复杂场景**: 户外、双臂操作等更高自由度场景的泛化性待验证

### 可复现性评估

- [ ] 代码开源（暂未公开）
- [ ] 预训练模型（暂未公开）
- [x] 训练细节部分完整（硬件已说明，$\lambda$ 值未公开）
- [x] 数据集可获取（LIBERO/RoboTwin 均为公开基准）

---

## 关联笔记

### 基于

- [[Fast-WAM]]: 直接基线，GeoSem-WAM 在其 MoT 架构上增加辅助分支
- [[Wan 2.2-5B]]: 使用其 Video DiT 作为骨干网络
- [[WorldVLA]]: World Action Model 范式的早期工作

### 对比

- [[Fast-WAM]]: 主要对比方法，同为 WAM 范式
- [[LingBot-VA]]: RoboTwin 2.0 上的对比方法（有 embodied pre-training）
- [[Motus]]: World Action Model 相关工作

### 方法相关

- [[World Action Model]]: 所属方法范式
- [[Mixture-of-Transformers]]: 核心架构设计
- [[Dense Prediction Transformer]]: DPT 辅助头的基础
- [[Flow Matching]]: 训练目标的数学框架
- [[Action Chunking]]: 动作输出格式

### 硬件/数据相关

- [[Franka Emika Panda]]: 真实机器人实验平台
- [[LIBERO]]: 仿真基准
- [[RoboTwin 2.0]]: 仿真基准

---

## 速查卡片

> [!summary] GeoSem-WAM
> - **核心**: 在 WAM 训练时加入几何+语义辅助监督，推理时不增加开销
> - **方法**: DPT 风格辅助头 + MoT 架构 + 四路联合损失
> - **结果**: LIBERO 98.55%（+0.95pp），真实 Franka 95.4%（+6.6pp）
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-06-04*
