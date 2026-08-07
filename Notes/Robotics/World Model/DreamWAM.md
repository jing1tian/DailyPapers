---
title: "DreamWAM: Beyond RGB Future Prediction for World Action Models"
method_name: "DreamWAM"
authors: [Shanglin Yuan, Weiheng Zhao, Xin Shi, Haoyi Jiang, Xianda Guo, Liu Liu, Wenyu Liu, Wei Sui, Xinggang Wang]
year: 2026
venue: arXiv
tags: [world-action-model, future-prediction, multi-modal-supervision, robot-manipulation, flow-matching, distribution-shift, robustness]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.04996
created: 2026-08-07
---

# 论文笔记：DreamWAM: Beyond RGB Future Prediction for World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Huazhong University of Science and Technology (HUST) |
| 日期 | August 2026 |
| 项目主页 | [github.com/hustvl/DreamWAM](https://github.com/hustvl/DreamWAM) |
| 对比基线 | [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.04996) / [Code](https://github.com/hustvl/DreamWAM) |

---

## 一句话总结

> DreamWAM 在训练时用运动、几何、语义四种视角扩展 RGB 未来预测，推理时完全退化为 RGB-only 接口，在分布外扰动下将机器人成功率从 51.36% 提升至 63.44%（仿真）、从 55.6% 提升至 74.4%（真实机器人）。

---

## 核心贡献

1. **识别未来表征为核心设计变量**: 指出现有 [[World Action Model|WAM（世界动作模型）]] 将未来学习组织在 RGB 像素空间，导致任务相关状态转变与纹理/光照/背景等无关变化相互纠缠，鲁棒性不足。
2. **异构未来建模（Heterogeneous Future Modeling）**: 提出四视角结构化未来表征——外观（RGB）、运动（[[RAFT|光流]]）、几何（[[Depth Anything 3|Depth Anything V3]]）、语义（[[DINOv2]]），分别通过联合潜变量去噪和门控残差注入两种路径进行监督。
3. **训练增强 + 推理简化的解耦设计**: 所有非 RGB 路径仅在训练期间激活，推理时全部关闭，保持与 [[Fast-WAM]] 相同的 RGB-only 接口，无额外推理开销。

---

## 问题背景

### 要解决的问题

[[World Action Model|WAM]] 需要预测未来帧并据此生成动作。现有方法（如 [[Fast-WAM]]）将未来预测完全组织在 RGB 像素空间，即用 [[VideoDiT]] 去噪未来视频帧的 RGB 潜变量，再通过 [[ActionDiT]] 联合去噪动作。

### 现有方法的局限

RGB-only 预测目标让模型同时学习任务相关的状态转变（物体位置、抓取关系）和与动作无关的视觉变化（背景纹理、光照、相机视角）。这种纠缠导致：
- 在训练分布内成功率尚可，但面对视觉扰动（背景变换、光照变化、相机视角偏移）时泛化能力大幅下降
- 已有 WAM 如 WorldVLA 在 LIBERO-Plus 分布外测试中平均成功率仅 25.61%，远低于 VLA 方法

### 本文的动机

不同表征形式对不同信息的提取能力不同：
- **运动**（[[RAFT|光流]]）直接编码帧间变化，与执行动作的结果强相关
- **几何**（深度）编码空间布局，对光照/纹理不变
- **语义**（[[DINOv2]] patch 特征）维持物体级一致性，对外观变化鲁棒

在训练时引入这些辅助监督，可以迫使 [[VideoDiT]] 的内部表征向动作相关方向倾斜，从而在推理时（即使仅看 RGB）也能更好地捕捉状态转变。

---

## 方法详解

### 模型架构

DreamWAM 采用 **双专家扩散** 架构，基于 [[Fast-WAM]] 扩展：
- **输入**: 语言指令 $l$ + 当前观测帧 $o_t$（RGB-only，推理时）
- **视频骨干**: [[VideoDiT]]（基于 [[Wan2.2]]-5B）负责去噪未来 RGB 视频潜变量
- **动作骨干**: [[ActionDiT]] 通过共享注意力层与 VideoDiT 交互，去噪动作轨迹
- **输出**: 未来 H 帧 RGB 预测 + 动作块 $a_{t:t+k}$
- **训练时额外输入**: 离线预计算的光流、深度、语义特征（来自未来帧真值）

### 核心模块

#### 模块1: 结构化未来视角（Structured Future Views）

**设计动机**: 利用不同模态编码器提取互补的、动作相关的未来状态信息

**四视角定义**:
- **外观** $z^{\text{rgb}}_{1:H}$：[[Wan2.2]] VAE 编码未来 RGB 帧的潜变量（标准 [[Flow Matching]] 目标）
- **运动** $z^{\text{mot}}_{1:H}$：[[RAFT]] 估计相邻未来帧间的稠密[[光流 (Optical Flow)|光流]]，再经同一 VAE 编码为潜变量
- **几何** $f^{\text{geo}}_{1:H}$：[[Depth Anything 3|Depth Anything V3]] 从未来帧提取的深度特征（ViT patch 特征，非完整 3D 重建）
- **语义** $f^{\text{sem}}_{1:H}$：[[DINOv2]] ViT-B/14 的 patch 级特征，保持物体/区域级一致性

所有非 RGB 视角均在训练前从训练视频中离线预计算，推理时完全移除。

#### 模块2: 联合潜变量去噪（Joint Latent Denoising）用于 RGB + 运动

**设计动机**: 光流潜变量与 RGB 潜变量具有完全相同的时空结构（均经过同一 VAE 编码），适合直接拼接进入同一去噪流

**具体实现**:
- 对两个模态分别施加 [[Flow Matching]] 扰动，然后沿通道维度拼接，输入 [[VideoDiT]]
- VideoDiT 在同一去噪流中同时学习两种视角，使时序变化在预测中得到显式编码
- 训练用 flow-matching 损失监督两个潜变量；推理时运动通道置零（零向量输入），残差通道移除

#### 模块3: 门控残差建模（Gated Residual Modeling）用于几何 + 语义

**设计动机**: 几何和语义特征来自外部编码器（空间/维度与 VAE 潜变量不对齐），不适合直接拼接去噪；需要在保留预训练视频路径的同时引入受控修正

**具体实现**:
- 在 [[VideoDiT]] 每个 Transformer 层后加轻量级残差分支 $R^j_\ell$（view 特定）和可学习门 $g^j_\ell$（学习控制贡献幅度）
- 分支输出通过逐元素乘法叠加在标准隐藏状态上
- 最终用预测头 $P_j$ 从最后一层隐藏状态预测对齐特征，以 L2 损失监督
- 推理时残差分支完全移除，VideoDiT 退化为标准 RGB-only 模型

---

## 关键公式

### 公式1: [[World Action Model|结构化未来表征]]

$$
\mathcal{Y}_{1:H} = \{z^{\text{rgb}}_{1:H},\; z^{\text{mot}}_{1:H},\; f^{\text{geo}}_{1:H},\; f^{\text{sem}}_{1:H}\}
$$

**含义**: DreamWAM 将未来 H 帧的状态分解为四个互补视角的联合表征，而非单一 RGB 序列。

**符号说明**:
- $z^{\text{rgb}}_{1:H}$: [[Wan2.2]] VAE 编码的 RGB 潜变量序列
- $z^{\text{mot}}_{1:H}$: [[RAFT]] 光流经 VAE 编码的运动潜变量序列
- $f^{\text{geo}}_{1:H}$: [[Depth Anything 3|Depth Anything V3]] 深度特征序列
- $f^{\text{sem}}_{1:H}$: [[DINOv2]] ViT patch 语义特征序列
- $H$: 预测未来帧数

### 公式2: [[Flow Matching|联合 Flow-Matching 扰动]]

$$
x^q_t = (1-t)\, z^q + t\, \epsilon^q, \quad q \in \{\text{rgb},\, \text{mot}\}
$$

$$
x^{\text{joint}}_t = \text{Concat}_C\bigl[x^{\text{rgb}}_t,\; x^{\text{mot}}_t\bigr]
$$

**含义**: 对 RGB 和运动潜变量分别施加 [[Flow Matching]] 线性插值扰动，拼接后共同输入 [[VideoDiT]] 进行联合去噪，使模型同时学习外观生成与运动预测。

**符号说明**:
- $t \sim \mathcal{U}(0, 1)$: [[Flow Matching]] 时间步
- $z^q$: 干净的潜变量（RGB 或运动）
- $\epsilon^q$: 高斯噪声
- $\text{Concat}_C$: 沿通道维度拼接

### 公式3: [[门控残差建模|门控残差注入]]

$$
\bar{h}_\ell = B_\ell(h_{\ell-1})
$$

$$
h_\ell = \bar{h}_\ell + \sum_{j \in \{\text{geo},\text{sem}\}} g^j_\ell(\bar{h}_\ell,\, c) \odot R^j_\ell(\bar{h}_\ell)
$$

$$
\hat{f}^j = P_j(h_L)
$$

**含义**: 几何和语义特征通过门控残差路径修正 [[VideoDiT]] 的隐藏状态，门控值自适应控制修正幅度，保护预训练视频路径不被破坏。

**符号说明**:
- $B_\ell$: 第 $\ell$ 层标准 Transformer 块
- $h_\ell$: 第 $\ell$ 层隐藏状态
- $g^j_\ell(\bar{h}_\ell, c)$: 条件化的可学习门控（标量/向量），$c$ 为条件信息
- $R^j_\ell$: view $j$ 的残差映射函数
- $\odot$: 逐元素乘法
- $P_j$: 从最终层隐藏状态预测目标特征 $f^j$ 的预测头
- $L$: 总层数

### 公式4: [[Flow Matching|训练总损失]]

$$
\mathcal{L} = \lambda_{\text{rgb}}\,\mathcal{L}^{\text{FM}}_{\text{rgb}} + \lambda_{\text{mot}}\,\mathcal{L}^{\text{FM}}_{\text{mot}} + \lambda_{\text{geo}}\,\mathcal{L}^{\text{pred}}_{\text{geo}} + \lambda_{\text{sem}}\,\mathcal{L}^{\text{pred}}_{\text{sem}} + \lambda_{\text{act}}\,\mathcal{L}^{\text{FM}}_{\text{act}} + \lambda_g\,\mathcal{L}_{\text{gate}}
$$

**含义**: 六项损失分别监督外观去噪（FM）、运动去噪（FM）、几何特征预测（L2）、语义特征预测（L2）、动作去噪（FM）和门控正则化，共同引导多视角未来表征学习。

**符号说明**:
- $\mathcal{L}^{\text{FM}}_{\text{rgb}}, \mathcal{L}^{\text{FM}}_{\text{mot}}, \mathcal{L}^{\text{FM}}_{\text{act}}$: [[Flow Matching]] 损失（预测速度场 $v_\theta$ 与目标速度的 MSE）
- $\mathcal{L}^{\text{pred}}_{\text{geo}}, \mathcal{L}^{\text{pred}}_{\text{sem}}$: 预测头对齐目标外部特征的 L2 损失
- $\mathcal{L}_{\text{gate}}$: 门控正则化（鼓励稀疏 / 小幅度修正）
- $\lambda_{\cdot}$: 各损失项的权重系数

---

## 关键图表

### Figure 1: 动机与四视角对比

![Figure 1](https://arxiv.org/html/2608.04996v1/x1.png)

**说明**: RGB-only [[WAM]] 将未来学习组织在视频外观空间，DreamWAM 额外学习运动（光流）、几何（深度）和语义（DINOv2 特征）三种互补视角。图中对比了四种视角对同一场景的不同表征方式，直观说明多视角可以提供更丰富的任务相关信号。

### Figure 2: DreamWAM 整体架构

![Figure 2](https://arxiv.org/html/2608.04996v1/x2.png)

**说明**: 训练时，[[VideoDiT]] 通过两条并行路径处理多视角信号：(a) RGB 与光流潜变量**联合去噪**（通道拼接）；(b) 几何和语义特征通过**门控残差分支**注入 VideoDiT 指定层。[[ActionDiT]] 通过共享注意力层与 VideoDiT 交互，从多视角未来表征中提取动作相关信息。推理时，蓝色标注的所有非 RGB 路径全部关闭。

### Figure 3: 真实机器人评估结果

![Figure 3](https://arxiv.org/html/2608.04996v1/x3.png)

**说明**: AgileX PiPER 双臂平台上，DreamWAM 在 4 个标准任务和 3 类视觉扰动（彩色背景斑块、光照变化、干扰物体）下的成功率对比。标准任务从 90.8% 提升至 96.7%；视觉扰动场景从 55.6% 大幅提升至 74.4%，体现了多视角训练带来的显著鲁棒性增益。

### Table 1: LIBERO 仿真基准（四套 10 任务）

| Method | Spatial | Object | Goal | Long | Avg. |
|--------|---------|--------|------|------|------|
| OpenVLA | 84.70 | 88.40 | 79.20 | 53.70 | 76.50 |
| π₀ | 96.80 | 98.80 | 95.80 | 85.20 | 94.15 |
| π₀.₅ | 98.80 | 98.20 | 98.00 | 92.40 | 96.85 |
| Motus | 96.80 | 99.80 | 96.60 | 97.60 | 97.70 |
| LingBot-VA | 98.50 | 99.60 | 97.20 | 98.50 | 98.45 |
| Fast-WAM（uncond） | 97.00 | 99.80 | 97.60 | 94.80 | 97.30 |
| **DreamWAM-uncond** | **99.20** | **99.60** | **98.40** | **96.40** | **98.40** |
| Fast-WAM-Joint | 99.40 | 98.20 | 98.80 | 95.60 | 98.00 |
| **DreamWAM** | **99.60** | **99.80** | **98.60** | **97.60** | **98.90** |

**关键发现**: DreamWAM 将无视频回滚模式的 [[Fast-WAM]] 成功率从 97.30% 提升至 98.40%，联合推理模式从 98.00% 提升至 98.90%，均达到当前 LIBERO 最佳水平。

### Table 2: LIBERO-Plus 分布外泛化（7 类视觉扰动）

| 类型 | 方法 | 预训练 | 相机 | 机器人外观 | 语言 | 光照 | 背景 | 噪声 | 物体布局 | 平均 |
|------|------|--------|------|-----------|------|------|------|------|---------|------|
| VLA | UniVLA | ✓ | 1.80 | 46.20 | 69.60 | 69.00 | 81.00 | 21.20 | 31.90 | 45.81 |
| VLA | OpenVLA-OFT | ✓ | 56.40 | 31.90 | 79.50 | 88.70 | 93.30 | 75.80 | 74.20 | 71.40 |
| VLA | π₀ | ✓ | 13.80 | 6.00 | 58.80 | 85.00 | 81.40 | 79.00 | 68.90 | 56.13 |
| VLA | π₀-FAST | ✓ | 65.10 | 21.60 | 61.00 | 73.20 | 73.20 | 74.40 | 68.80 | 62.47 |
| WAM | WorldVLA | ✓ | 0.10 | 27.90 | 41.60 | 43.70 | 17.10 | 10.90 | 38.00 | 25.61 |
| WAM | Fast-WAM | ✗ | 16.26 | 44.13 | 66.82 | 79.77 | 52.60 | 38.54 | 61.38 | 51.36 |
| WAM | **DreamWAM-uncond** | ✗ | **30.39** | **59.23** | **77.29** | **85.90** | **64.68** | **56.65** | **69.97** | **63.44** |
| WAM | Fast-WAM-Joint | ✗ | 39.59 | 60.90 | 92.32 | 94.57 | 57.62 | 58.59 | 80.52 | 69.16 |
| WAM | **DreamWAM** | ✗ | **53.78** | **63.61** | **94.80** | **96.67** | **71.56** | **67.15** | **80.72** | **75.47** |

**关键发现**: 无预训练数据增强的条件下，DreamWAM 在 7 种扰动上全面超越 [[Fast-WAM]]，无回滚模式平均 +12.08 点，联合推理模式平均 +6.31 点。相机视角扰动提升最大（+14.13 / +14.19），说明几何视角监督对相机不变性有显著帮助。

### Table 3: 消融实验（LIBERO，联合推理模式，D=去噪路径，R=残差路径）

| 配置 | 运动 | 几何 | 语义 | Spatial | Object | Goal | Long | 平均 |
|------|------|------|------|---------|--------|------|------|------|
| RGB only | – | – | – | 99.40 | 98.20 | 98.80 | 95.60 | 98.00 |
| Mot. only (残差R) | R | – | – | 99.00 | 98.20 | 98.60 | 95.60 | 97.85 |
| Mot. only (去噪D) | D | – | – | 98.60 | 99.80 | 98.20 | 97.40 | 98.50 |
| Geo. only (残差R) | – | R | – | 99.40 | 99.60 | 97.60 | 96.00 | 98.15 |
| Geo. only (去噪D) | – | D | – | 97.20 | 99.60 | 96.20 | 94.20 | 96.80 |
| Sem. only (残差R) | – | – | R | 99.00 | 97.40 | 99.20 | 96.80 | 98.10 |
| Sem. only (去噪D) | – | – | D | 96.80 | 98.80 | 97.40 | 94.20 | 96.80 |
| w/o 运动 (–/R/R) | – | R | R | 98.80 | 99.20 | 97.00 | 95.00 | 97.50 |
| w/o 几何 (D/–/R) | D | – | R | 98.40 | 99.00 | 98.40 | 97.20 | 98.25 |
| w/o 语义 (D/R/–) | D | R | – | 99.20 | 99.00 | 98.20 | 98.00 | 98.60 |
| 全去噪 D/D/D | D | D | D | 98.60 | 98.60 | 97.80 | 96.40 | 97.85 |
| **混合 D/R/R（最终）** | **D** | **R** | **R** | **99.60** | **99.80** | **98.60** | **97.60** | **98.90** |

**关键发现**:
1. **运动监督最关键**：去掉运动（–/R/R）导致最大性能下降（97.50%），证明帧间变化的显式编码是核心
2. **路径匹配表征形式**：几何/语义用去噪路径（D/D/D = 97.85%）反而低于 RGB-only 基线，而混合路径（D/R/R = 98.90%）显著更优；VAE 对齐的光流适合去噪，外部编码器特征适合残差注入
3. **视角互补**：单独任何视角的增益均小于全组合

---

## 实验

### 数据集与评估设置

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套 × 10 任务，每任务 500 demos | 标准仿真操作基准 | 主要训练+评估（每任务 50 rollout，共 2000） |
| [[LIBERO-Plus]] | 7 类视觉扰动 × 10 任务 | 无对应训练数据的分布外测试 | 鲁棒性评估 |
| 真实机器人 | 4 标准任务 + 3 类扰动，各 30 次 | AgileX PiPER 双臂平台 | 真实世界验证 |

**推理模式**:
- **无回滚（uncond）**: 第一帧输入，无视频前滚，对应原 Fast-WAM 无联合推理设置
- **联合推理（Joint）**: RGB 与动作联合去噪，对应 Fast-WAM-Joint 设置

### 实现细节

- **视频骨干**: [[Wan2.2]]-5B（[[VideoDiT]]）
- **动作骨干**: [[ActionDiT]]（paired expert）
- **光流估计**: [[RAFT]]（离线预计算）
- **深度估计**: [[Depth Anything 3|Depth Anything V3]]（离线预计算）
- **语义特征**: [[DINOv2]] ViT-B/14（离线预计算）
- **学习率**: $1 \times 10^{-5}$
- **Batch Size**: 16
- **训练轮数**: 5 epochs（真实机器人微调）
- **硬件**: 8× NVIDIA H20 GPU，bfloat16 精度

---

## 批判性思考

### 优点

1. **零推理开销**: 训练增强不增加推理复杂度，完全退化为 RGB-only 接口，工程友好性极高
2. **泛化增益显著**: 在不使用任何预训练数据的情况下（对比使用预训练的 VLA 方法），在 LIBERO-Plus 上取得最优；真实机器人扰动下 +18.8 个百分点
3. **路径设计有理论依据**: 根据特征与 VAE 的对齐程度选择去噪路径（光流）或残差路径（深度/语义），消融实验充分验证

### 局限性

1. **训练开销较高**: 需要为每段训练视频离线预计算 RAFT 光流、Depth Anything V3 深度、DINOv2 特征，数据准备管线复杂；扩展到大规模互联网视频时成本可观
2. **扰动类型有限**: LIBERO-Plus 的 7 类扰动偏向视觉外观层面，对语义扰动（如新类物体、全新任务语言描述）的鲁棒性未充分验证
3. **训练时依赖真值未来帧**: 光流/深度/语义均从真值未来帧提取（而非预测未来帧），这在数据收集阶段是自然的，但限制了模型无真值时的应用场景

### 潜在改进方向

1. 将辅助特征预计算替换为在线轻量估计，降低数据准备成本
2. 在更大规模、更多样化的数据集（如 OXE）上验证可扩展性
3. 探索更多视角（如触觉、力觉），进一步丰富 WAM 的未来表征

### 可复现性评估

- [x] 代码开源（https://github.com/hustvl/DreamWAM）
- [x] 预训练模型（随代码发布）
- [x] 训练细节完整（学习率、batch size、硬件）
- [x] 数据集可获取（LIBERO / LIBERO-Plus 均公开）

---

## 关联笔记

### 基于
- [[Fast-WAM]]: 直接基线，DreamWAM 在其两专家架构上扩展多视角训练
- [[Wan2.2]]: 提供 VideoDiT 骨干和 VAE 编码器
- [[Flow Matching]]: 所有去噪流的训练范式

### 对比
- [[Fast-WAM]]: 主要对比基线，RGB-only 设计
- [[LingBot-VA]]: LIBERO 强基线，有预训练加持
- [[Motus]]: 另一 WAM 方法，LIBERO 成功率接近

### 方法相关
- [[RAFT|RAFT 光流]]: 运动视角的特征提取器
- [[Depth Anything 3|Depth Anything V3]]: 几何视角的特征提取器
- [[DINOv2]]: 语义视角的特征提取器
- [[VideoDiT]]: 视频去噪 Transformer 骨干
- [[ActionDiT]]: 动作去噪专家
- [[光流 (Optical Flow)|光流]]: 运动表征的核心概念

### 数据集相关
- [[LIBERO]]: 主要训练和评估数据集
- [[LIBERO-Plus]]: 分布外鲁棒性评估基准

---

## 速查卡片

> [!summary] DreamWAM: Beyond RGB Future Prediction for World Action Models
> - **核心**: 训练时用运动/几何/语义四视角扩展 RGB 未来预测，推理时退化为 RGB-only 接口
> - **方法**: 光流联合去噪 + 深度/语义门控残差注入 [[VideoDiT]]，[[ActionDiT]] 共享注意力耦合
> - **结果**: LIBERO-Plus 无回滚 51.36%→63.44%；真实机器人扰动 55.6%→74.4%
> - **代码**: https://github.com/hustvl/DreamWAM

---

*笔记创建时间: 2026-08-07*
