---
title: "UniFlowWAM: A Unified World-Action Model with Visual Semantics and 4D Geometry"
method_name: "UniFlowWAM"
authors: [Jialin Tian, Zhuofan Li, Xinkai Wang, Bokai Lin, Chenyi Wang, Xinyu Zhan, Cewu Lu, Lixin Yang]
year: 2025
venue: arXiv
tags: [world-action-model, scene-flow, diffusion-transformer, bimanual-manipulation, multi-modal, 4d-geometry, robot-policy]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.30991v1
created: 2026-10-24
---

# 论文笔记：UniFlowWAM: A Unified World-Action Model with Visual Semantics and 4D Geometry

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Shanghai Jiao Tong University (SJTU), Shanghai Innovation Institute, Noematrix |
| 日期 | October 2025 |
| 项目主页 | [kksham111.github.io/UniFlowWAM_ProjectPage](https://kksham111.github.io/UniFlowWAM_ProjectPage/) |
| 对比基线 | [[LDA]], [[Pi05]], [[Fast-WAM]], [[4D-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.30991) / Code coming soon |

---

## 一句话总结

> UniFlowWAM 将[[场景流|3D 场景流]]作为第三模态，与视觉和动作在同一个三专家扩散 Transformer 中联合去噪，通过 10 种角色条件化目标实现统一策略、世界模型和逆/前向动力学，在双臂操作任务上大幅超越基线。

---

## 核心贡献

1. **3D 场景流作为第三模态**: 将[[场景流 (Scene Flow)|3D 场景流]]（稠密点位移场）引入 [[World Action Model]] 框架，与视觉语义特征和动作并列为三大模态之一，无需深度传感器（单目估计）。
2. **三专家 [[MM-DiT]] 架构**: 每个模态（动作 $a$、场景流 $m$、未来视觉特征 $v$）拥有独立专家，通过 [[Joint Self-Attention]] 共享信息，并用 [[MM-RoPE|MM-RoPE（多模态旋转位置编码）]] 在统一时空坐标系下对齐位置。
3. **10 种角色条件化目标**: 每个模态在每个目标下可处于"目标"、"条件"、"缺席"三种角色，10 种组合覆盖策略、视觉预测、逆/前向动力学等，单一模型兼具多种世界模型能力，测试时仅用策略目标无需流输入。

---

## 问题背景

### 要解决的问题

现有 [[World Action Model]] 框架（如 [[Fast-WAM]]、[[4D-WAM]]、[[DINO-WM]]）缺乏统一的几何 + 语义双模态世界表示。[[Optical Flow|光流]]只有 2D，3D 流估计有噪声，如何将 4D 几何信息有效融入策略学习并提高零样本泛化能力，仍是未解问题。

### 现有方法的局限

- **纯语义世界模型**（如 DINO-WM）：缺乏精细几何信息，对遮挡和新视角鲁棒性差
- **光流策略**（如 [[FlowVLA]]、LaMP）：2D 光流缺乏深度感知，无法表达真实 3D 运动
- **独立流模块**：流估计与策略训练解耦，无法互相增益
- **单目标预训练**（如 [[LDA]]）：预训练目标单一，世界模型能力弱

### 本文的动机

将 3D 场景流作为与视觉和动作**并列**的第三模态，在一个统一扩散 Transformer 中通过联合去噪共同训练，可以让几何感知和语义理解互相增益；10 种角色条件化目标使模型天然具备逆动力学（因果推断）和前向动力学（想象）能力，从而提升在分布外扰动下的泛化。

---

## 方法详解

### 模型架构

UniFlowWAM 采用 **三专家 [[MM-DiT|多模态扩散 Transformer]]** 架构：

- **输入上下文**: 语言指令 $\ell$ + 当前观测 $o_t$ + 本体感觉状态 $s_t$，由 [[Qwen3-VL]] (4B) 编码
- **三个模态专家**:
  - 动作专家 $a$：动作块 $a_{t:t+H}$，地平线 $H=16$ 步
  - 流专家 $m$：4D 流潜码，由 [[MotionCrafter 4D VAE|冻结 4D VAE]] 编码
  - 视觉专家 $v$：地平线末帧的冻结 [[DINO|DINOv3]] 特征
- **联合注意力**: [[Joint Self-Attention]] 让每个专家的查询跨三个专家的键值进行注意力
- **条件注入**: 每个专家通过 [[AdaLN]] 注入扩散时间步 + 目标嵌入；Qwen3-VL 特征通过 cross-attention 输入
- **总参数**: LDA-1B 基础上增加流专家（流专家从头训练）

### 核心模块

#### 模块1: 4D 场景流表示

**设计动机**: 用 [[场景流 (Scene Flow)|3D 场景流]] 代替 2D 光流，提供深度感知的几何信息；[[MotionCrafter 4D VAE]] 压缩高维流场为可去噪潜码。

**具体实现**:
- 离线用 Track4World 跟踪器提取伪标签：每帧像素 $u$ 的 3D 点 $P_i(u)$ 和帧间位移 $F_i(u)$
- 用 SAM 分割保留前景流，过滤背景噪声
- 双流编码：

$$
m = [\mathcal{E}_P(P_{t:t+H});\ \mathcal{E}_F(F_{t:t+H})]
$$

  两个冻结编码器沿通道维度拼接，解码器 $\mathcal{D}$ 恢复 $P$ 和 $F$
- 流 patchifier 每 token 合并两帧潜码，第 $t'$ 个流 token 的时间步取 $t = 2t'$

#### 模块2: 多模态旋转位置编码（[[MM-RoPE]]）

**设计动机**: 不同模态 token 需要在统一的 $(t, h, w)$ 时空坐标系中对齐位置，避免各专家位置编码不一致。

**具体实现**:
- [[RoPE]] 在共享 $(t, h, w)$ 坐标系旋转查询和键
- 视觉 token: $t = H$（末帧），$(h, w)$ 取 patch 空间位置
- 动作/状态 token: $t$ 取步骤索引，$(h, w) = (0, 0)$
- 流 token: $t = 2t'$（合并两帧潜码）
- 基础模型的可学习绝对嵌入保留，与 MM-RoPE 叠加使用

#### 模块3: 角色条件化多目标训练

**设计动机**: 单一策略目标利用率低；多个角色组合让模型兼具策略、世界模型、逆/前向动力学能力，共享参数提升泛化。

**具体实现**:
- 每个目标 $r$ 中，每个模态处于"目标"（加噪去噪）、"条件"（输入干净值）或"缺席"（占位符，无损失）三种角色
- 目标嵌入加入时间步嵌入后通过 [[AdaLN]] 注入
- 任务均衡采样器确保每批次所有 10 个目标均出现

---

## 关键公式

### 公式1: [[Flow Matching|流匹配]]目标函数

$$
\mathcal{L} = \sum_{k \in \mathcal{T}(r)} \mathcal{L}_{\text{FM}}^{(k)}, \quad k \in \{a, m, v\}
$$

**含义**: 总损失为当前目标 $r$ 所有"目标模态"集合 $\mathcal{T}(r)$ 中各模态的流匹配损失之和

**符号说明**:
- $r$: 目标索引（共 10 个目标）
- $\mathcal{T}(r)$: 目标 $r$ 的目标模态集合
- $\mathcal{L}_{\text{FM}}^{(k)}$: 模态 $k$ 的流匹配速度预测损失
- $k \in \{a, m, v\}$: 动作、3D 流、未来视觉特征三个模态

### 公式2: [[场景流 (Scene Flow)|4D 场景流潜码]]编码

$$
m = [\mathcal{E}_P(P_{t:t+H});\ \mathcal{E}_F(F_{t:t+H})]
$$

**含义**: 将时域点图序列和帧间位移序列分别编码后沿通道拼接，得到压缩的 4D 流潜码

**符号说明**:
- $P_{t:t+H}$: 时间窗口内各帧像素的 3D 点图序列（由 Track4World 估计）
- $F_{t:t+H}$: 帧间 3D 位移序列
- $\mathcal{E}_P, \mathcal{E}_F$: 冻结的 [[MotionCrafter 4D VAE|MotionCrafter 4D VAE 编码器]]（分别编码点图和位移）
- $[\cdot;\cdot]$: 沿通道维度拼接

### 公式3: 多专家联合注意力

对每个模态 $k \in \{a, m, v\}$，各专家独立计算 QKV 后跨模态联合注意力：

$$
(Q^k, K^k, V^k) = W^k \hat{x}^k, \quad \text{Attn}^k = \text{softmax}\!\left(\frac{Q^k [K^a, K^m, K^v]^\top}{\sqrt{d}}\right)[V^a, V^m, V^v]
$$

**含义**: 每个专家的查询对所有三个专家的键值进行注意力，实现跨模态信息交换

**符号说明**:
- $\hat{x}^k$: 模态 $k$ 的归一化输入 token
- $W^k$: 专家 $k$ 独有的 QKV 投影权重
- $[K^a, K^m, K^v]$: 三个专家键矩阵的拼接
- 查询-键归一化（QK-norm）约束 logit 幅值防止训练不稳定

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1 - UniFlowWAM Overview](https://arxiv.org/html/2610.30991v1/UniFlowWAM-teaser.png)

**说明**: UniFlowWAM 整体框架：视觉特征 $v$、3D 场景流 $m$ 和动作 $a$ 在一个三专家 [[MM-DiT]] 中联合去噪；10 种目标角色网格覆盖策略和世界模型能力；测试时以纯策略模式部署，无需流输入。

### Figure 2: Model Architecture / 模型架构

![Figure 2 - UniFlowWAM Pipeline](https://arxiv.org/html/2610.30991v1/UniFlowWAM-pipeline.png)

**说明**: 三模态架构细节。当前观测和未来观测经 [[Qwen3-VL]] 编码为视觉上下文；点图和场景流经 [[MotionCrafter 4D VAE]] 编码为流潜码 $m$；本体感觉与动作块组成动作序列。三路专家通过 [[Joint Self-Attention]] 交换信息，[[MM-RoPE]] 统一时空位置编码，[[AdaLN]] 注入目标嵌入和扩散时间步。

### Figure 3: Real-World Tasks / 真实世界任务

![Figure 3 - Real-World Tasks](https://arxiv.org/html/2610.30991v1/ppt-3-aligned.png)

**说明**: 四个双臂 ALOHA 风格任务（AgileX Cobot Magic，双 PiPER 臂，14-D 状态和动作），每任务 50 次演示和分阶段里程碑评分：
- **卷心菜转移**：抓取→放入锅→抓锅盖→盖锅
- **辣椒转移**：抓取绿/红辣椒→放盘
- **钢笔操作**：两支笔放入抽屉→关抽屉
- **毛巾折叠**：对折→四折

### Figure 4: Real-World Results / 真实世界结果

（原文为柱状图，无外链图片）

**说明**: UniFlowWAM 在所有任务上均优于基线（50 次试验）：

| 策略 | 成功率 (%) | 过程得分 |
|------|-----------|---------|
| **UniFlowWAM** | **87.0** | **92.3** |
| π0.5 | 68.5 | 77.0 |
| DreamZero | 30.0 | 34.5 |
| LDA | 15.5 | 17.8 |

### Figure 5: Demonstration Coverage / 演示覆盖率分析

![Figure 5 - Coverage Analysis](https://arxiv.org/html/2610.30991v1/pen_ood_figure.png)

**说明**: 演示覆盖分析。左：辣椒操作演示的旋转角度分布（−90°到60°，大多数聚集在0°附近纵向摆放）；右：钢笔操作初始位置分布（俯视图，红框为所有49次演示范围，蓝带为稀疏边界区域）。UniFlowWAM 在90°侧向辣椒（最难）成功率80%，政策基准仅20%。

### Table 1: 目标角色网格

| # | 目标 | 条件 | 缺席 | 目标模态 |
|---|------|------|------|---------|
| 1 | 直接策略 † | – | m, v | a |
| 2 | 视觉预测 † | a, m | – | v |
| 3 | 流预测 | a, v | – | m |
| 4 | 逆动力学 † | v | m | a |
| 5 | 流条件策略 | m | v | a |
| 6 | 前向动力学 † | a | m | v |
| 7 | 从流预测视觉 | m | a | v |
| 8 | 视觉流估计 | v | a | m |
| 9 | 动作驱动流生成 | a | v | m |
| 10 | 双条件逆动力学 | m, v | – | a |

注：† 为基础模型（LDA）的四个原有目标；上下文 $c_t = (o_t, s_t, \ell)$ 为所有目标共享。

### Table 2: RoboCasa-GR1 基准对比（成功率 %）

| 模型 | 成功率 (%) |
|------|-----------|
| UWM | 14.2 |
| GR00T-N1.6 | 47.6 |
| StarVLA | 47.8 |
| Being-H0.7 | 49.2 |
| FLARE | 55.0 |
| LDA-1B | 55.4 |
| **UniFlowWAM** | **58.9**（最佳种子） |
| OpenWAM-α | 60.5 |

重复评估（11 个随机种子，均值 ± 标准差）：

| 模型 | 均值 ± 标准差 (%) |
|------|-----------------|
| LDA（发布版） | 52.11 ± 2.24 |
| LDA（复现） | 52.76 ± 1.16 |
| UniFlowWAM w/o 中期训练 | 54.84 ± 1.00 |
| **UniFlowWAM w/ 中期训练** | **56.38 ± 1.47** |

中期训练相比 LDA 发布版提升 **+4.27 pp**；在 LDA 低于 50% 的 10 个任务上平均提升 **+6.47 pp**。

### Table 3: LIBERO 和 LIBERO-Plus 结果（%）

| 模型 | LIBERO | 相机 | 机器人初始 | 语言 | 光照 | 背景 | 噪声 | 布局 | **总计** |
|------|--------|------|---------|------|------|------|------|------|--------|
| OpenVLA-OFT | 97.1 | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0.5 | 96.9 | 72.2 | 76.8 | 85.1 | 96.2 | 95.5 | 87.0 | 86.9 | 84.7 |
| StarVLA-α | – | 48.7 | 63.4 | 86.8 | 95.8 | 94.6 | 75.0 | 80.2 | 77.8 |
| Fast-WAM | 97.6 | 40.0 | 65.3 | 68.8 | 91.4 | 58.2 | 63.8 | 75.5 | 65.3 |
| 4D-WAM | 98.6 | 45.2 | 64.3 | 90.6 | 94.3 | 57.7 | 69.1 | 79.2 | 71.0 |
| FLEX-π | 98.5 | 73.7 | 53.1 | 84.3 | 91.9 | 93.8 | 93.4 | 82.9 | 80.9 |
| OpenWAM-α | 99.3 | 33.8 | 76.1 | 88.0 | 97.0 | 87.1 | 39.8 | 77.5 | 69.2 |
| LaMP | 98.3 | 64.5 | 69.6 | 88.2 | 95.3 | 97.4 | 76.9 | 73.8 | 79.3 |
| Track4Action | 97.0 | 76.6 | 60.7 | 87.1 | 96.3 | 96.1 | 88.4 | 78.7 | 82.3 |
| LDA | 92.8 | 33.1 | 71.2 | 76.0 | 89.9 | 91.1 | 72.0 | 66.2 | 69.5 |
| **UniFlowWAM** | **98.1** | **96.9** | 66.0 | 87.4 | **98.0** | 87.3 | **98.1** | 81.7 | **87.7** |

UniFlowWAM 在相机扰动类别（+20.3 pp vs. 第二名）和噪声类别（+4.7 pp）上优势最大，总计 **87.7%** 为最优。

### Table 4: 消融实验（目标集对实际成功率的影响，真实世界 ALOHA，%）

| 训练目标 | 辣椒 | 钢笔 | 毛巾 |
|---------|------|------|------|
| 仅策略 (obj-1) | 76 | 76 | 72 |
| 无流目标 (obj-4) | 92 | 80 | 76 |
| 流仅作条件 (obj-7) | 92 | 88 | 76 |
| **完整 UniFlowWAM (obj-10)** | **94** | **90** | **84** |

**关键发现**: 完整目标集比仅策略基准高 12–18 pp；辣椒侧向（最难）90° 情形下 obj-1 仅 1/5，obj-10 达 80%，说明多目标训练显著提升几何泛化。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboCasa]] | 24 任务 × 1000 demos | GR-1 双臂，29 自由度，tabletop | 仿真微调 + 评估 |
| [[LIBERO]] | 40 任务（四套件） | Spatial/Object/Goal/Long | 仿真微调 + 评估 |
| [[LIBERO-Plus]] | 10,030 demos，7 扰动轴 | 相机/机器人/语言/光照/背景/噪声/布局 | 零样本泛化评估 |
| RoboCOIN（中期训练） | 47,050 episodes，6 平台 | 双臂，多具身 | 预训练流专家 |
| RoboMIND 2.0（中期训练） | 140,231 episodes | Franka + UR 臂 | 预训练流专家 |
| MolmoAct2-BimanualYAM | 27,162 episodes | YAM 臂 | 预训练流专家 |
| **中期训练合计** | **214,443 episodes，133M 帧** | 10 具身，约 42M 训练窗口 | 流专家中期训练 |

真实世界：AgileX Cobot Magic（双 PiPER 臂，14-D 状态/动作），每任务 50 次演示。

### 实现细节

- **基础模型**: LDA-1B（视觉 + 动作专家预训练）；流专家从头训练
- **VLM**: Qwen3-VL-4B（冻结，用于指令和当前帧编码）
- **流编码**: Track4World 伪标签 + [[MotionCrafter 4D VAE|冻结 MotionCrafter 4D VAE]]；SAM 前景分割
- **视觉特征**: 冻结 DINOv3
- **硬件**: 48 × NVIDIA H200，全局 batch size 1,536
- **RoboCasa 微调**: 500k 步，cosine 调度，5k warmup；动作模型 lr=1e-4，VLM lr=1e-5
- **LIBERO 微调**: 默认 120k 步（消融扩展到 200k 步）
- **推理**: 策略模式，16 步动作块；**测试时不需要流输入**

### 可视化结果

真实世界最困难的"辣椒侧向（90°）"情形（演示中仅 ~10% 覆盖）：UniFlowWAM 成功率 80%，策略基准仅 20%，说明 4D 流表示显著提升空间泛化能力。

---

## 批判性思考

### 优点

1. **无深度传感器的 3D 几何感知**: 单目流估计避免硬件依赖，泛化更广
2. **测试时无需流输入**: 流仅在训练中作为辅助监督，推理时不增加延迟或传感器需求
3. **10 目标联合训练**: 多角色设计天然赋予逆/前向动力学能力，零样本扰动泛化显著提升
4. **大规模中期训练验证有效**: 214k episodes 流数据提升 RoboCasa +4.3 pp，证明流预训练的可扩展性

### 局限性

1. **流标签噪声**: Track4World 伪标签存在时序不一致，SAM 前景分割不完美，可能引入错误监督
2. **流预训练规模受限**: 约 1,200 小时的演示数据远小于 LDA 的 EI-30k 语料库；更大规模联合预训练留待未来工作
3. **未开源代码**: 目前代码未发布，可复现性待验证

### 潜在改进方向

1. 使用更强的视频 3D 流估计器替换 Track4World，减少伪标签噪声
2. 扩大跨具身流中期训练规模，探索流预训练的 scaling law
3. 扩展到更多模态（触觉、力传感器）以进一步丰富世界模型表示

### 可复现性评估

- [ ] 代码开源（coming soon）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（超参数、数据集均有描述）
- [x] 数据集可获取（LIBERO/RoboCasa 公开，RoboCOIN/RoboMIND 公开）

---

## 关联笔记

### 基于

- [[LDA]]: 基础模型，提供视觉和动作专家初始化
- [[MM-DiT]]: 多模态扩散 Transformer 架构基础
- [[FlowVLA]]: 流条件策略的先驱工作之一

### 对比

- [[Fast-WAM]]: 世界模型基线，LIBERO 65.3% vs UniFlowWAM 87.7%
- [[4D-WAM]]: 4D 几何世界模型基线，与本文直接对比
- [[Pi05]]: VLA 基线，真实世界成功率 68.5% vs UniFlowWAM 87.0%
- [[DINO-WM]]: 纯语义世界模型，无 3D 流

### 方法相关

- [[场景流 (Scene Flow)]]: 核心世界表示
- [[MM-RoPE]]: 多模态时空位置编码
- [[Joint Self-Attention]]: 三专家信息交换机制
- [[Flow Matching]]: 去噪目标函数
- [[MotionCrafter 4D VAE]]: 流潜码编码器/解码器

### 硬件/数据相关

- [[RoboCasa]]: 仿真评估基准
- [[LIBERO]]: 仿真评估基准
- [[LIBERO-Plus]]: 零样本泛化评估基准
- [[Qwen3-VL]]: VLM 视觉语言编码器

---

## 速查卡片

> [!summary] UniFlowWAM
> - **核心**: 3D 场景流作为第三模态，三专家 MM-DiT 联合去噪视觉 + 流 + 动作
> - **方法**: 10 种角色条件化目标（策略 + 前/逆向动力学 + 流预测），MM-RoPE 统一时空位置，测试时无流
> - **结果**: LIBERO-Plus 87.7%（SOTA），RoboCasa 56.4%（+4.3 pp vs LDA），真实双臂 87.0% vs π0.5 68.5%
> - **代码**: Coming soon — https://kksham111.github.io/UniFlowWAM_ProjectPage/

---

*笔记创建时间: 2026-10-24*
