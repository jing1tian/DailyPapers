---
title: "Mask2Real-WM: Segmentation Masks as a Sim-to-Real Bridge for Controllable Dexterous World Models"
method_name: "Mask2Real-WM"
authors: [Riccardo O. Feingold, Davide Liconti, Chenyu Yang, Robert K. Katzschmann]
year: 2026
venue: arXiv
tags: [world-model, dexterous-manipulation, sim-to-real, video-diffusion, robot-learning, segmentation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.04546v2
created: 2026-08-21
---

# 论文笔记：Mask2Real-WM

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | ETH Zurich |
| 日期 | August 2026 |
| 项目主页 | [srl-ethz.github.io/Mask2Real-WM](https://srl-ethz.github.io/Mask2Real-WM/) |
| 对比基线 | [[Ctrl-World]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.04546) / Code (未开源) |

---

## 一句话总结

> 以分割掩码为中间桥梁，将动力学预测（WM1）与光真实渲染（WM2）解耦，借助 50+ 小时仿真数据预训练，实现 23 维灵巧手全关节可控的世界模型。

---

## 核心贡献

1. **分割掩码 Sim-to-Real 桥梁**: 将 [[Sim-to-Real Gap]] 缩减问题转化为掩码空间建模，显著降低仿真与现实的域差。
2. **两阶段解耦架构**: WM1（掩码动力学）+ WM2（RGB 渲染）分别针对仿真大数据与现实小数据优化，相互独立可更新。
3. **23 维灵巧手动作可控性**: 基于 [[Stable Video Diffusion]] 骨干，通过双路 MLP 交叉注意力实现所有关节独立可控，ID 达 0.95、OOD 达 0.87。

---

## 问题背景

### 要解决的问题
为 [[Dexterous Manipulation|灵巧手操作]] 构建高维度（23-DoF）、可精确控制的动作条件式世界模型，支持机器人策略的离线评估与想象回放。

### 现有方法的局限
- 单体（monolithic）视频预测模型在 23 维动作空间中难以建立精确的动作-响应对应，输出模糊。
- 现实数据量稀缺（灵巧手演示成本高），直接在真实数据上训练泛化差。
- 仿真迁移到 RGB 空间存在巨大的 Sim-to-Real Gap，影响渲染质量。

### 本文的动机
分割掩码是一种"三类颜色词典"（手/绿色、物体/红色、背景/黑色），与 RGB 相比 sim-to-real gap 极小，仿真器可直接输出 ground-truth 掩码，因此可大规模预训练掌握动力学规律；之后用少量真实数据微调渲染模型即可获得高质量 RGB 输出。

---

## 方法详解

### 模型架构

Mask2Real-WM 采用**两阶段 [[Video Diffusion Model|视频扩散]]** 架构：

- **输入**: 过去 $k=5$ 帧的 RGB $I_{t-k:t}$ + 分割掩码 $m_{t-k:t}$ + 未来 $H=5$ 步动作 $a_{t-k:t+H}$
- **WM1 骨干**: [[Stable Video Diffusion]]（SVD U-Net），在分割掩码空间建模
- **WM2 骨干**: [[Stable Video Diffusion]]（SVD U-Net），在 RGB 空间建模
- **掩码调制**: [[ControlNet]] 模块 + 轻量 CNN 编码器（WM2 专用）
- **动作调制**: 双路 MLP → 1024 维帧嵌入 → 逐帧交叉注意力注入 SVD 每个 U-Net 块
- **参数高效微调**: [[LoRA]]（rank 16, α=16）
- **观测空间**: 双视角 RGB，分辨率 $135 \times 240$

### 核心模块

#### WM1：掩码动力学模型

**设计动机**: 利用 [[Sim-to-Real Gap]] 在掩码空间极小的特性，在 50+ 小时 [[IsaacLab]] 仿真数据上预训练，获得完整关节范围的动力学先验。

**具体实现**:
- 将双视角（$V=2$）历史帧的 [[VAE]] 潜变量与含噪未来帧沿高度维度拼接，张量形状 $(B, k+H, C, V \cdot H/f, W/f)$，$f=8$ 为 VAE 空间压缩比
- 6-DoF 末端执行器姿态和 17 个 [[ORCA Hand|ORCA]] 关节角分别由独立 MLP 映射为 1024 维嵌入后相加
- 嵌入通过逐帧交叉注意力注入所有 SVD U-Net 块
- 仿真预训练（55,000 步，lr=1e-4，单卡 H200）→ 真实数据微调（45,000 步，lr=5e-6，LoRA）

#### WM2：外观渲染模型

**设计动机**: 仅在约 2.5 小时真实数据上训练，通过 WM1 预测的掩码作为强空间约束，避免 RGB 级 Sim-to-Real 难题。

**具体实现**:
- 三路条件信号：① 过去 RGB 帧（SVD VAE 编码）；② WM1 预测掩码（轻量 CNN + [[ControlNet]] 注入）；③ 动作（同 WM1 双路 MLP）
- CNN 编码器：3 个卷积块（Conv2d + GroupNorm + SiLU）+ 步幅下采样 + 转置卷积 + 1×1 卷积，避免 VAE 对离散掩码边界的模糊
- [[ControlNet]] 在 SVD 编码器/中间块多分辨率层注入掩码特征（零初始化卷积）
- 训练（70,000 步，8×H100）：SVD 冻结 + [[LoRA]] 微调，CNN 和 [[ControlNet]] 全参数训练
- 独立掩码/动作 [[Classifier-Free Guidance|条件 dropout]]（各 10%，同时 dropout 5%）

#### 推理流程

1. [[SAM2|SAM 3]] 处理 5 帧上下文窗口获取初始分割掩码
2. WM1 以动作序列为条件自回归生成未来掩码
3. WM2 以预测掩码 + 过去 RGB + 动作为条件生成光真实 RGB
4. 长程展开：将最后一帧预测结果滑入上下文窗口，迭代执行

---

## 关键公式

### 公式1：[[Video Diffusion Model|观测空间定义]]

$$
o_t = (I_t,\, m_t),\quad I_t \in \mathbb{R}^{V \times 3 \times H \times W},\quad m_t \in \mathbb{R}^{V \times 3 \times H \times W}
$$

**含义**: 每一时刻的观测由 $V=2$ 视角的 RGB 图像 $I_t$ 和分割掩码 $m_t$ 组成，掩码使用三类颜色词典。

**符号说明**:
- $V=2$: 摄像机视角数（固定工作空间视角 + 腕部视角）
- $H \times W = 135 \times 240$: 图像分辨率
- $m_t$: 三类颜色掩码（手/绿、物体/红、背景/黑）

---

### 公式2：[[Dexterous Manipulation|动作空间定义]]

$$
a_t \in \mathbb{R}^{23} = [a^{\text{ee}}_t,\; a^{\text{hand}}_t],\quad a^{\text{ee}}_t \in \mathbb{R}^6,\quad a^{\text{hand}}_t \in \mathbb{R}^{17}
$$

**含义**: 23 维动作向量由 6 维笛卡尔末端执行器姿态和 17 维 ORCA 手关节角度拼接而成。

**符号说明**:
- $a^{\text{ee}}_t \in \mathbb{R}^6$: [[Franka Emika Panda]] 末端执行器笛卡尔姿态（位置 + 旋转）
- $a^{\text{hand}}_t \in \mathbb{R}^{17}$: [[ORCA Hand]] 17 个关节位置

---

### 公式3：[[Video Diffusion Model|条件视频预测目标]]

$$
p(I_{t+1:t+H} \mid I_{t-k:t},\, m_{t-k:t},\, a_{t-k:t+H})
$$

**含义**: 给定过去 $k$ 帧 RGB + 掩码以及未来 $H$ 步动作，预测未来 $H$ 帧 RGB 图像序列。

**符号说明**:
- $k=5$: 上下文帧数
- $H=5$: 预测视野（预测步长）

---

### 公式4：[[Video Diffusion Model|联合分布分解]]（核心设计）

$$
p(I, m \mid I_{\text{past}}, m_{\text{past}}, a) = \underbrace{p(m \mid m_{\text{past}}, a)}_{\text{WM1：掩码动力学}} \cdot \underbrace{p(I \mid I_{\text{past}}, m, a)}_{\text{WM2：RGB 渲染}}
$$

**含义**: 将像素预测分解为两个独立条件分布：WM1 预测掩码，WM2 以预测掩码为条件生成 RGB，解耦仿真预训练与真实渲染。

---

### 公式5：[[Stable Video Diffusion|WM1 扩散损失]]

$$
\mathcal{L}_{\text{WM1}} = \mathbb{E}_{x,\varepsilon,\sigma}\left[\left\|\hat{x}_\theta\!\left(x_\sigma;\, m_{t-k:t},\, a_{t-k:t+H},\, \sigma\right) - x\right\|_2^2\right]
$$

**含义**: 以过去掩码和未来动作为条件，对 SVD 骨干进行噪声预测训练，最小化预测去噪目标与真实未来掩码的 MSE。

**符号说明**:
- $x$: 未来掩码的 VAE 潜变量（目标）
- $x_\sigma = x + \sigma \varepsilon$: 加噪潜变量，$\varepsilon \sim \mathcal{N}(0,I)$
- $\hat{x}_\theta$: WM1 去噪网络（SVD U-Net），参数 $\theta$
- $\sigma$: 扩散噪声水平

---

### 公式6：[[Stable Video Diffusion|WM2 扩散损失]]

$$
\mathcal{L}_{\text{WM2}} = \mathbb{E}_{x,\varepsilon,\sigma}\left[\left\|\hat{x}_\phi\!\left(x_\sigma;\, I_{t-k:t},\, m_{t-k:t+H},\, a_{t-k:t+H},\, \sigma\right) - x\right\|_2^2\right]
$$

**含义**: 以过去 RGB、未来掩码和动作为三路条件，训练 WM2 在 RGB 空间的去噪目标，使渲染具备掩码空间约束和动作条件控制。

**符号说明**:
- $x$: 未来 RGB 帧的 VAE 潜变量（目标）
- $\hat{x}_\phi$: WM2 去噪网络（SVD U-Net + [[ControlNet]]），参数 $\phi$
- $m_{t-k:t+H}$: 由 WM1 预测的未来掩码（推理时）或 ground-truth 掩码（训练时）

---

## 关键图表

### Figure 1：可控性对比 Teaser

![Figure 1](https://arxiv.org/html/2607.04546v2/teaser_controllability.png)

**说明**: 对比 Mask2Real-WM 与 monolithic 基线（[[Ctrl-World]]）在独立关节正弦命令下的可控性。Mask2Real-WM 清晰跟随每维动作命令，而基线产生模糊输出且多维耦合。

---

### Figure 2：方法架构总览

![Figure 2](https://arxiv.org/html/2607.04546v2/method_dexwm.png)

**说明**: 两阶段架构示意。左侧 WM1（掩码动力学）以仿真数据预训练：接收历史掩码 + 动作，输出未来掩码；右侧 WM2（RGB 渲染）以真实数据训练：接收历史 RGB + WM1 预测掩码 + 动作，通过 [[ControlNet]] 注入掩码特征生成光真实视频。

---

### Figure 3：仿真数据流水线

![Figure 3a - MimicGen 演示](https://arxiv.org/html/2607.04546v2/figures/mimicgen_image_crop.png)

![Figure 3b - 随机探索运动](https://arxiv.org/html/2607.04546v2/figures/random_motion_crop.png)

**说明**: 仿真数据由三类生成器产生（见 Table 4），保证 23 个关节的全运动范围覆盖。MimicGen 生成干净/带噪演示，随机运动序列进一步扩大状态覆盖。

---

### Figure 4：动作可控性结果

![Figure 4](https://arxiv.org/html/2607.04546v2/controllability.png)

**说明**: 对 23 个动作维度逐一施加正弦信号（其余固定），评估每维的独立响应。Mask2Real-WM 完整模型（仿真预训练 + 真实微调）在 ID 达 0.95、OOD 达 0.87，全面优于所有消融版本和基线。

---

### Figure 5：WM1 不同训练配置下的感知质量

![Figure 5](https://arxiv.org/html/2607.04546v2/results_perceptual.png)

**说明**: 对比三种 WM1 训练配置（仅真实数据 / 仅仿真数据 / 仿真+真实）在四种测试划分上的 PSNR/SSIM/LPIPS。仅真实数据在 ID 像素指标最高但 OOD 退化严重；仿真+真实（本文）在 ID 与 OOD 间取得最佳平衡。

---

### Figure 9：长程展开对比（50 帧）

![Figure 9 - Ours](https://arxiv.org/html/2607.04546v2/figures/long_horizon/ours_strip.png)

![Figure 9 - Baseline](https://arxiv.org/html/2607.04546v2/figures/long_horizon/baseline_strip.png)

**说明**: 自回归展开 50 帧对比。本文方法保持清晰的手部和物体边界；[[Ctrl-World]] 基线随帧数增加逐渐模糊，物体细节丢失。

---

### Figure 10：每维动作可控性细节（23 维）

![Figure 10](https://arxiv.org/html/2607.04546v2/figures/action_components.png)

**说明**: 逐维可控性分数热力图。灵巧手 17 个关节中大多数在仿真预训练后得到改善，尤其是 OOD 场景下的手指独立控制。

---

### Figure 20：真实机器人平台

![Figure 20](https://arxiv.org/html/2607.04546v2/figures/robot_setup.png)

**说明**: [[Franka Emika Panda]]（7-DoF 手臂）+ [[ORCA Hand]]（17-DoF 灵巧手）实验平台，配备 Rokoko 手套遥操作，双同步摄像机（固定工作空间视角 + 腕部视角）。

---

### Table 1：Fréchet 视频距离（FVD，RGB 空间）

| 测试划分 | Baseline (Ctrl-World) | WM1 仅真实 | WM1 仅仿真 | WM1 仿真+真实（本文） |
|----------|-----------------------|-----------|-----------|----------------------|
| ID (n=150) | 98.4 | 73.2 | 211.6 | **95.1** |
| OOD 综合 (n=150) | 351.0 | 406.2 | 410.2 | **386.3** |

**说明**: 仅真实数据的 WM1 ID-FVD 最低（过拟合外观），但 OOD 退化最严重。本文方法 ID 与基线相当，OOD 最优。

---

### Table 2：WM1 掩码空间质量（PSNR/LPIPS/mIoU）

| 测试划分 | WM1 训练配置 | PSNR ↑ | LPIPS ↓ | mIoU ↑ |
|----------|------------|--------|---------|--------|
| ID | 仅真实 | 22.00 | 0.119 | 0.621 |
| ID | 仅仿真 | 19.03 | 0.229 | 0.361 |
| ID | 仿真+真实（本文） | 20.72 | 0.158 | **0.517** |
| OOD | 仅真实 | 20.48 | 0.147 | 0.488 |
| OOD | 仅仿真 | 20.33 | 0.184 | 0.391 |
| OOD | 仿真+真实（本文） | **21.74** | **0.122** | **0.503** |

**关键发现**: 仿真+真实版本在 OOD 上全面最优（PSNR+1.26、LPIPS-0.025、mIoU+0.015 vs. 仅真实数据）。

---

### Table 3：未见物体泛化（Banana / Cup / Cylinder）

| 物体 | 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | WM1 seg SSIM ↑ |
|------|------|--------|--------|---------|----------------|
| Banana | Baseline | 19.40 | 0.613 | 0.202 | - |
| | **Ours (WM2)** | 17.30 | 0.564 | 0.238 | - |
| | **Ours (WM1 seg)** | - | - | - | **0.860** |
| Cup | Baseline | 19.01 | 0.618 | 0.200 | - |
| | **Ours (WM2)** | 16.65 | 0.581 | 0.224 | - |
| | **Ours (WM1 seg)** | - | - | - | **0.867** |
| Cylinder | Baseline | 19.06 | 0.616 | 0.194 | - |
| | **Ours (WM2)** | 16.92 | 0.579 | 0.221 | - |
| | **Ours (WM1 seg)** | - | - | - | **0.866** |

**说明**: WM1 掩码质量对未见物体保持高 SSIM（0.86+），优于 RGB 基线（不含掩码）；WM2 RGB 指标在未见物体上略低于基线，因外观微调仅见过红色方块。

---

### Table 4：仿真数据集组成

| 生成器类型 | Episode 数 |
|-----------|-----------|
| MimicGen（干净演示） | 5,000 |
| MimicGen（正弦噪声） | 5,000 |
| 随机运动探索 | 3,700 |
| 遥操作演示 | 250 |
| **合计** | **13,950** |

---

### Table 5：训练超参数

| 超参数 | WM1（仿真） | WM1（真实微调） | WM2 |
|--------|-----------|----------------|-----|
| 训练步数 | 55,000 | 45,000 | 70,000 |
| 学习率 | 1e-4 | 5e-6 | 1e-4 |
| LR 调度 | Cosine | Cosine | Cosine |
| Warm-up 步数 | 3,000 | 500 | 3,000 |
| Batch Size | 64 | 64 | 72 |
| GPU | 1×H200 | 1×H200 | 8×H100 |
| 精度 | BF16 | BF16 | FP16 |
| [[LoRA]] rank/α | - | 16/16 | 16/16 |
| 掩码 dropout | 10% | 10% | 10% |
| 动作 dropout | 10% | 10% | 10% |
| 同时 dropout | 5% | 5% | 5% |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[IsaacLab]] 仿真数据 | 13,950 episodes，50+ 小时 | ground-truth 掩码，完整关节范围，5 种物体 | WM1 预训练 |
| 真实机器人演示 | 218 episodes，约 2.5 小时 | Rokoko 遥操作，[[SAM2\|SAM 3]] 自动标注掩码 | WM1 微调 + WM2 训练 |

### 实验平台

- **机器人**: [[Franka Emika Panda]] (7-DoF) + [[ORCA Hand]] (17-DoF) = 24 总自由度
- **动作空间**: 23 维（6 维笛卡尔 EE + 17 维手关节）
- **系统辨识**: [[CMA-ES]] 优化 PD 增益和摩擦系数，使仿真轨迹匹配真实

### 评估指标

- **动作可控性**: 逐维正弦激励，3 分制手工评分（1=独立响应，0.5=有耦合，0=无响应），均值报告
- **感知质量**: PSNR ↑、SSIM ↑、LPIPS ↓
- **分布质量**: Fréchet 视频距离（FVD）↓
- **掩码质量**: mIoU ↑、PSNR ↑

### 测试划分

| 划分 | 描述 |
|------|------|
| ID（In-Distribution） | 保留的真实数据集内测试 episode |
| No Object | 工作台不含物体（训练中未见） |
| Random Play | 自由探索动作轨迹（不同动作统计） |
| Background | 旋转竞技场、未见光照条件 |

---

## 批判性思考

### 优点
1. **桥梁设计简洁有效**: 三类颜色掩码方案极简，sim-to-real gap 量化证明收益显著（OOD LPIPS 降低 25%+）。
2. **模块化可维护**: WM1 可独立用于动力学检验，WM2 可针对新场景仅重训 [[ControlNet]] 模块，降低部署成本。
3. **数据效率**: 50 小时仿真 + 2.5 小时真实数据，相比需要大量真实数据的单体模型具有明显工程优势。

### 局限性
1. **掩码遮挡问题**: WM1 缺乏深度信息，手遮挡物体时物体消失；长程展开中 3 类颜色掩码无法区分多个同类物体，导致身份漂移。
2. **可控性评估粗糙**: 3 分制量表、单一评分者、每维仅 5 个 ID/5 个 OOD 样本，统计可信度有限。
3. **WM2 外观泛化未测试**: 渲染模型仅在单一竞技场训练，未评估新场景下的外观迁移能力。
4. **仿真数据偏差**: 训练语料以正弦轨迹为主，收益部分可能来自评测时使用相同波形族，而非通用可控性提升。
5. **推理速度慢**: 两次序列扩散 pass 计算代价高，作者提出用 Consistency Model 蒸馏加速，但未在本文实现。

### 潜在改进方向
1. 引入深度图作为第四类掩码通道，解决遮挡问题。
2. 使用实例分割（多颜色 ID）替代语义掩码，支持多物体场景。
3. 对可控性评估引入定量、自动化协议（如频谱相干性）。
4. 对 WM2 进行更多场景的外观微调研究。

### 可复现性评估
- [ ] 代码开源（未开源）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参数表格完整）
- [ ] 数据集可获取（真实演示数据未公开）

---

## 关联笔记

### 基于
- [[Stable Video Diffusion]]: WM1/WM2 的视频扩散骨干网络
- [[ControlNet]]: WM2 中掩码条件注入模块
- [[MimicGen]]: 仿真数据生成的主要工具

### 对比
- [[Ctrl-World]]: 主要基线，单体 SVD 基视频预测，无仿真预训练
- [[DexWM]]: 基于 DINOv2 特征空间的灵巧手世界模型

### 方法相关
- [[LoRA]]: 骨干网络参数高效微调
- [[Classifier-Free Guidance]]: 掩码/动作条件 dropout 训练策略
- [[Video Diffusion Model]]: 整体生成范式
- [[Sim-to-Real Gap]]: 本文核心解决的问题

### 硬件/数据相关
- [[ORCA Hand]]: 17-DoF 灵巧手平台
- [[Franka Emika Panda]]: 7-DoF 机械臂
- [[IsaacLab]]: 仿真数据采集平台
- [[SAM2]]: SAM 3 前身，用于掩码自动标注

---

## 速查卡片

> [!summary] Mask2Real-WM
> - **核心**: 分割掩码作为 Sim-to-Real 桥梁，解耦掩码动力学（WM1）与 RGB 渲染（WM2）
> - **方法**: WM1 在 50h 仿真预训练 → WM2 在 2.5h 真实数据训练，双路 MLP 注入 23 维动作
> - **结果**: 动作可控性 0.95 ID / 0.87 OOD，优于单体基线（0.60 / 0.44）
> - **代码**: 暂未开源（https://srl-ethz.github.io/Mask2Real-WM/）

---

*笔记创建时间: 2026-08-21*
