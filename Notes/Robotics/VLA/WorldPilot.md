---
title: "World Pilot: Steering Vision-Language-Action Models with World-Action Priors"
method_name: "WorldPilot"
authors: [Zefu Lin, Rongxu Cui, Junjia Xu, Xiaojuan Jin, Wenling Li, Lue Fan, Zhaoxiang Zhang]
year: 2026
venue: arXiv
tags: [vla, world-model, flow-matching, latent-steering, robot-manipulation, ood-generalization]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.12403
created: 2026-06-12
---

# 论文笔记：World Pilot: Steering Vision-Language-Action Models with World-Action Priors

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | CASIA, Nanjing University, Beihang University |
| 日期 | June 2026 |
| 项目主页 | [world-pilot.github.io](https://world-pilot.github.io/) |
| 对比基线 | [[Cosmos-Policy]], [[ABot-M0]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.12403) / Code: N/A |

---

## 一句话总结

> World Pilot 通过将 [[世界-动作模型（WAM）|World-Action Model]] 的场景演化潜在和预期轨迹以双路径（潜在引导 + 动作引导）注入 VLA 决策链，实现 LIBERO-Plus 零样本 OOD 84.7% SOTA。

---

## 核心贡献

1. **双路径先验注入框架**: 提出将 [[WAM（World-Action Model）|WAM]] 输出通过 Latent Steering 和 Action Steering 两条互补路径接入 [[VLA（视觉-语言-动作模型）|VLA]]，无需重新设计主干
2. **潜在空间引导（Latent Steering）**: 用 [[Cross-Attention|交叉注意力]] 将场景演化潜在 $\mathbf{Z}^w_t$ 直接注入 VLM 隐藏状态，避免解码图像时的像素级噪声干扰
3. **轨迹级先验压缩（Action Steering）**: 将预期轨迹 $\widetilde{\mathbf{A}}^w_t$ 压缩为单个前缀 token，供 [[Flow Matching|流匹配]] 动作生成器使用，优于逐步条件化

---

## 问题背景

### 要解决的问题

标准 [[VLA（视觉-语言-动作模型）|VLA]] 方法将图像和语言编码后直接预测动作，缺乏对场景未来演化的建模。在分布外（OOD）场景（相机视角变化、外观扰动、布局变化等）下，VLA 性能大幅下降，因为它只能感知当前帧而无法预判后续动态。

### 现有方法的局限

- **Cosmos Policy** 等将世界模型与 VLA 融合时，依赖解码的未来图像帧作为先验，但解码图像引入像素级无关信息（纹理、光照、背景），稀释了动作相关信号
- 部分方法将预期轨迹逐步注入（per-step conditioning），导致与世界模型预测过度耦合，泛化性受限
- WAM 冻结与否的设计不统一，缺乏模块化

### 本文的动机

将 WAM 的输出保留在**潜在空间**（而非解码为像素），通过轻量融合模块注入 VLA，既保留场景演化的语义信息，又避免像素噪声；同时将轨迹先验压缩为单个 token，实现轨迹级而非步级的软约束。

---

## 方法详解

### 模型架构

![Figure 1: World Pilot Overview](https://arxiv.org/html/2606.12403v1/x1.png)

**说明**: World Pilot 通过两条先验路径将 [[WAM（World-Action Model）|WAM]] 输出接入 [[VLA（视觉-语言-动作模型）|VLA]] 决策链。语义路径（VLM 编码图像 + 语言）保持不变，Latent Steering 将场景演化潜在注入 VLM 隐藏状态，Action Steering 将预期轨迹压缩为前缀 token 送入动作生成器。

World Pilot 采用**双路径增强**架构：
- **输入**: 语言指令 $\ell$ + 视觉观测 $\mathbf{O}_t$ + 本体感知状态 $\mathbf{q}_t$（可选）
- **Backbone**: [[Qwen3-VL]]（VLM 主干）+ [[Cosmos-Policy|Cosmos Policy]]（WAM，冻结）
- **核心模块**: [[Cross-Attention|动态交叉注意力]] 用于 Latent Steering；[[Flow Matching|流匹配]] 动作生成器接收 Action Steering 前缀 token
- **输出**: [[Action Chunking|动作块]] $\hat{\mathbf{A}}_{\theta,t} = (a_t, \ldots, a_{t+K-1})$
- **WAM 参数冻结**: 训练时仅更新 VLA 参数 $\theta$ 及轻量融合模块

### 核心模块

#### 模块1: World-Action Model (WAM)

**设计动机**: 利用 [[视频预训练|视频预训练大模型]] 的场景预见能力为 VLA 提供未来感知先验

**具体实现**:
- 使用 [[Cosmos-Policy|Cosmos Policy]] 作为 WAM（5步去噪），联合预测场景演化潜在 $\mathbf{Z}^w_t$ 和预期轨迹 $\widetilde{\mathbf{A}}^w_t$
- WAM 参数 $\phi$ 在训练期间**完全冻结**，不参与反向传播
- WAM 每步推理增加一次前向传播开销

#### 模块2: Latent Steering（潜在引导）

**设计动机**: 将场景演化语义信息注入 [[VLM|视觉-语言模型]] 的感知层，而不引入像素级噪声

**具体实现**:
- 动态编码器 $f_\text{dyn}$ 将 WAM 潜在 $\mathbf{Z}^w_t$ 映射为动态特征，叠加时间嵌入 $\boldsymbol{\rho}_\text{fut}$ 得到 $\mathbf{D}^w_t$
- 通过 [[Cross-Attention|交叉注意力]] 将 $\mathbf{D}^w_t$ 注入 VLM 隐藏状态 $\mathbf{H}_t$，得到增强后的 $\bar{\mathbf{H}}_t$
- 与直接解码未来图像相比，潜在注入避免了纹理、光照等无关信息对动作预测的干扰（实验验证：+1.2% vs 解码图像）

#### 模块3: Action Steering（动作引导）

**设计动机**: 为 [[Flow Matching|流匹配]] 动作生成器提供轨迹级运动先验，避免逐步硬约束导致过拟合

**具体实现**:
- 对 WAM 预期轨迹 $\widetilde{\mathbf{A}}^w_t$ 用 $\text{Align}_K$ 重采样至目标时间步数 $K$
- 动作编码器 $f_\text{act}$ 将对齐后的轨迹压缩为单个前缀 token $\mathbf{s}^w_t$
- 前缀 token 作为软约束注入生成器，与状态 token $\mathbf{u}_t$、查询 token $\mathbf{Q}_t$、噪声轨迹 $\mathbf{X}_{\tau,t}$ 拼接后输入
- 单 token 优于逐步 token（84.7% vs 83.6%，见 Table 6）

---

## 关键公式

### 公式1: [[WAM（World-Action Model）|WAM 联合预测]]

$$
(\mathbf{Z}^{w}_{t},\, \widetilde{\mathbf{A}}^{w}_{t}) = W_{\phi}(\mathbf{O}_{t},\, \ell,\, \mathbf{q}_{t})
$$

**含义**: WAM $W_\phi$ 以当前视觉观测、语言指令和本体状态为输入，联合预测场景演化潜在和预期动作轨迹

**符号说明**:
- $\mathbf{O}_t$: 时刻 $t$ 的视觉观测（单/多视角图像）
- $\ell$: 语言指令
- $\mathbf{q}_t$: 本体感知状态（关节角度等，可选）
- $\mathbf{Z}^w_t$: 场景演化潜在，编码未来场景动态
- $\widetilde{\mathbf{A}}^w_t$: WAM 预期动作轨迹（轨迹先验）
- $\phi$: WAM 参数（冻结）

### 公式2: [[Cross-Attention|Latent Steering 动态增强]]

$$
\mathbf{D}^{w}_{t} = f_{\text{dyn}}(\mathbf{Z}^{w}_{t}) + \boldsymbol{\rho}_{\text{fut}}
$$

**含义**: 动态编码器将 WAM 潜在映射为动态特征，并叠加未来场景时间嵌入

**符号说明**:
- $f_\text{dyn}$: 轻量动态编码器（可训练）
- $\boldsymbol{\rho}_\text{fut}$: 标记"未来场景"的时间嵌入向量
- $\mathbf{D}^w_t$: 处理后的场景演化动态特征

### 公式3: [[Cross-Attention|VLM 隐藏状态增强]]

$$
\bar{\mathbf{H}}_{t} = \mathbf{H}_{t} + \operatorname{CrossAttn}(\mathbf{H}_{t},\, \mathbf{D}^{w}_{t})
$$

**含义**: 通过残差交叉注意力将场景动态先验注入 VLM 隐藏状态，增强感知层对未来场景的感知

**符号说明**:
- $\mathbf{H}_t \in \mathbb{R}^{L \times d}$: VLM 原始隐藏状态（序列长度 $L$，特征维度 $d$）
- $\operatorname{CrossAttn}(\cdot, \cdot)$: 以 $\mathbf{H}_t$ 为 Query、$\mathbf{D}^w_t$ 为 Key/Value 的交叉注意力
- $\bar{\mathbf{H}}_t$: 动态增强后的隐藏状态

### 公式4: [[Flow Matching|Action Steering 轨迹压缩]]

$$
\mathbf{s}^{w}_{t} = f_{\text{act}}\!\left(\operatorname{Align}_{K}(\widetilde{\mathbf{A}}^{w}_{t})\right)
$$

**含义**: 将 WAM 预期轨迹重采样对齐后压缩为单个轨迹先验 token，作为动作生成器的软约束前缀

**符号说明**:
- $\operatorname{Align}_K$: 轨迹时间步重采样函数，对齐到目标动作块长度 $K$
- $f_\text{act}$: 动作编码器（可训练）
- $\mathbf{s}^w_t \in \mathbb{R}^d$: 单个轨迹级先验 token

### 公式5: [[Flow Matching|流匹配动作生成]]

$$
\hat{\mathbf{A}}_{\theta,t} = g_{\theta}\!\left(\mathbf{X}_{\tau,t},\, \tau,\, \mathbf{u}_{t},\, \mathbf{s}^{w}_{t},\, \mathbf{Q}_{t} \;\middle|\; \bar{\mathbf{H}}_{t}\right)
$$

$$
\mathbf{X}_{\tau,t} = \tau\, \mathbf{A}^{\star}_{t} + (1-\tau)\,\boldsymbol{\epsilon}
$$

**含义**: 流匹配生成器以噪声轨迹、流时间、状态 token、轨迹先验 token 和查询 token 为输入（以增强隐藏状态为条件），预测目标动作块

**符号说明**:
- $g_\theta$: 流匹配动作生成器（可训练）
- $\mathbf{X}_{\tau,t}$: 流时间 $\tau$ 处的含噪轨迹（线性插值）
- $\mathbf{A}^{\star}_t$: 专家示范动作块（训练标签）
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$: 高斯噪声
- $\tau \sim \mathcal{U}(0,1)$: 流时间（训练时采样）
- $\mathbf{u}_t$: 可选本体状态 token
- $\mathbf{Q}_t$: 可学习的未来查询 token

### 公式6: [[Flow Matching|World Pilot 训练目标]]

$$
\mathcal{L}_{\text{World Pilot}} = \mathbb{E}_{\tau, \boldsymbol{\epsilon}}\!\left[\, w(\tau)\,\left\|\hat{\mathbf{A}}_{\theta,t} - \mathbf{A}^{\star}_{t}\right\|_{2}^{2}\,\right], \quad w(\tau) = \frac{1}{(1-\tau)^{2}}
$$

**含义**: 以速度空间重加权的 Flow Matching 回归损失，对高流时间（接近数据分布）的预测误差给予更大权重

**符号说明**:
- $w(\tau) = \frac{1}{(1-\tau)^2}$: 时间重加权函数（速度空间损失等价形式）
- $\mathbf{A}^{\star}_t$: 专家动作块监督标签
- 仅 VLA 参数 $\theta$ 和融合模块参与优化，WAM 参数 $\phi$ 冻结

---

## 关键图表

### Figure 1: 系统概览

![Figure 1](https://arxiv.org/html/2606.12403v1/x1.png)

**说明**: World Pilot 在标准 VLA 决策链上叠加两条 WAM 先验路径。Latent Steering 将场景演化潜在 $\mathbf{Z}^w_t$ 注入 VLM 隐藏状态，Action Steering 将预期轨迹 $\widetilde{\mathbf{A}}^w_t$ 压缩为前缀 token 送入动作生成器。

### Figure 2: 模型架构详图

![Figure 2](https://arxiv.org/html/2606.12403v1/x2.png)

**说明**: 详细展示语义路径（[[Qwen3-VL]] 编码图像与语言）与两条先验路径的交互方式。动态编码器 $f_\text{dyn}$ 和动作编码器 $f_\text{act}$ 是主要的可训练融合模块；[[Flow Matching|流匹配]] 生成器同时接收来自两条路径的先验信号。

### Figure 3: 真实机器人实验设置

![Figure 3](https://arxiv.org/html/2606.12403v1/x3.png)

**说明**: 左侧为机器人平台，中间为训练分布内场景，右侧为四类 OOD 场景变体（外观变化、几何变化、可变形物体状态、位姿变化）。ID 到 OOD 的性能下降仅 10-20 个百分点，显著优于基线的 25-50 个百分点。

### Table 1: LIBERO-Plus 零样本 OOD Benchmark

| 方法 | Camera | Robot | Language | Light | Background | Noise | Layout | **Total** |
|------|--------|-------|----------|-------|------------|-------|--------|-----------|
| RIPT-VLA | — | — | — | — | — | — | — | 68.4% |
| Cosmos Policy | — | — | — | — | — | — | — | 79.7% |
| ABot-M0 | — | — | — | — | — | — | — | 80.5% |
| Being-H0.7 | — | — | — | — | — | — | — | 82.1% |
| **World Pilot** | **82.8%** | **60.6%** | **87.2%** | **98.6%** | **96.4%** | **93.6%** | **80.5%** | **84.7%** |

**关键发现**: World Pilot 在 Camera 轴（+13.2 pp vs 次优）和外观扰动轴取得最大优势，体现了 Latent Steering 对视角/外观变化的强鲁棒性。

### Table 2: 真实机器人四任务实验

| 任务 | Stack Blocks ID | Stack Blocks OOD | Fold Towel ID | Fold Towel OOD | Fruit-to-Plate ID | Fruit-to-Plate OOD | Container-Lid ID | Container-Lid OOD |
|------|-----------------|------------------|---------------|----------------|-------------------|--------------------|------------------|-------------------|
| World Pilot | **70%** | **50%** | **85%** | **70%** | **90%** | **70%** | **80%** | **65%** |
| 基线最大 OOD 下降 | — | -50% | — | -45% | — | -50% | — | -50% |
| World Pilot OOD 下降 | — | **-20%** | — | **-15%** | — | **-20%** | — | **-15%** |

**关键发现**: World Pilot 的 ID→OOD 性能下降仅为 10-20 pp，基线方法下降 25-50 pp，OOD 鲁棒性显著提升。

### Table 3: 消融 — 各路径贡献

| 配置 | LIBERO-Plus Total | Δ vs 基线 |
|------|-------------------|----------|
| 基线（无引导） | 80.5% | — |
| 仅 Latent Steering | 83.7% | +3.2 |
| 仅 Action Steering | 83.1% | +2.6 |
| **完整 World Pilot** | **84.7%** | **+4.2** |

**关键发现**: 两条路径均有独立贡献，合并后效果优于单独使用任一路径，且存在协同效应（4.2 > 3.2 + 2.6 的独立相加并不成立，但均为正向贡献）。

### Table 4: 消融 — 世界模型迁移性

| WAM 配置 | LIBERO-Plus Total |
|----------|-------------------|
| Cosmos-Predict（无动作后训练） | 82.6% |
| Cosmos Policy（有动作后训练） | **83.7%** |

**关键发现**: 即使使用未经动作后训练的纯视频生成模型（Cosmos-Predict），场景演化先验依然有效（+2.1%），说明框架对 WAM 选择具备泛化性。

### Table 5: 消融 — 潜在表示形式

| 先验表示 | LIBERO-Plus Total |
|----------|-------------------|
| 1步潜在注入 | 84.5% |
| 3步潜在注入 | 84.7% |
| 5步潜在注入 | **84.7%** |
| 解码未来图像 | 83.5% |

**关键发现**: 潜在直接注入（任意步数）均优于解码图像（-1.2%），验证了避免像素级干扰的核心设计选择。

### Table 6: 消融 — 轨迹编码方式

| 编码方式 | LIBERO-Plus Total |
|----------|-------------------|
| 原始轨迹 | 83.0% |
| 流匹配初始化 | 84.1% |
| 逐步 token | 83.6% |
| **单 token（最优）** | **84.7%** |

**关键发现**: 单 token 轨迹压缩效果最优，过度详细的逐步条件化反而限制了泛化（83.6% < 84.7%）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO-Plus | 10,030 个扰动任务 | 7 个 OOD 扰动轴（相机/机器人/语言/光照/背景/噪声/布局） | 零样本 OOD 测试 |
| RoboCasa | — | 日常厨房长地平线操作任务 | OOD 泛化测试 |
| Real-Robot | 100 个遥操演示/任务 | 4 任务×ID+2 OOD 变体，每设置 20 次试验 | 真实世界验证 |

### 实现细节

- **VLM 主干**: [[Qwen3-VL]]
- **世界模型**: [[Cosmos-Policy|Cosmos Policy]]（5步去噪，完全冻结）
- **训练正则化**: 0.3 dropout 率
- **梯度范围**: 仅限 VLA 参数 $\theta$ + 融合模块（$f_\text{dyn}$, $f_\text{act}$）
- **硬件**: 8× RTX PRO 6000 GPU
- **真实机器人**: 从 100 个遥操演示微调（每任务）

### 可视化结果

真实机器人实验（Figure 3）显示，在几何变化（积木堆叠高度）、可变形物体（新颜色毛巾）、布局变化（水果摆放）、位姿变化（容器盖对准）等 OOD 场景下，World Pilot 均能保持合理成功率，基线方法则大幅退化。

---

## 批判性思考

### 优点

1. **设计解耦清晰**: WAM 冻结 + 轻量融合模块，模块化程度高，易于替换不同 WAM
2. **潜在优于像素的论证充分**: Table 5 消融实验直接对比了潜在注入与解码图像，结论可信
3. **单 token 压缩优雅**: 避免了逐步条件化带来的过拟合，同时保留了轨迹级语义
4. **泛化性好**: 即使使用无动作后训练的 Cosmos-Predict，框架依然有效（Table 4）

### 局限性

1. **WAM 覆盖受限**: WAM 的训练数据分布约束了其对极端 OOD 场景的预见能力
2. **LIBERO-Plus 各轴不均衡**: Robot 轴（60.6%）远低于其他轴，说明机器人形态变化仍是难点
3. **额外推理开销**: 每步多一次 WAM 前向传播，实时控制延迟增加
4. **OOD 仍有下降**: 真实机器人 ID→OOD 仍有 10-20 pp 下降，未完全解决分布外泛化

### 潜在改进方向

1. 针对 Robot 轴（形态变化）专项增强 WAM 或引入具身感知先验
2. 蒸馏 WAM 以降低推理延迟，支持更高频控制
3. 探索多 WAM 融合以覆盖更广的 OOD 场景分布

### 可复现性评估

- [ ] 代码开源（项目主页暂未提供代码链接）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（论文中有较完整的超参数描述）
- [x] 数据集可获取（LIBERO-Plus 和 RoboCasa 均公开）

---

## 关联笔记

### 基于

- [[Cosmos-Policy]]: 作为 WAM 提供场景演化潜在和预期轨迹
- [[VLA（视觉-语言-动作模型）]]: 被增强的基础框架
- [[Qwen3-VL]]: VLM 主干网络
- [[Flow Matching]]: 动作生成器核心技术

### 对比

- [[Cosmos-Policy]]: 直接使用解码图像作为先验，World Pilot 改为潜在注入
- [[ABot-M0]]: LIBERO-Plus 强基线，World Pilot 超越 4.2%
- [[DreamZero]]: 另一类世界模型辅助 VLA 方法

### 方法相关

- [[WAM（World-Action Model）]]: World Pilot 依赖的核心先验来源
- [[Cross-Attention]]: Latent Steering 的核心操作
- [[Action Chunking]]: 输出动作块的基本范式
- [[Flow Matching]]: 动作生成器使用的生成框架

### 硬件/数据相关

- [[LIBERO-Plus]]: 主要 OOD 评测 benchmark
- [[RoboCasa]]: 厨房操作评测环境

---

## 速查卡片

> [!summary] World Pilot (arXiv 2606.12403)
> - **核心**: 用 WAM 的场景演化潜在 + 预期轨迹，通过双路径（Latent Steering + Action Steering）增强 VLA
> - **方法**: 潜在交叉注意力注入 VLM 隐藏状态；轨迹压缩为单 prefix token 进入 Flow Matching 生成器
> - **结果**: LIBERO-Plus 84.7%（SOTA），真实机器人 OOD 下降仅 10-20 pp
> - **代码**: [world-pilot.github.io](https://world-pilot.github.io/)（代码暂未公开）

---

*笔记创建时间: 2026-06-12*
