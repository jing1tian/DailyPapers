---
title: "Unified Visuomotor Targets: Supervising VLAs Beyond Physical Actions"
method_name: "UVT"
authors: [Zhenyang Feng, Unnat Jain]
year: 2026
venue: IROS 2026
tags: [vla, vision-language-action, imitation-learning, latent-action, supervision-target, multimodal-vae, bimanual-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.03563
created: 2026-08-06
---

# 论文笔记：Unified Visuomotor Targets: Supervising VLAs Beyond Physical Actions

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Department of Computer Science, University of California, Irvine |
| 日期 | August 2026 |
| 项目主页 | https://unified-visuomotor-targets.github.io |
| 对比基线 | [[VLA-Adapter]], [[π₀.₅]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.03563) / Code N/A |

---

## 一句话总结

> UVT 将机器人动作块与视觉动态码融合为统一的潜在监督目标，无需改动 [[视觉语言动作模型|VLA]] 架构，显著加速训练收敛并提升鲁棒性。

---

## 核心贡献

1. **统一视觉运动目标（UVT）**: 通过多模态 [[MVAE]] 将机器人动作块 $a_t$ 与 [[LAM]] 提取的视觉动态码 $z_t$ 融合为 32 维潜在目标 $u_t$，同时编码"机器人如何运动"与"场景如何变化"两类信息
2. **无架构修改的监督替换**: VLA 策略网络直接预测 $u_t$ 而非原始动作，可训练解码器在推理时将 $u_t$ 还原为可执行动作；对骨干网络和策略头零改动，兼容回归式和流匹配式 VLA
3. **训练效率与鲁棒性双提升**: 在 LIBERO 和 LIBERO-Plus 基准上，10k 步时平均成功率分别提升 +8.5% 和 +23.5%；在三项真实双臂操作任务上成功率全面提升

---

## 问题背景

### 要解决的问题

[[视觉语言动作模型|VLA]] 以视觉-语言模型（VLM）为骨干，但监督信号是低层运动控制信号（关节角度等动作块）。VLM 的高层语义表示与低层运动指令之间存在**表示层鸿沟**，导致收敛慢、鲁棒性差。

### 现有方法的局限

- **离散化动作 token**（如 RT-2）：将关节值 bin 化为词元，损失连续性信息
- **连续动作回归头**（MLP、Transformer Decoder）：直接回归低层运动信号，与 VLM 特征空间不匹配
- **扩散/流匹配策略头**（[[π₀.₅]] 等）：迭代去噪改善样本质量，但监督目标本身仍是原始动作
- **[[潜在动作]]模型**（[[LAPA]]、[[UniVLA]]）：作为预训练目标使用，未在下游 fine-tuning 阶段提供结构化监督

所有方法都在问"架构怎么适配动作目标"，而非"动作目标本身该是什么"。

### 本文的动机

[[逆动力学模型|潜在动作模型（LAM）]]从图像对 $(v_t, v_{t+k})$ 提取的视觉动态码 $z_t$ 具有**任务感知的场景变化信息**，与 VLM 的语义空间更对齐。若将 $z_t$ 与运动动作块 $a_t$ 融合为统一表示，可同时保留运动精度与场景语义，为 VLA 提供更易学习的监督目标。

---

## 方法详解

### 模型架构

UVT 采用**两阶段框架**：

- **阶段一（离线 MVAE 训练）**: 输入演示数据中的动作块 $a_t$ 和冻结 [[LAM]] 提取的动态码 $z_t$，训练多模态 VAE（[[MVAE]]），输出预计算的统一视觉运动目标 $u_t \in \mathbb{R}^{32}$
- **阶段二（VLA 微调）**: VLA 骨干不变，策略头预测 $\hat{u}_t$，可训练解码器将 $\hat{u}_t$ 映射回可执行动作；联合优化 UVT 对齐损失和动作重建损失
- **输入**: 多模态观测 $x_t$（RGB 图像 + 语言指令）
- **Backbone**: 任意 VLA（实验中使用 [[VLA-Adapter]] 和 [[π₀.₅]]）
- **输出**: 通过解码器 $g^a(\hat{u}_t)$ 得到可执行动作块 $\hat{a}_t \in \mathbb{R}^{k \times d}$

### 核心模块

#### 模块 1: 多模态 VAE（[[MVAE|UVT 构建器]]）

**设计动机**: 利用 [[Product of Experts|精度加权融合（Product-of-Experts）]] 将两个异构模态（连续动作、离散动态码）融合为单一高斯分布，既保留各模态信息，又形成结构化的预测空间。

**具体实现**:
- **动作编码器** $f^a$：将连续动作块 $a_t \in \mathbb{R}^{k \times d}$ 映射为高斯后验 $(\mu_t^a, \log \sigma_t^{2,a})$
- **动态码编码器** $f^z$：将离散动态码 $z_t$ 映射为高斯后验 $(\mu_t^z, \log \sigma_t^{2,z})$
- **精度加权融合**：以可学习权重 $\alpha_a, \alpha_z$ 对两模态精度（方差倒数）加权平均，高确定性维度贡献更大
- **UVT 目标** $u_t = \mu_t$（融合后验的均值）
- **动作解码器** $g^a$：从 $u_t$ 重建动作块（L1 损失）
- **动态码解码器** $g^z$：从 $u_t$ 重建离散码（交叉熵损失）
- 双解码器强制 $u_t$ 同时保留运动控制信息和场景动态信息

#### 模块 2: VLA 微调与可训练动作解码

**设计动机**: 预计算的 $u_t$ 作为固定目标监督 VLA，但预测的 $\hat{u}_t$ 分布可能偏离 $u_t$，需允许解码器自适应适配。

**具体实现**:
- VLA 策略头输出 32 维向量 $\hat{u}_t$，与预计算 $u_t$ 做 L2 对齐
- MVAE 解码器 $g^a$ 在 VLA 训练期间**保持可训练**，接受 $\hat{u}_t$ 输入
- 联合 [[行为克隆]] 监督：对齐损失 + $\gamma=0.5$ 加权的动作重建损失
- 兼容回归式策略头（[[VLA-Adapter]]）和流匹配式策略头（[[π₀.₅]]）

---

## 关键公式

### 公式 1: [[LAM|LAM 动态码提取]]

$$
z_t = \text{LAM}(v_t, v_{t+k})
$$

**含义**: 冻结的预训练[[逆动力学模型|潜在动作模型]]从当前帧 $v_t$ 和未来帧 $v_{t+k}$ 提取离散动态码，描述动作窗口内的场景变化，无需任何机器人运动信息

**符号说明**:
- $v_t, v_{t+k}$: 时刻 $t$ 和 $t+k$ 的视觉观测帧
- $z_t$: 离散动态码，捕获任务相关的场景变化

---

### 公式 2: [[MVAE|MVAE 编码器]]

$$
(\mu_t^a, \log \sigma_t^{2,a}) = f^a(a_t), \quad (\mu_t^z, \log \sigma_t^{2,z}) = f^z(z_t)
$$

**含义**: 两个独立编码器分别将动作块和动态码映射为 UVT 空间中的对角高斯分布，为后续精度加权融合提供输入

**符号说明**:
- $\mu_t^a, \mu_t^z$: 动作/动态模态在 UVT 空间的高斯均值
- $\sigma_t^{2,a}, \sigma_t^{2,z}$: 各维度方差，反映各模态的不确定性

---

### 公式 3: [[Product of Experts|精度加权融合（Product-of-Experts）]]

$$
\tau_t = \alpha_a \tau_t^a + \alpha_z \tau_t^z, \qquad \tau = \frac{1}{\sigma^2}
$$

$$
\mu_t = \frac{\alpha_a \tau_t^a \odot \mu_t^a + \alpha_z \tau_t^z \odot \mu_t^z}{\tau_t}, \quad u_t = \mu_t
$$

**含义**: 将两模态的精度（方差倒数）加权求和得到融合精度；融合均值 $u_t$ 是以精度为权重的加权平均，高确定性维度（低方差）在融合中贡献更大；$\alpha_a, \alpha_z$ 是可学习的全局平衡标量

**符号说明**:
- $\tau_t^a = 1/\sigma_t^{2,a}$: 动作模态各维度精度（elementwise）
- $\tau_t^z = 1/\sigma_t^{2,z}$: 动态码模态各维度精度（elementwise）
- $\alpha_a, \alpha_z$: 可学习标量，平衡两模态的全局贡献
- $\odot$: 逐元素乘法
- $u_t \in \mathbb{R}^{32}$: 统一视觉运动目标（UVT）

---

### 公式 4: [[MVAE|MVAE 训练损失]]

$$
\mathcal{L}_{\text{ACT}} = \|a_t - \hat{a}_t\|_1, \quad \mathcal{L}_{\text{DYN}} = \text{CE}(z_t, \hat{z}_t)
$$

$$
\mathcal{L}_{\text{KL}} = D_{\text{KL}}\!\left(\mathcal{N}(\mu_t,\,\text{diag}(\sigma_t^2))\;\|\;\mathcal{N}(0, I)\right)
$$

$$
\mathcal{L}_{\text{UVT}} = \mathcal{L}_{\text{ACT}} + \mathcal{L}_{\text{DYN}} + \mathcal{L}_{\text{KL}}
$$

**含义**: 三项损失分别监督动作重建（L1）、离散动态码重建（交叉熵）、UVT 分布向标准正态的 [[KL 散度]] 正则化；三者之和使 MVAE 学到同时编码运动控制和视觉动态的结构化潜在空间

**符号说明**:
- $\hat{a}_t = g^a(u_t)$: 解码器重建的动作块
- $\hat{z}_t = g^z(u_t)$: 解码器重建的动态码 logits
- $\mathcal{L}_{\text{KL}}$: 将 UVT 空间正则化为标准正态，保证预测空间良态

---

### 公式 5: [[VLA-Adapter|VLA 微调损失]]

$$
\hat{u}_t = f_\theta(x_t)
$$

$$
\mathcal{L}_{\text{UVT-align}} = \|\hat{u}_t - u_t\|_2^2, \quad \mathcal{L}_{\text{dec}} = \|g^a(\hat{u}_t) - a_t\|_1
$$

$$
\mathcal{L}_{\text{VLA}} = \mathcal{L}_{\text{UVT-align}} + \gamma\,\mathcal{L}_{\text{dec}}, \quad \gamma = 0.5
$$

**含义**: VLA 策略头预测 $\hat{u}_t$，UVT 对齐损失锚定策略的内部表示到预计算目标空间；可训练解码器损失 $\mathcal{L}_{\text{dec}}$ 使解码器自适应适配预测 $\hat{u}_t$ 的分布，确保输出可执行动作

**符号说明**:
- $f_\theta(x_t)$: VLA 策略网络（骨干参数为 $\theta$），输出 32 维向量
- $u_t$: 离线预计算的 UVT 目标（固定，不参与梯度）
- $g^a(\hat{u}_t)$: 解码器将预测 UVT 映射为动作块
- $\gamma = 0.5$: 平衡 UVT 对齐与动作重建的超参数

---

## 关键图表

### Figure 1: UVT 整体框架

![Figure 1](https://arxiv.org/html/2608.03563v1/x1.png)

**说明**: 上半部分（UVT 提取）：冻结 [[LAM]] 从未来帧提取离散动态码 $z_t$；[[动作分块|动作块]] $a_t$ 与 $z_t$ 经各自可训练编码器映射为高斯后验；通过可学习精度权重 $\alpha_a, \alpha_z$ 的 [[Product of Experts|Product-of-Experts]] 融合为 $u_t \in \mathbb{R}^{32}$；双解码器用 $\mathcal{L}_{\text{ACT}}$、$\mathcal{L}_{\text{DYN}}$、$\mathcal{L}_{\text{KL}}$ 三项损失联合训练。下半部分（VLA 微调）：VLA 策略预测 $\hat{u}_t$，可训练解码器将其映射回可执行动作，实现无架构改动的监督替换。

### Figure 2: Lift Pot 真实任务关键帧

![Figure 2](https://arxiv.org/html/2608.03563v1/x2.png)

**说明**: UVT 策略执行双臂抬锅任务的时序关键帧。两臂需同时抓握锅的两个侧把手后抬起；放大图显示基线策略的典型失败模式（夹爪扣在锅沿或错过把手）。UVT 策略抓握成功率从基线的 38% 提升至 54%。

### Figure 3: Close Marker 真实任务关键帧

![Figure 3](https://arxiv.org/html/2608.03563v1/x3.png)

**说明**: UVT 策略执行笔盖对准合并任务的时序关键帧。左臂抓笔身，右臂抓笔盖，需毫米级精度对准插入；放大图标注对准阶段细节。"#Half" 表示双臂成功抓取但未完成插入的半成功次数。基线成功率为 0%，UVT 实现 5 次完全成功、28 次半成功（50 次中）。

### Figure 4: Plate Handover 真实任务关键帧

![Figure 4](https://arxiv.org/html/2608.03563v1/x4.png)

**说明**: UVT 策略执行纸盘双臂交接任务的时序关键帧。右臂需从桌面薄纸盘底部滑入，提起后传递给左臂放回桌面；放大图标注初始接近和交接阶段。"#Half" 表示成功拿起但交接失败的次数。UVT 将全成功次数从 7 次提升至 12 次（50 次中），半成功从 4 次提升至 28 次。

### Figure 5: UVT vs 原始动作的 [[t-SNE]] 可视化

![Figure 5](https://arxiv.org/html/2608.03563v1/x5.png)

**说明**: 在四个 LIBERO 子套件（Object/Spatial/Goal/Long）上对 $u_t$（左）和 $a_t$（右）做 [[t-SNE]] 降维可视化。UVT 嵌入未使用套件标签却自然按任务套件聚类，形成清晰的群落；原始动作嵌入不同套件的点大量混叠，无结构可见。表明 UVT 编码了更丰富的任务语义，与 VLM 特征空间更对齐。

### Figure 6: 解码动作与演示动作的时序对比

![Figure 6](https://arxiv.org/html/2608.03563v1/x6.png)

**说明**: 从预测 $\hat{u}_t$ 解码得到的动作轨迹（左）与对应演示动作块 $a_t$（右）的关节角时序曲线对比。解码动作保留了演示的整体运动趋势，但高频抖动明显减少——UVT 潜在空间过滤了演示中低层执行器噪声，提供了更平滑的监督信号。

### Table 1: 训练效率对比（成功率 %）

| 基准 | 任务套件 | 方法 | @10k 步 | @100k 步 |
|------|----------|------|---------|---------|
| LIBERO | Object | VLA-Adapter | 93.4 | 97.6 |
| | | **UVT (Ours)** | **94.0** | **99.6** |
| | Spatial | VLA-Adapter | 78.8 | 96.2 |
| | | **UVT (Ours)** | **95.0** | **98.6** |
| | Goal | VLA-Adapter | 90.4 | 95.6 |
| | | **UVT (Ours)** | **95.0** | **97.2** |
| | Long | VLA-Adapter | 56.2 | 89.0 |
| | | **UVT (Ours)** | **68.8** | **91.2** |
| | **Average** | VLA-Adapter | 79.7 | 94.6 |
| | | **UVT (Ours)** | **88.2** | **96.7** |
| LIBERO-Plus | Object | VLA-Adapter | 52.7 | 58.3 |
| | | **UVT (Ours)** | 52.4 | **60.4** |
| | Spatial | VLA-Adapter | 34.0 | 86.8 |
| | | **UVT (Ours)** | **81.5** | **87.2** |
| | Goal | VLA-Adapter | 42.3 | 72.4 |
| | | **UVT (Ours)** | **67.8** | **74.7** |
| | Long | VLA-Adapter | 41.7 | 64.3 |
| | | **UVT (Ours)** | **63.0** | **72.1** |
| | **Average** | VLA-Adapter | 42.7 | 70.5 |
| | | **UVT (Ours)** | **66.2** | **73.6** |

**关键发现**: 10k 步时 LIBERO-Plus Spatial 提升最为显著（34.0% → 81.5%，+47.5%），说明 UVT 在环境扰动下的加速效应尤为突出；100k 步时全面超过或持平基线，无速度与最终性能的 trade-off。

### Table 2: 最终性能对比（LIBERO，100k 步）

| 模型 | Object | Spatial | Goal | Long | **平均** |
|------|--------|---------|------|------|---------|
| VLA-Adapter | 97.6 | 96.2 | 95.6 | 89.0 | 94.6 |
| [[π₀.₅]] | 97.8 | 98.4 | 98.0 | 90.6 | 96.2 |
| UVT (VLA-Adapter) | **99.6** | 98.6 | 97.2 | 91.2 | **96.7** |
| UVT (π₀.₅) | 99.0 | **98.4** | **97.8** | **92.4** | **96.9** |

**关键发现**: UVT 在两种 VLA 架构上均有提升，且 UVT(π₀.₅) 以 96.9% 达到最优；UVT 的增益与策略头类型无关，验证了方法的通用性。

### Table 3: 消融实验（LIBERO-Spatial，LAM 正则化变体）

| 方法 | @10k 步 | @100k 步 |
|------|---------|---------|
| VLA-Adapter（基线） | 78.8 | 96.2 |
| + Villa-X 辅助头 | 83.2 | 96.8 |
| + UniVLA 辅助头 | 86.4 | 95.8 |
| **UVT（Ours）** | **95.0** | **98.6** |

**关键发现**: 简单地添加动态码辅助预测头（Villa-X +4.4%、UniVLA +7.6%）相比 UVT 的统一监督目标（+16.2%）逊色明显，证明将两种信号**融合为统一目标**而非独立预测头是关键。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套件，各 50 任务 | 标准仿真操作基准 | 训练效率 & 最终性能评测 |
| LIBERO-Plus | 4 套件，各 50 任务 | 对象配置 & 环境扰动增强版 | 鲁棒性评测 |
| 真实双臂数据集（自采） | 每任务 30 条演示（VR 遥操） | 双臂协调，位置/旋转随机化 | 真实世界微调 & 评测 |

### 实现细节

- **Backbone**: [[VLA-Adapter]]（回归式）和 [[π₀.₅]]（流匹配式）
- **UVT 维度**: $u_t \in \mathbb{R}^{32}$
- **平衡系数**: $\gamma = 0.5$（UVT 对齐 vs 动作重建）
- **早期评测步数**: 10k 步（约 8 GPU 小时，A6000）
- **最终评测步数**: 100k 步
- **真实场景训练**: 10 GPU 小时（A6000）
- **评测规模**: 每任务 50 次试验
- **分析工具**: [[t-SNE]] 用于 UVT 嵌入可视化

### 真实场景任务成功率

| 任务 | 基线 | UVT | 说明 |
|------|------|-----|------|
| Lift Pot | 38% | **54%** | 双臂同步抓握 |
| Close Marker | 0% | **≈66%**（含半成功） | 毫米级精度插入 |
| Plate Handover | **≈15%** | **≈56%**（含半成功） | 薄纸盘接触 + 双臂交接 |

---

## 批判性思考

### 优点

1. **无侵入性**: 零架构改动，只替换监督目标，可即插即用到任何 VLA fine-tuning 流程
2. **理论动机清晰**: 通过精度加权 [[Product of Experts]] 融合异构模态，[[t-SNE]] 可视化对设计合理性提供了直观验证
3. **样本效率大幅提升**: LIBERO-Plus 10k 步提升最高 +47.5%，在 limited data 场景下价值尤为突出
4. **真实世界验证充分**: 三类双臂任务覆盖精度操作、物体交接、软物体处理，基线从 0% 提升至部分成功

### 局限性

1. **依赖预训练 LAM**: 需要高质量的潜在动作模型；若 LAM 泛化性差或与目标任务分布不匹配，效果存疑
2. **两阶段离线计算**: 需先训练 MVAE、预计算全数据集的 $u_t$，换新任务/机器人时需重训
3. **精度任务仍存挑战**: Close Marker 完全成功率仍较低，sub-centimeter 插入对 UVT 平滑化后的动作也是挑战
4. **评测规模受限**: 每任务仅 30 条演示、50 次测试，真实场景结论统计功效有限
5. **领域局限**: 仅验证桌面双臂操作，无导航/多机器人等场景验证

### 潜在改进方向

1. 将 UVT 思路扩展到语言导航（高层语义 + 低层路点）和多智能体（联合行为预测目标）
2. 在线自适应更新 MVAE，减少离线预计算的分布偏移
3. 结合在线 RL 微调，将 UVT 作为 reward shaping 的辅助信号

### 可复现性评估

- [ ] 代码开源（论文中未提及代码链接）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（MVAE 结构、损失权重 $\gamma=0.5$、硬件、步数均有说明）
- [x] 数据集可获取（LIBERO 开源；真实数据集为自采集）

---

## 关联笔记

### 基于
- [[LAM]]: 提供冻结的视觉动态码提取能力，是 UVT 的核心输入来源
- [[LAPA]]: 最早将潜在动作模型用于机器人预训练的工作
- [[UniVLA]]: 任务中心的动态码方法，与 UVT 消融实验中作为辅助头对比
- [[MVAE]]: 多模态 VAE 框架，UVT 构建器的核心架构
- [[Product of Experts]]: 精度加权融合的理论基础

### 对比
- [[VLA-Adapter]]: 主要回归式基线，UVT 在其基础上替换监督目标
- [[π₀.₅]]: 主要流匹配式基线，验证 UVT 对不同策略头的通用性
- [[动作分块]]: 原始监督目标，UVT 的替换对象

### 方法相关
- [[视觉语言动作模型]]: UVT 针对的模型类别
- [[逆动力学模型]]: LAM 的核心实现机制
- [[潜在动作]]: UVT 动态码部分的来源概念
- [[模仿学习]]: UVT 的应用场景
- [[KL 散度]]: MVAE 正则项
- [[t-SNE]]: 嵌入空间分析工具

### 硬件/数据相关
- [[LIBERO]]: 主要仿真评测基准
- [[行为克隆]]: VLA 微调的训练范式

---

## 速查卡片

> [!summary] UVT
> - **核心**: 将机器人动作块与视觉动态码融合为统一潜在目标，替代 VLA 的原始动作监督
> - **方法**: 多模态 VAE + 精度加权 Product-of-Experts 融合 → $u_t \in \mathbb{R}^{32}$；可训练解码器推理时还原动作；零架构改动
> - **结果**: LIBERO 10k 步 +8.5%、LIBERO-Plus 10k 步 +23.5%（最高 +47.5%）；真实双臂任务全面提升（Lift Pot 38%→54%，Close Marker 0%→显著改善）
> - **代码**: N/A（项目主页: unified-visuomotor-targets.github.io）

---

*笔记创建时间: 2026-08-06*
