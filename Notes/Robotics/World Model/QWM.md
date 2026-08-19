---
title: "Morphology-Conditioned World Model for Cross-Embodiment Quadrupedal Locomotion"
method_name: "QWM"
authors: [Mohamad H. Danesh, Chenhao Li, Amin Abyaneh, Anas Houssaini, Kirsty Ellis, Glen Berseth, Marco Hutter, Hsiu-Chin Lin]
year: 2026
venue: arXiv
tags: [quadrupedal-locomotion, world-model, cross-embodiment, morphology-conditioning, zero-shot-transfer, model-based-rl]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2604.08780v2
created: 2026-08-19
---

# 论文笔记：Morphology-Conditioned World Model for Cross-Embodiment Quadrupedal Locomotion

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | McGill University / Mila - Quebec AI Institute / ETH Zurich / Université de Montréal |
| 日期 | April 2026 |
| 项目主页 | N/A |
| 对比基线 | [[PPO]] (PME-PPO model-free baseline) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.08780) / Code N/A |

---

## 一句话总结

> QWM 将单一生成动力学模型条件化于形态向量 μ，实现四足机器人家族内的零样本跨形态迁移，无需微调即可在真实硬件上部署。

---

## 核心贡献

1. **形态条件化世界模型 (QWM)**: 基于 [[DreamerV3]] 扩展，在 [[RSSM]] 的每个时间步注入形态向量 μ，使单一动力学模型覆盖整个四足机器人家族。
2. **[[Physical Morphology Encoder]] (PME)**: 从 URDF/物理描述中提取尺度不变特征（运动学、几何、动力学、驱动能力），编码为归一化向量 μ ∈ [-1,1]^d。
3. **[[Adaptive Reward Normalizer]] (ARN)**: 用指数移动平均的百分位区间自适应缩放异构奖励信号，防止多机器人联合训练中某类机器人主导梯度。
4. **Hetero-Isaac 环境**: 支持 8 种不同四足机器人同步并行训练的 [[Isaac Lab]] 异构向量化环境，每种机器人使用独立的奖励结构和物理参数。

---

## 问题背景

### 要解决的问题

现有[[世界模型]]通常与特定机器人形态绑定——一旦换用新型号，就必须重新收集数据、重新训练，无法实现零样本跨机器人迁移。

### 现有方法的局限

- **模型无关策略（如 [[PPO]]）**: 即使直接输入形态信息，在未见机器人上表现严重退化（PME-PPO 在 Go1 上仅获 23.1 奖励，远低于专家策略的 39.7）。
- **专家策略**: 每台机器人需单独训练，无法复用知识，工程代价高。
- **已有跨形态方法**: 主要针对策略层面，忽视动力学层面的形态共享结构。

### 本文的动机

**核心洞见**：四足家族的运动动力学随形态连续平滑变化，而最优控制策略则不一定如此。因此，将形态信息路由到动力学模型（世界模型）比直接条件化策略更有效——世界模型先学会"这个机器人的腿怎么动"，再用[[想象力训练]]推导对应的控制策略。

---

## 方法详解

### 模型架构

QWM 采用 **形态条件化 [[RSSM]]** 架构（基于 [[DreamerV3]]）：

- **输入**: 本体感知观测 $o_t \in \mathbb{R}^{48}$（关节位置/速度、底座速度、重力投影、速度指令、上一动作）+ 形态向量 $\mu \in [-1,1]^d$
- **Encoder**: 双塔结构——5 层本体感知塔（1024 单元） + 2 层形态塔（512 单元），融合后输入 RSSM
- **Latent State**: 确定性状态 $h_t \in \mathbb{R}^{512}$（GRU）+ 随机状态 $z_t$（32 类别 × 32 变量的 [[Categorical VAE]]）
- **形态注入位置**: 在 GRU recurrent cell 的每个时间步注入 μ（而非只在初始化时），防止隐状态记忆静态物理属性
- **输出**: 预测观测 $\hat{o}_t$、奖励 $\hat{r}_t$、续集标志 $\hat{c}_t$；策略在想象轨迹上用 [[PPO]] 优化
- **动作空间**: 12-DOF 目标关节位置

### 核心模块

#### 模块 1: Physical Morphology Encoder (PME)

**设计动机**: 不同四足机器人的原始物理参数（如绝对质量、腿长）不可直接跨机器人比较，需提取**尺度不变**特征。

**特征四组**：

- **运动学** $\mu_{kin}$: 遍历前左腿的各段长度 + 膝关节配置（X 型 vs 犬型）
- **几何** $\mu_{geo}$: 纵向/横向髋关节间距 + 宽高比（体型比例的尺度不变指标）
- **动力学** $\mu_{dyn}$: 对数缩放的总质量 + 躯干质量占比（决定转动惯量与力响应）
- **驱动** $\mu_{act}$: 权重归一化力矩（单位体重的最大力矩，跨机器人可比）

最终向量 $\mu_{raw} = [\mu_{kin}, \mu_{geo}, \mu_{dyn}, \mu_{act}]$，经 min-max 归一化为 $\mu \in [-1,1]^d$。

#### 模块 2: Adaptive Reward Normalizer (ARN)

**设计动机**: 不同机器人的奖励量纲差异极大（Spot 用专属 Gait Reward，权重 10.0；ANYmal 总奖励量级不同），直接联合训练会导致某类机器人主导梯度、其余机器人停滞。

**实现**: 用指数移动平均追踪每台机器人的奖励范围（P95−P05），动态归一化（见公式 7）。

#### 模块 3: Hetero-Isaac

**设计动机**: 现有仿真环境假设同构机器人，无法同时训练不同形态的机器人。

**实现**: 基于 [[Isaac Lab]] 的异构向量化环境，支持 8 种四足机器人的 4096 个并行环境同步运行，每种机器人使用独立的奖励函数和物理参数（见 Table 2-5）。

---

## 关键公式

### 公式 1: [[RSSM|DreamerV3 循环确定性状态]]

$$
h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})
$$

**含义**: 原始 DreamerV3 的 GRU 循环状态更新，h_t 为确定性隐状态。

**符号说明**:
- $h_t \in \mathbb{R}^{512}$: 确定性循环状态（GRU 输出）
- $z_{t-1}$: 上一时刻的随机潜变量
- $a_{t-1}$: 上一时刻的动作

---

### 公式 2: [[RSSM|表示模型]]

$$
z_t \sim q_\phi(z_t \mid h_t, o_t)
$$

**含义**: 给定确定性状态和当前观测，后验推断随机潜变量。

**符号说明**:
- $z_t$: 随机潜变量（32×32 离散 Categorical）
- $o_t$: 当前本体感知观测

---

### 公式 3: [[RSSM|转移预测器]]

$$
\hat{z}_t \sim p_\phi(\hat{z}_t \mid h_t)
$$

**含义**: 不依赖观测的先验预测，用于想象力展开（imagination rollout）。

---

### 公式 4: [[DreamerV3|解码器与奖励/续集预测]]

$$
\hat{o}_t,\; \hat{r}_t,\; \hat{c}_t \sim p_\phi(\cdot \mid h_t, z_t)
$$

**含义**: 从潜状态重建观测、预测奖励和 episode 是否继续。

---

### 公式 5: [[DreamerV3|世界模型损失]]

$$
\mathcal{L}_{WM} = \mathbb{E}_q\!\left[
  -\ln p_\phi(o_t|h_t,z_t)
  - \ln p_\phi(r_t|h_t,z_t)
  - \ln p_\phi(c_t|h_t,z_t)
  + \beta\, \mathrm{KL}\!\left[q_\phi(z_t|h_t,o_t) \,\|\, p_\phi(\hat{z}_t|h_t)\right]
\right]
$$

**含义**: 重建损失（观测+奖励+续集）+ KL 正则，标准 DreamerV3 损失。

**符号说明**:
- $\beta$: KL 权重（DreamerV3 默认 1.0，采用 free bits 截断）
- $\mathrm{KL}[\cdot\|\cdot]$: 后验对先验的 KL 散度

---

### 公式 6: [[Physical Morphology Encoder|形态条件化循环动力学]]

$$
h_t = f_\phi(h_{t-1},\; z_{t-1},\; a_{t-1},\; \mu)
$$

**含义**: QWM 的核心改动——在每个时间步将形态向量 μ 注入 GRU，使循环动力学以机器人物理特性为条件。

**符号说明**:
- $\mu \in [-1,1]^d$: PME 输出的归一化形态向量
- 相比公式 1，增加了 μ 作为额外条件输入

---

### 公式 7: [[Adaptive Reward Normalizer|自适应奖励归一化]]

$$
\sigma_R \leftarrow \alpha\, \sigma_R + (1-\alpha)(P_{95} - P_{05}),
\quad
r_{norm} = \frac{r}{\max(1.0,\; \sigma_R)}
$$

**含义**: 用指数移动平均追踪奖励的百分位区间（P95−P05），防止不同机器人的奖励量级差异主导训练。

**符号说明**:
- $\alpha$: EMA 衰减系数
- $P_{95}, P_{05}$: 奖励分布的第 95 和第 5 百分位
- $\sigma_R$: 当前估计的奖励范围
- $\max(1.0, \cdot)$: 防止除零，保留原始小奖励的尺度

---

## 关键图表

### Figure 1: QWM 系统概览

![Figure 1](https://arxiv.org/html/2604.08780v2/overall_quads.png)

**说明**: 展示异构四足机器人训练队列（ANYmal 系列 + Unitree 系列 + Spot），以及带有 PME 和 ARN 组件的训练流水线，以及向未见机器人的零样本部署流程。

---

### Figure 2: 训练性能与动力学预测精度

![Figure 2](https://arxiv.org/html/2604.08780v2/perf_and_horizon.png)

**说明**: 左图：异构训练队列上 QWM 与 PME-PPO 的学习曲线（QWM 收敛速度约快 2×）；右图：长时域动力学预测对比，QWM 在不同体型机器人之间保持同步，而基线方法出现发散。

---

### Figure 3: 真实世界量化结果

**说明**: 10 次试验中 ANYmal-D 和 Unitree Go1 的速度跟踪误差（$e_{xy}$ 和 $e_{yaw}$）。QWM 零样本策略误差约为专家策略的 10-13%，验证了真实硬件可部署性。

---

### Figure 4: 真实机器人部署

![Figure 4](https://arxiv.org/html/2604.08780v2/real_robot.png)

**说明**: 冻结的零样本 QWM 策略在真实 Unitree Go1 和 ANYmal-D 上部署，无需任何微调即可执行速度指令跟踪任务。

---

### Table 1: 零样本泛化性能（仿真）

| 方法 | 指标 | ANYmal-D | Unitree Go1 | Unitree B2 |
|------|------|----------|-------------|------------|
| PME-PPO（模型无关基线）| Reward | 10.1±1.9 | 23.1±2.3 | -0.2±3.2 |
| PME-PPO | Episode Length | 530±8.2 | 602±14.9 | 337±23.1 |
| **QWM（本文）** | **Reward** | **18.2±2.3** | **35.5±3.4** | **12.1±4.2** |
| **QWM** | **Episode Length** | **948.6±12.1** | **974.4±6.2** | **925.4±17.2** |
| Specialist PPO（上界） | Reward | 21.8±1.2 | 39.7±2.1 | 13.6±4.2 |
| Specialist PPO | Episode Length | 981.3±4.2 | 996.1±1.1 | 961.1±8.9 |

**关键发现**: QWM 恢复专家策略 80-90% 的性能；PME-PPO 模型无关基线在 B2 上几乎完全失败（奖励 -0.2）。

---

### Table 2: ANYmal & Unitree 标准奖励函数

| 跟踪项 | 公式 | 惩罚项 | 公式 |
|--------|------|--------|------|
| 线速度跟踪 (XY) | $e^{-\|v_{xy} - v^{cmd}_{xy}\|^2/\sigma_v^2}$ | 线速度 Z | $-v_z^2$ |
| 角速度跟踪 (Z) | $e^{-(\omega_z - \omega^{cmd}_z)^2/\sigma_\omega^2}$ | 角速度 XY | $-\|\omega_{xy}\|^2$ |
| 足部空中时间 | $\sum(t_{air}-0.5)\cdot I_{landed}$ | 平坦朝向 | $-\|g_{proj,xy}\|^2$ |
| — | — | 关节力矩 | $-\|\tau\|^2$ |
| — | — | 关节加速度 | $-\|\ddot{q}\|^2$ |
| — | — | 动作变化率 | $-\|a_t - a_{t-1}\|^2$ |
| — | — | 非期望接触* | $-I(\|F_c\|>1.0)$ |

*仅 ANYmal 系列启用；Unitree 系列禁用。

---

### Table 4: ANYmal & Unitree 奖励权重（摘录）

| 奖励项 | ANYmal-D/C/B | Unitree A1/Go1/Go2 | Unitree B2 |
|--------|-------------|-------------------|-----------|
| 线速度跟踪 (XY) | 1.0 | 1.5 | 1.0 |
| 角速度跟踪 (Z) | 0.5 | 0.75 | 0.75 |
| 足部空中时间 | 0.5 | 0.25 | 0.5 |
| 线速度 Z (L²惩罚) | -2.0 | -2.0 | -2.0 |
| 平坦朝向 (L²惩罚) | -5.0 | -2.5 | -5.0 |
| 关节力矩 (L²惩罚) | -2.5e-5 | -2.0e-4 | -2.0e-5 |

**关键发现**: 各平台奖励权重差异显著（力矩惩罚相差 10×），ARN 的重要性因此凸显。

---

### Table 6: 物理与观测随机化范围

| 参数 | 值 / 分布 |
|------|----------|
| 外部扰动速度 | $\mathcal{U}(-0.5, 0.5)$ m/s，每 10-15s 施加一次 |
| 控制延迟 | 20ms（dt=0.005s，Decimation=4） |
| 线速度噪声 | $\mathcal{U}(-0.1, 0.1)$ m/s |
| 角速度噪声 | $\mathcal{U}(-0.2, 0.2)$ rad/s |
| 关节位置噪声 | $\mathcal{U}(-0.01, 0.01)$ rad |
| 关节速度噪声 | $\mathcal{U}(-1.5, 1.5)$ rad/s |

---

### Table 7: 形态随机化

| 参数 | 分布范围 |
|------|---------|
| 质量（大型机器人） | Base ± 5.0 kg |
| 质量（小型机器人） | Base + [-1.0, 3.0] kg |
| 质心位置（底座） | ±5cm (XY)，±1cm (Z) |

---

## 实验

### 数据集（训练机器人队列）

| 机器人 | 厂商 | 膝关节配置 | 体型 |
|--------|------|-----------|------|
| ANYmal-B | ANYbotics | X 型 | 大 |
| ANYmal-C | ANYbotics | X 型 | 大 |
| ANYmal-D | ANYbotics | X 型 | 大 |
| Unitree A1 | Unitree | 犬型 | 小 |
| Unitree Go1 | Unitree | 犬型 | 小 |
| Unitree Go2 | Unitree | 犬型 | 小 |
| Unitree B2 | Unitree | 犬型 | 大（力矩更大） |
| Boston Dynamics Spot | Boston Dynamics | 犬型 | 大 |

### 实现细节

- **Backbone**: 修改版 [[DreamerV3]]（双塔编码器 + 形态条件 GRU）
- **Latent Space**: 32 类别 × 32 变量的 [[Categorical VAE]]
- **RSSM 确定性状态**: $h_t \in \mathbb{R}^{512}$
- **观测空间**: $\mathbb{R}^{48}$（关节位置/速度、底座速度、角速度、重力投影、速度指令、上一动作）
- **动作空间**: 12-DOF 目标关节位置
- **并行环境数**: 4,096
- **仿真器**: [[Isaac Lab]]（NVIDIA GPU 加速）
- **评估协议**: Leave-one-out（逐一排除 ANYmal-D、Go1、B2 作为测试机器人）

### 可视化结果

- 训练队列上 QWM 收敛速度约为 PME-PPO 的 2 倍，渐近性能相当。
- 动力学预测（长时域展开）：QWM 在不同体型机器人上保持同步，基线出现累积误差。
- 真实世界：Go1 和 ANYmal-D 上的速度跟踪误差约为专家策略的 10-13%。
- 鲁棒性（附录 H）：20% 形态参数污染下保持约 80% Episode Length；力矩损失 + 载重等非建模干扰下保持 75-84% 性能。

---

## 批判性思考

### 优点

1. **动力学层的形态共享**: 与直接条件化策略相比，通过世界模型传递形态信息更有理论依据（动力学平滑，策略不必平滑）。
2. **零样本 + 真实硬件验证**: 不只是仿真中的零样本，实际在 Go1 和 ANYmal-D 上完成了真实部署。
3. **ARN 的工程实用性**: 简洁优雅地解决了多机器人联合训练中的奖励尺度问题，无需手动调参。
4. **B2 的 OOD 结果**: Unitree B2 是联合训练中最大的形态外分布机器人，QWM 仍接近专家性能（12.1 vs 13.6），证明了真正的泛化。

### 局限性

1. **四足家族限制**: 仅支持三段腿 + 12-DOF 的四足机器人，不能推广到双足或其他拓扑结构。
2. **平地速度跟踪**: 不支持地形感知，无视觉外感知输入，实际应用场景有限。
3. **形态参数依赖**: 需要精确的 URDF/物理描述，实际机器人存在建模误差（附录 H 部分缓解）。
4. **Spot 因奖励差异过大被排除在泛化评估外**: Spot 使用完全不同的奖励函数，暴露出多任务奖励对齐问题尚未彻底解决。

### 潜在改进方向

1. 引入地形感知（高度图 / 深度图）的形态条件动力学。
2. 将形态向量扩展到腿数可变的情形（四足→六足→双足的连续形态空间）。
3. 利用 QWM 做在线适应（部署后根据真实数据微调形态向量估计）。

### 可复现性评估

- [ ] 代码开源（未提供 GitHub）
- [ ] 预训练模型（未提供）
- [x] 训练细节较完整（奖励函数、随机化范围、网络架构均在附录中详述）
- [ ] Hetero-Isaac 环境未开源

---

## 关联笔记

### 基于

- [[DreamerV3]]: QWM 的 Backbone，本文在其 RSSM 循环单元中注入形态向量
- [[RSSM]]: 循环状态空间模型，被修改以接受形态条件
- [[PPO]]: 用于在 World Model 想象空间中训练策略

### 对比

- PME-PPO（模型无关基线）: 直接将形态向量输入策略网络，无世界模型，在零样本测试上显著劣于 QWM

### 方法相关

- [[Physical Morphology Encoder]]: PME，核心的尺度不变特征提取组件
- [[Adaptive Reward Normalizer]]: ARN，多机器人联合训练的奖励归一化方案
- [[Cross-Embodiment]]: 跨形态迁移的宏观研究方向
- [[Categorical VAE]]: QWM 潜空间的离散化方案

### 硬件/仿真相关

- [[Isaac Lab]]: 底层 GPU 并行仿真器，Hetero-Isaac 基于此构建
- Unitree Go1 / ANYmal-D: 真实部署的硬件平台

---

## 速查卡片

> [!summary] QWM — Morphology-Conditioned World Model
> - **核心**: 将形态向量 μ 注入 DreamerV3 的 RSSM，实现四足家族的零样本跨形态迁移
> - **方法**: PME 提取尺度不变形态特征 + ARN 解决多机器人奖励异质性 + 形态条件 GRU
> - **结果**: 仿真中恢复专家策略 80-90%；真实 Go1/ANYmal-D 零样本部署误差约为专家的 10-13%
> - **代码**: 未开源

---

*笔记创建时间: 2026-08-19*
