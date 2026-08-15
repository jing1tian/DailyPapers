---
title: "DreamX-Phi 1.0: Action-Conditioned Video World Model for Robotic Manipulation"
method_name: "DreamX-Phi"
authors: [Rui Chen, Xiangxiang Chu, Geng Li, Jifan Li, Qingfeng Shi, Datao Tang, Jing Tang, Jun Wang, Pengfei Zhang]
year: 2026
venue: arXiv
tags: [video-world-model, robotic-manipulation, action-conditioning, video-diffusion, bimanual-manipulation, se3-geometry, knowledge-distillation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.13489v1
created: 2026-08-15
---

# 论文笔记：DreamX-Phi 1.0

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | AMAP-ML (Alibaba) |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[Alpha-World]], [[FlowWAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.13489) / [Code](https://github.com/AMAP-ML/DreamX-Phi) |

---

## 一句话总结

> 给定单帧 RGB、语言指令和双臂末端执行器轨迹，DreamX-Phi 通过 [[PRoPE]] 几何注意力、[[SAM3]] 对象重加权和 [[V-JEPA]] 关系对齐生成物理可信的未来视频，在 [[WorldArena]] 2.0 Challenge 中获得 Track 1 第一名。

---

## 核心贡献

1. **[[PRoPE|PRoPE 几何动作表示]]**: 将 [[SE3|SE(3)]] 变换结构注入注意力头组，按臂分组编码双臂末端轨迹，保持刚体几何连续性
2. **操作感知监督三元组**: [[Depth Anything 3]] 深度辅助分支 + [[SAM3]] 掩码重加权损失 + [[V-JEPA]] Gram 矩阵对齐，确保场景几何一致和抓取状态演化
3. **[[Distribution Matching Distillation|DMD]] 少步蒸馏**: 对抗训练压缩多步采样为少步推理，在不显著损失质量的前提下大幅提升部署效率

---

## 问题背景

### 要解决的问题

以视频预测为核心的机器人世界模型需要同时满足：(1) 精确执行外部指定的双臂动作轨迹（动作保真度）；(2) 生成物理上合理的抓取与物体交互（物理可信度）；(3) 在有限算力下实时运行（推理效率）。现有方法在三者之间难以平衡。

### 现有方法的局限

- 动作条件通常以紧凑 token 形式注入，丢失 [[SE3|SE(3)]] 刚体几何结构，导致多臂身份混淆
- 生成监督信号集中在像素级重建，忽视接触区域物体一致性
- 扩散模型推理步数多，部署延迟高

### 本文的动机

机器人末端执行器运动天然具有 SE(3) 群结构。若将此结构直接注入注意力机制（PRoPE），模型可将动作与视觉后果在图像空间精确定位，同时通过对象感知监督和 JEPA 关系对齐保障物理一致性，最后用 DMD 蒸馏解决推理效率问题。

---

## 方法详解

### 模型架构

DreamX-Phi 采用**视频扩散 Transformer** 架构，以 [[Wan2.2]]（Wan2.2-TI2V-5B）为基础模型：

- **输入**: 观测帧 $x_0$ + 语言指令 $c$ + 双臂动作序列 $a_{1:T}$（末端执行器 SE(3) 轨迹 + 夹爪状态）
- **Backbone**: [[Wan2.2]] 视频扩散 Transformer（5B 参数）
- **核心模块**: [[PRoPE]] 几何动作注入 + 深度辅助分支 + [[SAM3]] 重加权 + [[V-JEPA]] 对齐
- **输出**: 预测视频帧序列 $x_{1:T}$
- **后训练**: [[Distribution Matching Distillation|DMD]] + 对抗训练少步蒸馏

### 核心模块

#### 模块 1: PRoPE — 几何动作条件注入

**设计动机**: 利用 [[SE3|SE(3) 变换群]] 结构将机器人末端轨迹注入注意力头，实现臂身份保持和空间精确定位。

**具体实现**:

1. **末端帧构建**: 为时刻 $t$、臂 $k$ 构建末端坐标系 $G_t^k$，然后相对臂 1 初始位姿归一化
2. **平移尺度归一化**: 以运动幅度 $\gamma$ 归一化，消除静止臂间距带来的偏置
3. **臂分组注入**: 每臂独占一组注意力头，将 SE(3) 变换矩阵 $\tilde{G}_t^k$ 直接作用于 Q/K/V
4. **夹爪注入**: 夹爪开合 $g_t^k$ 作为标量偏置叠加到输出 token 上

#### 模块 2: 深度辅助分支

**设计动机**: 以轻量辅助分支引导 RGB 主流学习更强的几何表征，而不增加推理时计算量。

**具体实现**:

- 复用 RGB Transformer 最后 $M$ 个 block 构成深度通道
- 深度到 RGB 的**单向** cross-attention（不影响 RGB 计算路径）
- 在潜在空间直接做 MSE 监督（非扩散过程），目标来自 [[Depth Anything 3]]

#### 模块 3: 对象中心一致性

两种互补机制确保接触-物体交互物理可信：

- **[[SAM3]] 掩码重加权**: 离线分割掩码对 flow-matching 损失重加权，放大接触局部误差信号
- **[[V-JEPA]] Gram 矩阵对齐**: 冻结教师网络约束对象特征的 [[Gram矩阵对齐|Gram 矩阵]]一致性，防止时序漂移

#### 模块 4: DMD 少步蒸馏

**设计动机**: 将多步扩散教师压缩为少步学生，加速推理同时保持生成质量。

- 分布匹配蒸馏（DMD）最小化学生与数据分布之间的 KL 散度
- 配合对抗训练（判别器 $D$）进一步稳定少步生成

---

## 关键公式

### 公式 1: [[Video Diffusion|条件视频生成分布]]

$$
p_\theta(x_{1:T} \mid x_0, a_{1:T}, c)
$$

**含义**: 给定初始帧 $x_0$、动作序列 $a_{1:T}$ 和语言条件 $c$，建模未来视频帧的条件分布。

**符号说明**:
- $x_0$: 观测初始帧
- $a_{1:T}$: 双臂末端轨迹 + 夹爪状态序列
- $c$: 语言指令
- $\theta$: 模型参数

---

### 公式 2: [[PRoPE|末端执行器帧构建]]

$$
G_t^k = \begin{bmatrix} R_t^k & p_t^k \\ 0^\top & 1 \end{bmatrix}, \quad \bar{G}_t^k = (G_1^1)^{-1} G_t^k
$$

**含义**: 将臂 $k$ 在时刻 $t$ 的末端位姿表示为 [[SE3|SE(3)]] 齐次矩阵，并相对臂 1 初始位姿做归一化。

**符号说明**:
- $R_t^k \in SO(3)$: 旋转矩阵
- $p_t^k \in \mathbb{R}^3$: 平移向量
- $G_1^1$: 臂 1 在时刻 1 的参考系，用于归一化

---

### 公式 3: [[PRoPE|运动幅度归一化]]

$$
\begin{aligned}
\gamma &= \max_{k,t} \|\bar{p}_t^k - \bar{p}_1^k\|_2 \\
s_\gamma &= \begin{cases} \gamma & \gamma > \varepsilon \\ 1 & \text{otherwise} \end{cases} \\
\tilde{G}_t^k &= \begin{bmatrix} \bar{R}_t^k & \bar{p}_t^k / s_\gamma \\ 0^\top & 1 \end{bmatrix}
\end{aligned}
$$

**含义**: 以运动幅度 $\gamma$ 归一化平移分量，消除臂间距静态偏置，使不同场景的轨迹在相同尺度。

**符号说明**:
- $\gamma$: 全局最大运动幅度
- $s_\gamma$: 归一化尺度因子（防止除以 0）
- $\varepsilon$: 小正数下界

---

### 公式 4: [[PRoPE|几何注意力变换]]

$$
Q'_i = D_i^\top Q_i, \quad K'_i = D_i^{-1} K_i, \quad V'_i = D_i^{-1} V_i
$$

$$
O_i^{\text{act}} = D_i \left[\operatorname{Attn}(Q', K', V')\right]_i
$$

**含义**: 将 SE(3) 变换矩阵 $D_i$（从 $\tilde{G}_t^k$ 导出）注入注意力头 $i$ 的 Q/K/V，使注意力计算感知空间几何关系，输出后再还原变换。

**符号说明**:
- $D_i$: 第 $i$ 个注意力头对应的 SE(3) 矩阵
- $Q_i, K_i, V_i$: 第 $i$ 头的查询、键、值
- $O_i^{\text{act}}$: 经几何感知注意力后的头输出

---

### 公式 5: [[PRoPE|夹爪状态注入]]

$$
b_t^k = W_g g_t^k + b_g
$$

$$
o_{t,u,h}^{\text{act}} \leftarrow o_{t,u,h}^{\text{act}} + b_t^k
$$

**含义**: 夹爪开合状态 $g_t^k$ 经线性变换得到标量偏置，叠加到对应臂 $k$ 在时刻 $t$ 的所有 token 输出上。

**符号说明**:
- $g_t^k \in [0,1]$: 臂 $k$ 在时刻 $t$ 的夹爪开合状态
- $W_g, b_g$: 可学习线性投影参数
- $o_{t,u,h}^{\text{act}}$: 空间位置 $u$、头 $h$ 处的激活

---

### 公式 6: [[Depth Estimation|深度辅助分支 Cross-Attention]]

$$
h_d^j = \operatorname{DepthBlock}_j\!\left(h_d^{j-1};\, K_{\text{rgb}}^j,\, V_{\text{rgb}}^j\right)
$$

**含义**: 深度分支的第 $j$ 个 block 通过单向 cross-attention 从 RGB 主流读取几何信息，但 RGB 主流不依赖深度分支。

**符号说明**:
- $h_d^j$: 深度分支第 $j$ 层特征
- $K_{\text{rgb}}^j, V_{\text{rgb}}^j$: RGB 主流第 $j$ 层的键和值

---

### 公式 7: [[Depth Estimation|深度辅助损失]]

$$
\mathcal{L}_{\text{depth}} = \frac{1}{|z^d|} \|\hat{z}^d - z^d\|_2^2
$$

**含义**: 在潜在空间以 MSE 监督深度预测，目标 $z^d$ 来自 [[Depth Anything 3]] 的深度图编码，不引入扩散噪声。

**符号说明**:
- $\hat{z}^d$: 深度分支预测的深度潜变量
- $z^d$: Depth Anything 3 提供的深度目标潜变量

---

### 公式 8: [[SAM3|对象感知流匹配损失]]

$$
\tilde{w}_i = 1 + (\lambda_m - 1) m_i, \quad w_i = \frac{\tilde{w}_i}{\frac{1}{|V|} \sum_{j \in V} \tilde{w}_j}
$$

$$
\mathcal{L}_{\text{rgb}}^{\text{obj}} = \frac{1}{|V|} \sum_{i \in V} w_i \ell_i^{\text{FM}}
$$

**含义**: 用 [[SAM3]] 离线分割掩码 $m_i$ 为像素重新赋权，接触区域（$m_i=1$）权重提升为 $\lambda_m$，放大抓取-物体交互位置的损失信号。

**符号说明**:
- $m_i \in \{0, 1\}$: SAM3 掩码（1 = 物体/接触区域）
- $\lambda_m$: 掩码增益超参数（$>1$）
- $\ell_i^{\text{FM}}$: 像素 $i$ 的 flow-matching 逐像素损失
- $|V|$: 有效像素总数

---

### 公式 9: [[Gram矩阵对齐|V-JEPA Gram 矩阵损失]]

$$
\ell_{\text{JEPA}}^{(b)} = \frac{1}{M_b^2} \|S_b S_b^\top - Q_b Q_b^\top\|_1
$$

**含义**: 约束预测帧特征 $Q_b$ 的 Gram 矩阵与冻结 [[V-JEPA]] 教师特征 $S_b$ 的 Gram 矩阵对齐，强制对象关系一致性（风格一致性扩展到时序）。

**符号说明**:
- $S_b$: [[V-JEPA]] 冻结教师网络对 batch $b$ 的特征矩阵（维度 $M_b \times d$）
- $Q_b$: 学生模型预测特征矩阵
- $S_b S_b^\top$: Gram 矩阵，编码特征间相关结构

---

### 公式 10: [[V-JEPA|V-JEPA 门控损失]]

$$
r_b = \mathbb{I}[M_b \geq M_{\min}] \cdot \mathbb{I}[\sigma_b \leq \sigma_{\max}]
$$

$$
\mathcal{L}_{\text{JEPA}} = \frac{\sum_b r_b \ell_{\text{JEPA}}^{(b)}}{\max\!\left(1,\, \sum_b r_b\right)}
$$

**含义**: 通过门控指示函数过滤掉 token 数过少（$M_b < M_{\min}$）或方差过高（$\sigma_b > \sigma_{\max}$）的不可靠 batch，对有效 batch 取平均。

**符号说明**:
- $M_b$: batch $b$ 的 token 数量
- $\sigma_b$: batch $b$ 的特征方差
- $M_{\min}, \sigma_{\max}$: 门控阈值超参数

---

### 公式 11: [[Distribution Matching Distillation|DMD KL 散度损失]]

$$
\mathcal{L}_{\text{DMD}}(\eta) = \mathbb{E}_{y \sim p_{\text{data}}(y),\, \tau} \left[ D_{\text{KL}}\!\left( q_{\eta,\tau}(\cdot \mid y) \,\|\, p_{\text{data},\tau}(\cdot \mid y) \right) \right]
$$

**含义**: 最小化少步学生 $q_{\eta,\tau}$ 在噪声级别 $\tau$ 处与真实数据分布 $p_{\text{data},\tau}$ 之间的 KL 散度，实现分布层面的匹配。

**符号说明**:
- $\eta$: 学生模型参数
- $\tau$: 扩散时间步（噪声级别）
- $q_{\eta,\tau}$: 少步学生在时间步 $\tau$ 的边缘分布
- $y$: 条件（初始帧 + 动作 + 语言）

---

### 公式 12-13: [[Distribution Matching Distillation|DMD 对抗训练目标]]

$$
\mathcal{L}_{\text{adv}}^G = -\mathbb{E}\!\left[\log D(z_u^f, u, y)\right]
$$

$$
\mathcal{L}_{\text{adv}}^D = -\mathbb{E}\!\left[\log D(z_u^r, u, y) + \log\!\left(1 - D(z_u^f, u, y)\right)\right]
$$

$$
\mathcal{L}_{\text{student}}^{\text{DMD}} = \mathcal{L}_{\text{DMD}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}^G
$$

**含义**: 判别器 $D$ 区分真实潜变量 $z_u^r$ 与学生生成的假潜变量 $z_u^f$；生成器（学生）同时最小化 DMD 散度和对抗损失，提升少步采样质量。

**符号说明**:
- $D(\cdot)$: 判别器网络
- $z_u^f$: 学生在时间步 $u$ 生成的假潜变量
- $z_u^r$: 真实数据潜变量
- $\lambda_{\text{adv}}$: 对抗损失权重

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1 Overview](https://arxiv.org/html/2608.13489v1/dreamx_phi_overview_crop.png)

**说明**: DreamX-Phi 1.0 概览。给定观测帧、语言指令和外部指定的双臂末端执行器轨迹，模型预测未来 RGB 视频帧序列，核心是 [[PRoPE]] 几何动作条件注入。

---

### Figure 2: Framework Architecture / 框架全图

![Figure 2 Framework](https://arxiv.org/html/2608.13489v1/dreamx_phi_train.png)

**说明**: 完整框架分三个部分：(1) 推理阶段——臂分组 [[PRoPE]] + 机器人光流线索的动作条件预测；(2) 训练阶段——[[SAM3]] 掩码重加权 RGB 目标 + [[Depth Anything 3]] 深度潜在监督 + 冻结 [[V-JEPA]] 对象关系监督；(3) DMD 后训练——多步教师蒸馏为少步学生 + 对抗训练稳定。

---

### Figure 3a: 标准场景定性结果

![Figure 3a Clean Scenes](https://arxiv.org/html/2608.13489v1/world_arena_clean_crop.png)

**说明**: WorldArena 2.0 Track 1 标准 [[RoboTwin 2.0]] 场景的预测结果。每行是一个预测片段，从左到右按时序采样帧，展示双臂运动连贯性和物体持久性。

---

### Figure 3b: 域随机化场景定性结果

![Figure 3b Randomized Scenes](https://arxiv.org/html/2608.13489v1/world_arena_random_crop.png)

**说明**: 域随机化场景（背景、纹理、光照、干扰物随机变化）的预测结果，验证模型对视觉扰动的鲁棒性。

---

### Table 1: 训练数据来源

| 数据源 | 领域 | 数据量 |
|--------|------|--------|
| [[Ego4D]] | 以第一视角为主的视频 | 3,700 h |
| AgiBot World 2026 | 真实机器人数据 | 1,900 h |
| InternData-A1 | 真实 + 仿真机器人 | 78 h 真实 + 3,747 h 仿真 |
| Cosmos3-DROID | 真实机器人数据 | 350 h |
| RoboCOIN | 真实机器人数据 | 618 h |
| [[RoboTwin 2.0]] | 仿真机器人（有动作标注） | 25,000 clips |

**说明**: 总计约 10,000 小时，[[RoboTwin 2.0]] 数据经 DreamX-Refiner 超分后处理；所有数据统一归一化为 LeRobot v2.1 表示。

---

### Table 2: WorldArena 2.0 Track 1 — 视觉质量指标

| 模型 | 视觉质量 | 运动质量 | 内容一致性 | 图像质量 | 美学质量 |
|------|---------|---------|-----------|---------|---------|
| **DreamX-Phi-1.0** | 63.25 | 43.38 | 92.93 | 22.90 | 5.81 |
| Alpha-World | 64.59 | 43.83 | 92.96 | 22.83 | 5.80 |
| FlowWAM-FiveAges | 64.31 | 44.00 | 92.54 | 22.76 | 5.80 |

**说明**: 在视觉/运动质量上三者相近，DreamX-Phi 在内容一致性上最强（92.93）。

---

### Table 3: WorldArena 2.0 Track 1 — 物理与控制指标

| 模型 | 物理合理性 | 3D 精度 | 可控性 | EWMScore-P |
|------|----------|--------|-------|-----------|
| **DreamX-Phi-1.0** | 57.36 | **57.15** | 98.55 | **60.65** |
| Alpha-World | 57.18 | 49.22 | 97.14 | 60.13 |
| FlowWAM-FiveAges | 53.74 | 49.76 | 98.99 | 59.72 |

**关键发现**: DreamX-Phi 以 3D 精度大幅领先（57.15 vs 49.22/49.76），最终 EWMScore-P 60.65 居第一（31 个参赛队）。

---

### Table 4: WorldArena 2.0 Track 2 — 策略训练结果

| 模型 | Adjust Bottle 成功率 (%) |
|------|------------------------|
| WOVR-PLUS | 68.75 |
| **DreamX-Phi-1.0** | **67.19** |
| Lute | 67.19 |

**说明**: Track 2 利用世界模型生成的视频训练下游策略，DreamX-Phi 以 67.19% 并列第二。

---

### Table 5: WorldArena 1.0 离线评测对比

| 模型 | 视觉质量 | 运动质量 | 内容一致性 | 物理合理性 | 3D 精度 | 可控性 | EWMScore-P |
|------|---------|---------|-----------|----------|--------|-------|-----------|
| **DreamX-Phi-1.0** | 55.72 | 40.87 | 92.73 | 77.90 | **58.98** | 93.17 | **76.88** |
| UNIS | 53.94 | 40.79 | 90.60 | **87.30** | 41.89 | 85.25 | 73.64 |
| SisyphusWorld | 45.57 | 38.38 | 96.58 | 71.98 | 44.58 | 94.85 | 73.06 |

**关键发现**: DreamX-Phi 在 WorldArena 1.0 超越头部官方提交（UNIS）3.24 分，物理合理性略低于 UNIS 但 3D 精度显著领先（58.98 vs 41.89）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[Ego4D]] | 3,700 h | 以人第一视角 | 动作无关预训练 |
| AgiBot World 2026 | 1,900 h | 真实机器人操作 | 动作条件微调 |
| InternData-A1 | ~3,825 h | 真实+仿真混合 | 动作条件微调 |
| [[RoboTwin 2.0]] | 25,000 clips | 仿真双臂任务，带动作标注 | 动作条件微调 + 评测 |
| WorldArena 2.0 | 标准评测集 | Track 1 视频预测，Track 2 策略训练 | 评测 |

### 实现细节

- **基础模型**: [[Wan2.2]] (Wan2.2-TI2V-5B，5B 参数)
- **动作表示**: LeRobot v2.1 格式，多相机流空间拼接
- **数据预处理**: DreamX-Refiner 对 RoboTwin 视频超分；过滤移动底座/灵巧手/静止片段
- **损失函数总览**: $\mathcal{L} = \mathcal{L}_{\text{rgb}}^{\text{obj}} + \lambda_d \mathcal{L}_{\text{depth}} + \lambda_j \mathcal{L}_{\text{JEPA}}$
- **推理**: DMD 后训练后少步采样（具体步数未披露）

### 可视化结果

在标准 [[RoboTwin 2.0]] 场景下，DreamX-Phi 预测的双臂动作连贯、物体持久性强；域随机化场景（不同背景/纹理/光照/干扰物）下依然保持稳定，显示出较好的泛化能力。

---

## 批判性思考

### 优点

1. **几何感知的动作条件**: [[PRoPE]] 将 SE(3) 结构直接注入注意力，理论上比紧凑 token 更能保留刚体轨迹信息，且臂分组设计优雅地解决双臂身份混淆问题
2. **多层次监督的协同**: 深度（场景几何）+ [[SAM3]]（接触区域）+ [[V-JEPA]]（对象关系）三者覆盖不同粒度的物理一致性约束，且辅助损失不增加推理开销
3. **端到端可扩展**: 以 [[Wan2.2]] 大规模预训练为起点，针对机器人操作的少量领域自适应便取得第一

### 局限性

1. **仅开环预测**: 模型接收外部提供的动作轨迹，不生成动作策略，Track 2 的策略训练效果也仅在单个任务（Adjust Bottle）验证
2. **真实世界泛化未验证**: 所有评测局限于 WorldArena/RoboTwin 仿真环境，真实机器人迁移性未知
3. **消融实验缺失**: 未提供 PRoPE / 深度分支 / V-JEPA 各组件的独立贡献消融
4. **训练细节披露不足**: 学习率、batch size、DMD 步数等超参数均未公开

### 潜在改进方向

1. 扩展为联合生成视频和动作的世界-动作模型（WAM），支持闭环控制
2. 在更多操作任务上验证 Track 2 策略训练效果
3. 补充组件消融实验，量化 PRoPE / 深度 / JEPA 各自贡献

### 可复现性评估

- [x] 代码开源（WorldArena 2.0 Challenge 结束后，目前已在 GitHub 公开仓库）
- [ ] 预训练模型（待比赛结束后发布）
- [ ] 训练细节完整（超参数未披露）
- [x] 数据集可获取（Ego4D、RoboTwin 2.0 等均公开）

---

## 关联笔记

### 基于

- [[Wan2.2]]: 基础视频扩散 Transformer 骨干（5B 参数）
- [[RoPE]]: PRoPE 由 RoPE 泛化而来，将旋转位置编码推广至 SE(3) 变换
- [[Distribution Matching Distillation|DMD]]: 少步蒸馏后训练策略

### 对比

- [[FlowWAM]]: WorldArena 2.0 Track 1 第三名，对比基线
- [[Alpha-World]]: WorldArena 2.0 Track 1 第二名，视觉质量略高但 3D 精度低

### 方法相关

- [[PRoPE]]: 核心动作条件注入机制（SE(3) 几何注意力）
- [[V-JEPA]]: 对象关系对齐教师网络
- [[SAM3]]: 离线对象分割，用于损失重加权
- [[Depth Anything 3]]: 深度辅助监督来源
- [[Gram矩阵对齐]]: V-JEPA 损失中的关系对齐机制
- [[SE3|SE(3) 变换群]]: PRoPE 的数学基础

### 硬件/数据相关

- [[RoboTwin 2.0]]: 主要仿真评测平台和训练数据来源
- [[Ego4D]]: 大规模以人为主的视频预训练数据
- [[WorldArena]]: 评测 benchmark（Track 1 视频预测，Track 2 策略训练）
- [[Bimanual Manipulation]]: 论文专注的操作任务类型

---

## 速查卡片

> [!summary] DreamX-Phi 1.0
> - **核心**: Action-conditioned video world model，以 SE(3)-aware PRoPE 实现精确双臂动作条件注入
> - **方法**: PRoPE 几何注意力 + 深度辅助分支 + SAM3 重加权 + V-JEPA Gram 对齐 + DMD 蒸馏
> - **结果**: WorldArena 2.0 Track 1 **第一**（EWMScore-P 60.65），Track 2 **第二**（67.19%）
> - **代码**: [github.com/AMAP-ML/DreamX-Phi](https://github.com/AMAP-ML/DreamX-Phi)

---

*笔记创建时间: 2026-08-15*
