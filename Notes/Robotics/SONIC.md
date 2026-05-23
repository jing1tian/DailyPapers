---
title: "SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control"
method_name: "SONIC"
authors: [Zhengyi Luo, Ye Yuan, Tingwu Wang, Chenran Li, Fernando Castañeda, Sirui Chen, Zi-Ang Cao, Jiefeng Li, David Minor, Qingwei Ben, Jinhyung Park, David Sami, Zi Wang, Xingye Da, Runyu Ding, Cyrus Hogg, Lina Song, Edy Lim, Eugene Jeong, Tairan He, Haoru Xue, Wenli Xiao, Simon Yuen, Jan Kautz, Yan Chang, Umar Iqbal, Linxi Fan, Yuke Zhu]
year: 2025
venue: arXiv
tags: [humanoid-control, motion-tracking, whole-body-control, reinforcement-learning, motion-capture, foundation-model, vla, teleoperation]
zotero_collection: Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2511.07820
created: 2026-05-23
---

# 论文笔记：SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA, Carnegie Mellon University, University of Texas at Austin |
| 日期 | November 2025 |
| 项目主页 | https://github.com/NVlabs/GR00T-WholeBodyControl |
| 对比基线 | [[BeyondMimic]], [[OpenHomie]] |
| 链接 | [arXiv](https://arxiv.org/abs/2511.07820) / [Code](https://github.com/NVlabs/GR00T-WholeBodyControl) |

---

## 一句话总结

> SONIC 通过大规模动作捕捉数据（700小时）和模型缩放（42M参数），将运动跟踪作为可扩展的基础任务，训练出能统一处理多模态输入（视频、文本、音乐、VR）的通用人形机器人全身控制策略。

---

## 核心贡献

1. **可扩展的运动跟踪基础**: 证明缩放数据（700h MoCap）、模型规模（1.2M→42M参数）和计算量（21k GPU小时）能显著提升人形机器人控制能力，成功率从98.0%提升至99.6%
2. **通用 Token 空间架构**: 设计三种编码器（机器人/人体/混合）映射到共享 [[有限标量量化|FSQ]] 量化 token，统一支持 MoCap、VR 遥操作和 [[视觉语言动作模型|VLA]] 等多模态控制接口
3. **实时运动学规划器**: 基于自回归掩码 token 预测的运动生成模块，笔记本上推理 <5ms，支持从速度命令到导航再到全身操作的跨任务复用

---

## 问题背景

### 要解决的问题

人形机器人需要在真实世界中执行多样化的全身运动，包括动态运动追踪、交互式导航、VR 遥操作和基于 VLA 的自主操作。现有方法通常针对特定任务设计，难以在通用控制框架中统一处理。

### 现有方法的局限

- **任务专用策略**（如 [[OpenHomie]]）：专注于步态控制，在动作多样性上表现弱，速度追踪生存率仅 43% vs SONIC 的 98.5%
- **判别器方法**（[[AMP]]、[[ASE]]）：基于 adversarial 方式，容易出现 mode collapse，训练不稳定
- **显式 SMPL 姿态接口**：在 VLA 任务中表现显著低于 FSQ token 接口（均值差距 +42 百分点）

### 本文的动机

运动追踪提供"每帧密集监督信号"，数据多样性越高学习信号越丰富，天然适合大规模扩展。通过构建通用 token 空间，不同来源的运动描述（MoCap 关节、SMPL 姿态、稀疏关键点）可无缝互换，使同一控制策略支持多种控制接口。

---

## 方法详解

### 模型架构

SONIC 采用**多编码器 + 共享量化 token + 双解码器**架构，在 Unitree G1 人形机器人上运行：

- **输入**: 未来运动目标（关节位置/速度 或 SMPL 姿态 或 稀疏关键点）+ 当前本体感知状态 $\bm{s}^p_t$
- **三种编码器**: [[运动捕捉|MoCap]] 编码器 $\mathcal{E}_r$、人体姿态编码器 $\mathcal{E}_h$、混合编码器 $\mathcal{E}_m$（均为 MLP）
- **量化层**: [[有限标量量化|FSQ]]（32维×32级，2个 token）
- **控制解码器** $\mathcal{D}_c$: 输出电机关节角度目标 $\bm{a}_t$
- **运动重建解码器** $\mathcal{D}_r$: 辅助监督，重建机器人运动目标
- **总参数**: 42M（最大配置）

![Figure 7 - SONIC Architecture](https://arxiv.org/html/2511.07820v3/x6.png)

### 核心模块

#### 模块1: Universal Token Space（通用 Token 空间）

**设计动机**: 利用 [[有限标量量化|FSQ]] 避免 VQ-VAE 的 codebook collapse，同时让来自不同模态的运动描述共享同一离散表示

**具体实现**:
- 三个编码器分别处理各自格式输入，输出经 FSQ 量化为离散 token $\bm{z}$
- FSQ 通过 straight-through estimation 传播梯度，无需 commitment loss
- 默认配置 FSQ-32-32（维度32，量化级32），对比 VQ-VAE MPJPE 从 35.3mm 降至 26.6mm
- 三个编码器的输出 token 在跨编码器 L2 距离上高度对齐（去掉一致性损失后距离增大 8×）

#### 模块2: 多编码器 + 一致性训练

**设计动机**: 使不同格式的运动输入（机器人关节、SMPL、稀疏关键点）在 token 空间对齐，支持运行时无缝切换

**具体实现**:
- **机器人编码器** $\mathcal{E}_r$: 接收未来 $F_r$ 帧关节位置/速度，时间步长 $\Delta t_r$
- **人体编码器** $\mathcal{E}_h$: 处理未来 $F_h$ 帧 SMPL 3D 关节位置，时间步长 $\Delta t_h$
- **混合编码器** $\mathcal{E}_m$: 组合稀疏上半身关键点（头部+手部）+ 下肢机器人运动目标
- 训练时三个编码器同时更新，受 token 对齐损失和循环一致性损失约束

#### 模块3: 实时运动学规划器（Kinematic Motion Planner）

**设计动机**: 将上层命令（速度/方向/风格/文本）转换为可追踪的运动序列，弥合运动追踪与导航/操作任务的差距

**具体实现**:
- 自回归架构：以起止姿态为条件，迭代细化中间 token
- [[掩码 Token 预测|Masked Token Prediction]] 每次迭代更新置信度低的 token
- 生成片段时长可变（0.8–2.4秒），由规划器自动决定
- 根轨迹生成使用[[临界阻尼弹簧模型]]，支持位置/朝向命令
- 推理速度：笔记本 <5ms，Jetson Orin ~12ms

#### 模块4: 强化学习训练（PPO）

**设计动机**: 通过 [[近端策略优化|PPO]] 在仿真中训练策略，配合域随机化实现 sim-to-real 迁移

**具体实现**:
- 奖励由追踪项 $\mathcal{R}$ 和惩罚项 $\mathcal{P}$ 组成
- 追踪项：根位置/方向、身体链路位置/方向、速度、角速度误差
- 惩罚项：anti-shake（头部和手腕角速度）、足部加速度
- 域随机化：摩擦系数、恢复系数、关节初始位置、质心位置
- 在 Isaac Gym 中训练，128 GPUs，21k GPU 小时

---

## 关键公式

### 公式1: [[强化学习|期望累积回报]]

$$
\mathbb{E}\left[\sum_{t=1}^{T}\gamma^{t-1}r_{t}\right]
$$

**含义**: [[近端策略优化|PPO]] 的优化目标，最大化折扣累积奖励

**符号说明**:
- $\gamma$: 折扣因子
- $r_t$: $t$ 时刻奖励
- $T$: 轨迹长度

---

### 公式2: [[奖励函数|运动追踪奖励]]

$$
r_{t}=\mathcal{R}(\bm{s}^{\text{p}}_{t},\bm{s}^{\text{g}}_{t})+\mathcal{P}(\bm{s}^{\text{p}}_{t},\bm{a}_{t})
$$

**含义**: 追踪奖励与惩罚项之和，$\mathcal{R}$ 鼓励接近目标姿态，$\mathcal{P}$ 惩罚不自然运动

**符号说明**:
- $\bm{s}^{\text{p}}_{t}$: 当前本体感知状态
- $\bm{s}^{\text{g}}_{t}$: 目标运动状态
- $\mathcal{R}$: 追踪奖励函数
- $\mathcal{P}$: 惩罚函数（anti-shake、足部加速度等）

---

### 公式3: [[多任务学习|总训练损失]]

$$
\mathcal{L}=\mathcal{L}_{\text{ppo}}+\mathcal{L}_{\text{recon}}+\mathcal{L}_{\text{token}}+\mathcal{L}_{\text{cycle}}
$$

**含义**: 四项联合损失，分别负责 RL 优化、运动重建、跨编码器 token 对齐、循环一致性

**符号说明**:
- $\mathcal{L}_{\text{ppo}}$: PPO 策略梯度损失
- $\mathcal{L}_{\text{recon}}$: 运动重建监督损失
- $\mathcal{L}_{\text{token}}$: 跨编码器 token 对齐损失
- $\mathcal{L}_{\text{cycle}}$: 循环一致性损失

---

### 公式4: [[运动重建损失|重建损失]]

$$
\mathcal{L}_{\text{recon}}=\left\|\bm{\mathcal{D}}_{r}(\bm{z}_{r})-\bm{g}_{r}\right\|^{2}+\left\|\bm{\mathcal{D}}_{r}(\bm{z}_{h})-\bm{g}_{r}\right\|^{2}+\left\|\bm{\mathcal{D}}_{r}(\bm{z}_{m})-\bm{g}_{r}\right\|^{2}
$$

**含义**: 三个编码器产生的 token，经运动解码器还原后均应与机器人目标运动 $\bm{g}_r$ 对齐

**符号说明**:
- $\bm{\mathcal{D}}_{r}$: 运动重建解码器
- $\bm{z}_{r}, \bm{z}_{h}, \bm{z}_{m}$: 机器人/人体/混合编码器输出 token
- $\bm{g}_{r}$: 机器人运动目标

---

### 公式5: [[Token 对齐|Token 对齐损失]]

$$
\mathcal{L}_{\text{token}}=\left\|\bm{z}_{r}-\bm{z}_{h}\right\|^{2}+\left\|\bm{z}_{r}-\bm{z}_{m}\right\|^{2}+\left\|\bm{z}_{m}-\bm{z}_{h}\right\|^{2}
$$

**含义**: 强制三个编码器在 token 空间对齐，使不同模态输入可无缝互换

**符号说明**:
- $\bm{z}_{r}, \bm{z}_{h}, \bm{z}_{m}$: 三编码器输出 token

---

### 公式6: [[循环一致性损失|循环一致性损失]]

$$
\mathcal{L}_{\text{cycle}}=\left\|\bm{\mathcal{E}}_{r}(\bm{\mathcal{D}}_{r}(\bm{z}_{h}))-\bm{z}_{r}\right\|^{2}
$$

**含义**: 人体 token 经解码再编码后应还原为机器人 token，保证人-机器人运动翻译的语义一致性

**符号说明**:
- $\bm{\mathcal{E}}_{r}$: 机器人编码器
- $\bm{\mathcal{D}}_{r}$: 运动重建解码器
- $\bm{z}_{h}$: 人体编码器 token
- $\bm{z}_{r}$: 机器人编码器 token

---

### 公式7: [[临界阻尼弹簧模型|根轨迹生成]]

$$
x(t)=\left(x_{T}-x_{0}+\left(v_{0}+\frac{c}{2}(x_{T}-x_{0})\right)t\right)e^{-\frac{c}{2}t}
$$

**含义**: 运动学规划器使用临界阻尼弹簧模型生成平滑的根轨迹，从当前位置平滑过渡到目标位置

**符号说明**:
- $x_0, x_T$: 起始和目标位置
- $v_0$: 初始速度
- $c$: 阻尼系数（位置方向：$5\ln 2$，偏航：$20\ln 2$）
- $t$: 时间

---

### 公式8: [[掩码 Token 预测|运动生成 Token 预测]]

$$
h=\mathcal{F}(\{p_{t},r_{t}\}_{t=1}^{4},\{p_{t},r_{t}\}_{t=T-4}^{T},\{z_{t}\}_{t=1}^{T/4})
$$

**含义**: 以起止4帧姿态和当前 token 序列为条件，预测下一步细化的 token 分布

**符号说明**:
- $\mathcal{F}$: 自回归规划网络
- $p_t, r_t$: 位置和旋转
- $\{z_t\}$: 当前 token 序列（时间下采样4倍）

---

### 公式9: 控制解码

$$
\bm{a}_{t}=\bm{\mathcal{D}}_{c}(\bm{z},\bm{s}^{\text{p}}_{t})
$$

**含义**: 控制解码器以量化 token 和当前本体感知状态为输入，输出关节角度目标

**符号说明**:
- $\bm{\mathcal{D}}_{c}$: 控制解码器
- $\bm{z}$: FSQ 量化 token
- $\bm{s}^{\text{p}}_{t}$: 本体感知状态（关节位置/速度）

---

## 关键图表

### Figure 1: SONIC 系统概览

![Figure 1 - SONIC Overview](https://arxiv.org/html/2511.07820v3/x1.png)

**说明**: SONIC 通过通用控制策略支持多样化人形机器人任务，展示多种控制接口（MoCap 追踪、VR 遥操作、文本/视频/音乐指令、VLA 自主操作）和全身动作输出。

---

### Figure 2: 缩放分析与基线对比

![Figure 2 - Scaling Analysis](https://arxiv.org/html/2511.07820v3/x2.png)

**说明**: 展示数据规模、模型规模和计算量对成功率（SR）和 MPJPE 的影响，以及 sim-to-real 迁移结果。模型从 1.2M 扩展到 42M 参数，成功率从 98.0% 提升至 99.6%。

---

### Figure 3: 交互式控制演示

![Figure 3 - Interactive Control](https://arxiv.org/html/2511.07820v3/x3.png)

**说明**: 交互式导航（风格变化、速度 0–6 m/s）、蹲/跪/爬（任意高度）、响应式拳击动作，展示运动学规划器的实时交互能力。

---

### Figure 4: 多模态控制与 VR 遥操作

![Figure 4 - Multi-modal Control](https://arxiv.org/html/2511.07820v3/x4.png)

**说明**: 视频遥操作、文本/音乐条件控制、VR 全身遥操作演示，通过通用 token 空间统一支持不同控制接口。

---

### Figure 5: VLA 驱动的全身操作

![Figure 5 - VLA Loco-manipulation](https://arxiv.org/html/2511.07820v3/x5.png)

**说明**: 5类全身运动操作任务（放苹果、抓胡萝卜/刷子、开垃圾桶、移动饮料罐、搬运钻头），[[视觉语言动作模型|VLA]] 模型直接输出 FSQ token 驱动全身协调运动，平均成功率 75%。

---

### Figure 6: BONES-SEED 数据集样本

![Figure 6 - Dataset Samples](https://arxiv.org/html/2511.07820v3/figs/panorama_grid.png)

**说明**: BONES-SEED 数据集的多样运动样本，覆盖 33 类运动（步态、舞蹈、手势、格斗、物体操作等），共 142,220 条标注序列（288 小时）。

---

### Figure 7: SONIC 架构详图

![Figure 7 - Architecture Detail](https://arxiv.org/html/2511.07820v3/x6.png)

**说明**: 通用控制策略架构细节，展示三种编码器（$\mathcal{E}_r$、$\mathcal{E}_h$、$\mathcal{E}_m$）、[[有限标量量化|FSQ]] 量化层、控制解码器和运动重建解码器的连接关系。

---

### Figure 8: 潜在空间对齐可视化

![Figure 8 - Latent Space Alignment](https://arxiv.org/html/2511.07820v3/x7.png)

**说明**: 有无一致性损失时的编码器间距离矩阵对比。去除一致性损失后，跨编码器 L2 距离增大 8×，证明对齐损失对多模态融合至关重要。

---

### Table 1: VLA 任务成功率

| 任务 | 控制接口 | 训练轨迹数 | 测试次数 | 成功率 |
|------|---------|-----------|---------|--------|
| 苹果放盘子 | 3点接口 | 300（单物体） | 20 | 90% |
| 抓取胡萝卜 | 全身 | 3,900（多物体） | 20 | 75% |
| 抓取刷子 | 全身 | 3,900（多物体） | 20 | 95% |
| 脚开垃圾桶 | 全身 | 200 | 10 | 70% |
| 饮料罐入桶 | 全身 | 1,000（多物体） | 10 | 60% |
| 搬运钻头和箱子 | 全身 | 300 | 10 | 70% |
| **平均（5任务）** | — | — | — | **75%** |

**关键发现**: FSQ token 接口相比显式 SMPL 姿态接口平均高出 +42 百分点，差距在复杂任务上更大。

---

### Table 2: 数据集划分统计

| 指标 | 训练集 | Test-Content（OOD） | Test-Rep. |
|------|--------|-------------------|-----------|
| 片段数 | 317,189 | 6,998 | 6,306 |
| 时长（小时） | 611 | 15 | 12 |
| 唯一子类别 | 8,447 | 182 | 1,088 |
| 与训练子类别重叠 | — | 0% | 100% |
| 与训练片段重叠 | — | 0% | 0% |

---

### Table 3: VLA 动作空间消融

| 任务 | FSQ Token | SMPL 姿态 | Δ |
|------|-----------|-----------|---|
| 抓取胡萝卜 | 75% | 60% | +15 |
| 脚开垃圾桶 | 70% | 20% | +50 |
| 饮料罐入桶 | 60% | 0% | +60 |
| **平均** | **68%** | **27%** | **+42** |

**关键发现**: FSQ token 在复杂任务上优势更显著，SMPL 姿态在高难度任务完全失败（0%）。

---

### Table 4(a): 量化器设计消融（128 GPU）

| 配置 | SR Test-Content | MPJPE-L | SR Test-Rep. | MPJPE-L Rep. |
|------|----------------|---------|--------------|--------------|
| **FSQ（ours）** | **99.3%** | **26.6mm** | **99.6%** | **25.5mm** |
| VQ-VAE | 98.7% | 35.3mm | 99.3% | 32.2mm |

---

### Table 4(b): FSQ 配置消融（32 GPU）

| 配置 | SR Content | MPJPE-L | SR Rep. | MPJPE-L Rep. |
|------|-----------|---------|---------|--------------|
| FSQ-16-16 | 96.9% | 35.7mm | 97.5% | 32.7mm |
| FSQ-16-32 | 98.3% | 29.7mm | 98.7% | 28.4mm |
| FSQ-32-16 | 98.3% | 30.3mm | 98.4% | 28.9mm |
| **FSQ-32-32（ours）** | **98.8%** | **27.5mm** | **99.3%** | **26.3mm** |

**关键发现**: Token 维度（32）比量化级数对性能影响更大。

---

### Table 4(c): 编码器对比（128 GPU）

| 编码器 | SR Content | MPJPE-L | SR Rep. | MPJPE-L Rep. |
|--------|-----------|---------|---------|--------------|
| 机器人编码器 $\mathcal{E}_r$ | 99.6% | **23.8mm** | 99.8% | **22.5mm** |
| 人体编码器 $\mathcal{E}_h$ | 99.6% | 24.4mm | 99.8% | 23.1mm |
| 混合编码器 $\mathcal{E}_m$ | 99.2% | 26.5mm | 99.7% | 25.2mm |

**关键发现**: 人体编码器与机器人编码器差距仅 +0.6mm，证明跨模态对齐有效。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| BONES-SEED（自建） | 142,220 序列（288h），训练 611h | 33类运动，522个演员，8,447个唯一子类别 | 训练+测试（含分布内和 OOD） |
| PHUMA（外部） | — | 外部 benchmark | OOD 测试 |

### 实现细节

- **控制器**: Unitree G1 人形机器人，50 Hz 策略频率
- **仿真**: Isaac Gym，[[域随机化]]（摩擦、恢复系数、关节位置、质心）
- **模型规模**: 1.2M / 7M / 42M 参数（最大配置发布）
- **训练硬件**: 128 × GPU，21,000 GPU 小时
- **部署**: TensorRT + CUDA Graph on Jetson Orin，1–2ms 前向推理
- **多频率架构**: 策略 50Hz、命令流 500Hz、操作输入 100Hz、运动规划 10Hz

### 真实世界验证

- 123 条多样运动序列测试，成功率 99.2%（仿真 100%）
- 整体 MPJPE-L：真实 25.7mm vs 仿真 22.3mm
- 基线对比：BeyondMimic 在 test-repetition 仅 85.8%，OpenHomie 在速度追踪生存率仅 43%

---

## 批判性思考

### 优点

1. **真正的大规模扩展验证**: 系统性地消融数据量、模型参数、计算量三个维度，scaling law 在 humanoid 控制上首次得到验证
2. **通用 token 空间设计精妙**: FSQ 量化避免 VQ-VAE 的 codebook collapse，一致性损失使不同格式输入在 token 空间高度对齐，使同一策略支持多种控制接口
3. **工程完整度高**: 从仿真训练到真实部署全链路完整，推理优化到 Jetson Orin 实时运行，配合 BONES-SEED 数据集开源，可复现性强

### 局限性

1. **安全性和能效缺乏形式化处理**: 论文承认对长时间部署缺乏安全性和能效的形式化保障，极端条件可能导致失衡
2. **全身操作成功率有限**: 平均 75%，部分任务（饮料罐入桶 60%）仍不稳定，工业应用尚需提升
3. **Unitree G1 专用**: 当前设计与特定硬件耦合，跨平台迁移成本未评估

### 潜在改进方向

1. 探索将运动学规划器扩展到更长时域规划，支持多步复合操作
2. 结合安全约束（CBF / safety layer）提升极端条件下的鲁棒性
3. 跨硬件平台迁移，验证通用 token 空间在不同机器人形态上的可迁移性

### 可复现性评估

- [x] 代码开源（github.com/NVlabs/GR00T-WholeBodyControl）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（论文附录有详细超参数）
- [x] 数据集可获取（BONES-SEED: huggingface.co/datasets/bones-studio/seed，CC BY-SA 4.0）

---

## 关联笔记

### 基于

- [[BeyondMimic]]: SONIC 对比的主要基线，运动追踪方法
- [[AMP]]: Adversarial Motion Prior，对比方法，易 mode collapse
- [[ASE]]: Adversarial Skill Embeddings，对比方法
- [[PPO]]: 训练策略所用的强化学习算法

### 对比

- [[OpenHomie]]: 步态专用控制器，速度追踪 43% vs SONIC 98.5%
- [[BeyondMimic]]: 运动追踪对比，test-rep 85.8% vs SONIC 99.6%

### 方法相关

- [[有限标量量化]]: 核心量化方法，避免 VQ-VAE codebook collapse
- [[近端策略优化]]: PPO，强化学习训练骨干
- [[掩码 Token 预测]]: 运动学规划器的生成机制
- [[临界阻尼弹簧模型]]: 根轨迹生成物理模型

### 硬件/数据相关

- [[Unitree G1]]: 实验平台人形机器人
- [[SMPL]]: 人体姿态参数化模型，人体编码器的输入格式
- [[运动捕捉]]: 数据来源技术，700h MoCap 数据

---

## 速查卡片

> [!summary] SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control
> - **核心**: 运动追踪作为可扩展基础任务，通过大规模数据和模型缩放训练通用人形机器人控制策略
> - **方法**: 多编码器 FSQ 通用 token 空间 + PPO + 实时运动学规划器
> - **结果**: 99.6% 运动追踪成功率（OOD），75% VLA 全身操作成功率，优于 BeyondMimic/OpenHomie
> - **代码**: https://github.com/NVlabs/GR00T-WholeBodyControl

---

*笔记创建时间: 2026-05-23*
