---
title: "MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation"
method_name: "MotionWAM"
authors: [Jia Zheng, Teli Ma, Yudong Fan, Zifan Wang, Shuo Yang, Junwei Liang]
year: 2026
venue: arXiv
tags: [world-action-model, humanoid-robot, loco-manipulation, whole-body-control, diffusion-transformer, motion-latent, real-time]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.09215
created: 2026-06-10
---

# 论文笔记：MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Mondo Robotics, HKUST (GZ), HKUST |
| 日期 | June 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[GR00T-N1.7]]、[[Cosmos-Policy]]、[[Diffusion Policy]]、[[Action Chunking]]、[[Pi0.5\|π0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.09215) / Code 未公开 |

---

## 一句话总结

> MotionWAM 将 [[Video DiT|视频扩散 Transformer]] 中间特征耦合 [[Motion DiT|运动 Transformer]]，以统一运动潜变量同时驱动人形机器人的移动与操作，实现 4.9 Hz 实时全身控制，比最优 VLA 基线高出 32% 以上。

---

## 核心贡献

1. **实时 WAM 驱动策略**: 利用 [[Video DiT]] 在固定去噪时间步 $\tau_f \approx 1$ 处的中间隐层特征，单次前向传递即可生成视觉预测信号，避免完整迭代去噪，实现 4.9 Hz 控制频率。
2. **统一运动潜变量**: 用 [[SONIC]] 离散 token（64 维）与连续末端执行器命令组成统一 $m_t$，一次性覆盖上身操作和下身移动，取代上身关节目标 + 下身基础速度的层级解耦范式，支持任务驱动足部交互（踢球、踩踏）。
3. **三阶段课程训练**: Stage 1 在 ≈2136 小时自我中心人类/人形视频上预训练 [[Video DiT]]；Stage 2 跨体型后训练 [[Motion DiT]]；Stage 3 在遥操作全身示范上微调，实现异构数据充分利用。

---

## 问题背景

### 要解决的问题
现有人形机器人操作系统将控制拆分为上身关节目标和下身基础速度命令，导致腿脚只能用于平衡保持，无法执行踢球、踩物等任务驱动的足部交互动作。

### 现有方法的局限
- [[GR00T-N1.7]]、[[Diffusion Policy]]、[[Action Chunking|ACT]]、[[pi05|π0.5]] 均采用上下身解耦架构，腿部功能受限
- 以 [[Cosmos-Policy]] 为代表的视频世界模型策略需完整迭代去噪，推理频率仅 0.7 Hz，无法实时控制
- 现有 [[WAM|World Action Model]] 未针对全身人形机器人的统一运动空间进行设计

### 本文的动机
视频 [[DiT|扩散 Transformer]] 在预训练中学到丰富的视觉动态知识，其中间去噪特征本身就是对未来帧的高层预测；只需在固定 $\tau_f$ 处提取一次特征，就能将其作为轻量 Motion DiT 的条件，在实时频率内完成动作预测。

---

## 方法详解

### 模型架构

MotionWAM 采用 **双 DiT（Video DiT + Motion DiT）** 架构：

- **输入**: 语言指令 $l$ + 当前帧观测 $o_t$（Intel RealSense D435i 头戴相机）+ 本体感受状态 $p_t$ + 具身标签 $e$
- **Video DiT Backbone**: [[Cosmos-Predict2|Cosmos-Predict2.5-2B]]（基于 [[DiT]]），在固定流时间步 $\tau_f \approx 1$ 处提取中间隐层特征 $h_t^{\tau_f}$
- **Motion DiT**: DiT-B 配置，隐层维度 2560，最大序列长度 1024，以 $h_t^{\tau_f}$ 为条件预测统一运动潜变量 $m_t$
- **输出**: 统一运动潜变量 $m_t = (m_t^{\text{cont}}, \tilde{k}_t)$，经 [[SONIC]] 解码为全身动作 $a_t$
- **总参数**: 约 2.5B（Video DiT 2B + Motion DiT ≈ 0.5B）

### 核心模块

#### 模块1: 统一运动潜变量（Unified Motion Latent）

**设计动机**: 用单一连续空间同时表达上身操作与下身移动，消除层级解耦带来的协调代价。

**具体实现**:
- **离散部分** $\tilde{k}_t$：通过 [[SONIC]] 的 [[FSQ|有限标量量化（FSQ）]] 瓶颈映射为 64 维离散运动 token，编码运动目标（步态意图）、躯干方向、高度调节、足部交互
- **连续部分** $m_t^{\text{cont}}$：直接输出夹爪 / 灵巧手末端执行器连续命令
- 推理时：$\tilde{k}_t \xrightarrow{\text{round}} \hat{k}_t \xrightarrow{\text{SONIC}} a_t$

#### 模块2: Video DiT 特征耦合（Feature Coupling）

**设计动机**: 用预训练视频生成模型的隐状态作为"世界预测"信号，替代完整迭代去噪，实现实时性。

**具体实现**:
- Video DiT 在固定 $\tau_f \approx 1$（接近纯噪声）处执行**单次**前向传递
- 提取中间隐层特征 $h_t^{\tau_f} = H[v_\theta^{\text{video}}](z_{t+1}^{\tau_f}, \tau_f \mid z_t^0, l)$
- $h_t^{\tau_f}$ 包含对 $o_{t+1}$ 的粗粒度预测，作为 Motion DiT 的条件输入
- Motion DiT 接收 $(h_t^{\tau_f}, p_t, e)$ 后通过 [[Flow Matching]] 完整去噪预测 $m_t$

#### 模块3: 三阶段训练框架

**Stage 1 — Video DiT 预训练**:
- 数据：约 2136 小时自我中心人类及人形视频（EgoDex、GR00T-X-Embodiment-Sim、RoboCOIN 等）
- 仅训练 Video DiT，学习通用视觉动态先验
- 学习率 $1\times10^{-5}$，batch 8/GPU × 128 GPU，训练 100K 步

**Stage 2 — 跨体型动作后训练**:
- 附加 Motion DiT，联合最小化 $\mathcal{L}_{\text{Stage2}} = \mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{video}}$
- 使用具身标签 $e$ 区分异构 Unitree G1 数据集
- 学习率：Video DiT $1\times10^{-5}$，Motion DiT $1\times10^{-4}$，batch 8/GPU × 32 GPU，训练 50K 步

**Stage 3 — 全身微调**:
- 数据：遥操作全身示范，每任务 200 条轨迹，通过 SMPL-24 → Unitree G1 重定向获得
- 联合微调全模型，学习率同 Stage 2，batch 8/GPU × 8 GPU，训练 15K 步

---

## 关键公式

### 公式1: [[WAM|WAM 问题表述]]

$$
o_{t+1} \sim p_v(\cdot \mid o_t, l), \quad m_t \sim p_a\!\left(\cdot \mid o_t,\, p_t,\, H\!\left(o_{t+1}^{\tau_v}\right)\right)
$$

**含义**: 将 WAM 分解为两个条件概率：Video DiT 生成下一帧预测，Motion DiT 以其为条件生成动作。当 $\tau_v \to 0$ 时，$o_{t+1}^{\tau_v} \to o_{t+1}$（完整去噪还原）；实践中固定 $\tau_v = \tau_f \approx 1$ 实现单次推理。

**符号说明**:
- $o_t$：当前时刻观测（RGB 图像）
- $l$：自然语言任务指令
- $p_t$：本体感受状态（关节角、速度）
- $H(\cdot)$：Video DiT 中间特征提取算子
- $m_t$：统一运动潜变量

### 公式2: [[Video DiT|Video DiT 特征提取]]

$$
h_t^{\tau_f} = H\!\left[v_\theta^{\text{video}}\right]\!\left(z_{t+1}^{\tau_f},\, \tau_f \,\middle|\, z_t^0,\, l\right)
$$

**含义**: 以当前帧潜编码 $z_t^0$ 和指令 $l$ 为条件，在固定流时间步 $\tau_f$ 处执行**一次**前向传递，提取中间隐层特征 $h_t^{\tau_f}$ 作为未来帧的粗粒度预测信号。

**符号说明**:
- $z_{t+1}^{\tau_f}$：在 $\tau_f$ 处加噪的下一帧潜变量
- $\tau_f \approx 1$：固定流时间步，接近纯噪声端
- $v_\theta^{\text{video}}$：Video DiT 速度预测网络
- $z_t^0$：当前帧干净潜编码

### 公式3: [[Flow Matching|Video DiT 训练目标]]

$$
\mathcal{L}_{\text{video}} = \mathbb{E}\!\left[\left\|v_\theta^{\text{video}}\!\left(z_{t+1}^{\tau_v},\tau_v \mid z_t^0, l\right) - \left(\varepsilon_v - z_{t+1}^0\right)\right\|_2^2\right]
$$

**含义**: 标准 [[Flow Matching]] 回归目标，训练 Video DiT 预测从噪声到真实下一帧的速度场。

**符号说明**:
- $\varepsilon_v \sim \mathcal{N}(0, I)$：视频噪声
- $z_{t+1}^0$：下一帧干净潜编码（Ground Truth）
- $\tau_v \sim \mathcal{U}(0,1)$：随机流时间步

### 公式4: [[Flow Matching|Motion DiT 训练目标]]

$$
\mathcal{L}_{\text{motion}} = \mathbb{E}\!\left[\left\|v_\phi^{\text{motion}}\!\left(m_t^{\tau_a},\tau_a \mid h_t^{\tau_f}, p_t, e\right) - \left(\varepsilon_m - m_t^0\right)\right\|_2^2\right]
$$

**含义**: Motion DiT 以 Video DiT 中间特征 $h_t^{\tau_f}$、本体状态 $p_t$、具身标签 $e$ 为条件，通过 Flow Matching 回归预测运动潜变量的速度场。

**符号说明**:
- $m_t^0$：干净运动潜变量（Ground Truth）
- $m_t^{\tau_a}$：在时间步 $\tau_a$ 加噪的运动潜变量
- $\varepsilon_m \sim \mathcal{N}(0, I)$：运动噪声
- $e$：具身标签，区分不同机器人数据集
- $v_\phi^{\text{motion}}$：Motion DiT 速度预测网络

### 公式5: [[Flow Matching|Stage 2 联合训练损失]]

$$
\mathcal{L}_{\text{Stage2}} = \mathcal{L}_{\text{motion}} + \mathcal{L}_{\text{video}}
$$

**含义**: Stage 2 同时优化 Video DiT 和 Motion DiT，保持视频预测能力的同时学习动作生成。

### 公式6: [[SONIC|运动 token 解码]]

$$
m_t = (m_t^{\text{cont}},\, \tilde{k}_t) \;\xrightarrow{\text{round}}\; \hat{k}_t \;\xrightarrow{\text{SONIC}}\; a_t
$$

**含义**: 连续预测值 $\tilde{k}_t$ 经取整量化为离散索引 $\hat{k}_t$，再由 [[SONIC]] 低层控制器解码为全身关节动作 $a_t$。

**符号说明**:
- $m_t^{\text{cont}}$：连续部分（末端执行器命令）
- $\tilde{k}_t$：连续形式的离散 token 预测值
- $\hat{k}_t$：量化后的离散 token 索引

---

## 关键图表

### Figure 1: MotionWAM 系统概览

![Figure 1 - MotionWAM Overview](https://arxiv.org/html/2606.09215v1/x2.png)

**说明**: Unitree G1 执行 MotionWAM 控制的实际轨迹，展示腰部控制、高度调节、蹲伏移动、身手协调和任务驱动足部交互五类全身行为。

### Figure 2: MotionWAM 整体框架

![Figure 2 - MotionWAM Architecture](https://arxiv.org/html/2606.09215v1/x3.png)

**说明**: 双 DiT 视频-运动模型三阶段训练框架。Stage 1 单独预训练 Video DiT；Stage 2 附加 Motion DiT，以 Video DiT 隐层特征为条件联合训练；Stage 3 在遥操作全身示范上微调整模型。

### Figure 3: 9 项真实世界任务套件

![Figure 3 - Task Suite](https://arxiv.org/html/2606.09215v1/x4.png)

**说明**: 在 Unitree G1 上设计的 9 项全身拟人操作任务（PnP Bottle、Kick Soccer、Retrieve Item、Load Cart、Toss Garbage、Lift Basket、Stock Shelves、Wipe Board、Do Laundry），每项任务均需主动腿部和躯干参与，不仅仅是保持平衡。

### Figure 4: 与 SOTA VLA 的性能对比

![Figure 4 - Comparison Results](https://arxiv.org/html/2606.09215v1/x5.png)

**说明**: 9 项任务各 20 次试验的成功率对比。MotionWAM 平均 76.1%，最优基线 GR00T-N1.7 仅 43.9%，绝对提升超 32%。

### Figure 5: 典型失败案例

![Figure 5 - Failure Cases](https://arxiv.org/html/2606.09215v1/x6.png)

**说明**: MotionWAM 的主要失败模式：操作物体离开相机视野、稳定性不足导致任务中断等。

### Figure 6: 推理演示

![Figure 6 - Inference Demonstrations](https://arxiv.org/html/2606.09215v1/x7.png)

**说明**: 9 项任务的代表性推理演示，展示 MotionWAM 在各类全身操作场景下的执行能力。

### Table 1: 解耦 vs 统一动作空间对比

| 方面 | 层级管道（现有方法） | MotionWAM |
|------|-------------------|-----------| 
| 上身控制 | 关节目标 | 统一运动 token |
| 下身控制 | 基础速度/高度命令 | 统一运动 token |
| 腿部功能 | 仅平衡保持 | **任务驱动足部交互** |
| 典型示例 | — | 踢球、踩踏、蹲伏取物 |

**说明**: 统一运动潜变量使机器人能够执行解耦架构物理上无法完成的足部任务。

### Table 2: 消融实验（5 项任务子集）

| 配置 | Lift Basket | Retrieve | Load Cart | Toss Garbage | Kick Soccer | 均值 |
|------|------------|---------|-----------|-------------|------------|------|
| w/o Stage 1（仅 Stage 2） | 65 | 45 | 30 | 30 | 40 | 42.0% |
| w/o Stage 2（仅 Stage 1） | 70 | 75 | 60 | 35 | 55 | 59.0% |
| **完整模型（Stage 1+2+3）** | **80** | **90** | **75** | **45** | **60** | **70.0%** |

**关键发现**: Stage 1 和 Stage 2 均不可或缺。去掉 Stage 1 的视频预训练时，均值下降 28%（70% → 42%），证明大规模视觉预训练提供了关键的世界知识；去掉 Stage 2 的跨体型后训练时，均值下降 11%（70% → 59%），说明异构数据适应对泛化至关重要。

### Table 3: 推理速度与参数量对比

| 模型 | 参数量 | 推理频率（Hz） |
|------|--------|--------------|
| GR00T-N1.7 | 1.6B | 6.5 |
| Qwen3DiT（消融） | 2.3B | 9.0 |
| Cosmos Policy | 2.0B | 0.7 |
| **MotionWAM** | **2.5B** | **4.9** |

**说明**: MotionWAM 以 2.5B 参数实现 4.9 Hz 实时控制，比完整迭代去噪的 Cosmos Policy 快 7×，同时大幅超越所有对比方法的性能。

---

## 实验

### 数据集

| 数据集 | 体型 | 采样权重 | 用途 |
|--------|------|---------|------|
| EgoDex | 人类（自我中心） | 0.300 | Stage 1 预训练 |
| GR00T-X-Embodiment-Sim | Fourier GR1（仿真） | 0.255 | Stage 1 预训练 |
| RoboCOIN | G1edu/Galbot/Leju | 0.080 | Stage 1 预训练 |
| 其他 | 多种 | 0.365 | Stage 1 预训练 |
| 遥操作示范 | Unitree G1 | — | Stage 3 微调（200 条/任务） |

### 实现细节

- **Video DiT**: Cosmos-Predict2.5-2B，[[Flow Matching]] 目标，固定 $\tau_f \approx 1$ 单次前向推理
- **Motion DiT**: DiT-B，隐层维度 2560，最大序列长度 1024
- **优化器**: AdamW（$\beta_1=0.9$, $\beta_2=0.95$）
- **Batch Size**: 8/GPU
- **Stage 1**: 128 GPU × 100K 步，视频学习率 $1\times10^{-5}$
- **Stage 2**: 32 GPU × 50K 步，视频学习率 $1\times10^{-5}$，运动学习率 $1\times10^{-4}$
- **Stage 3**: 8 GPU × 15K 步，同 Stage 2 学习率
- **硬件（推理）**: NVIDIA RTX 4090
- **机器人**: Unitree G1 + ALOHA2 夹爪，Intel RealSense D435i 头戴相机

### 可视化结果

MotionWAM 在踢球（Kick Soccer）任务中成功率约 60%，远超 GR00T-N1.7 的约 20%。推车（Load Cart）和擦板（Wipe Board）等需要躯干弯曲的任务中，统一全身运动的优势最为明显。失败主要发生在物体离开视野或需要极端姿态时。

---

## 批判性思考

### 优点
1. **单次推理设计优雅**: 固定 $\tau_f$ 提取特征的思路巧妙规避了迭代去噪的计算瓶颈，在保持世界模型知识的同时实现实时控制
2. **统一动作空间突破**: 将 [[SONIC]] token 与连续末端执行器命令融合，是首个系统展示任务驱动足部交互（踢球）的全身 WAM
3. **数据效率高**: 三阶段课程训练充分利用无标注视频（Stage 1）和异构机器人数据（Stage 2），每任务仅需 200 条示范

### 局限性
1. **单一平台验证**: 仅在 Unitree G1 上验证，未测试跨机器人迁移能力
2. **视野依赖**: 操作物体离开头戴相机视野时系统失效，缺乏主动视觉机制
3. **无新颖物体泛化研究**: 未设计受控实验评估泛化到训练外物体的能力

### 潜在改进方向
1. 引入主动视觉（颈部自由度或多相机）解决视野盲区问题
2. 探索更高 $\tau_f$ 处（更接近完整去噪）以提升动作质量，或自适应选择 $\tau_f$
3. 扩展到双臂平台（如 Fourier GR1）验证跨体型迁移

### 可复现性评估
- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（详细描述了三阶段配置）
- [x] 数据集可获取（EgoDex、GR00T-X-Embodiment-Sim 等公开数据集）

---

## 关联笔记

### 基于
- [[SONIC]]: 离散全身运动 token 和低层控制器，MotionWAM 的动作表示基础
- [[WAM]]: World Action Model 理论框架
- [[Cosmos-Predict2|Cosmos-Predict2.5-2B]]: Video DiT 骨干网络

### 对比
- [[GR00T-N1.7]]: NVIDIA 的人形机器人 VLA 基线（最优对比方法）
- [[Cosmos-Policy]]: 基于完整视频扩散模型的策略，速度 0.7 Hz
- [[Pi0.5|π0.5]]: 另一 VLA 基线
- [[Diffusion Policy]]: 扩散策略基线
- [[Action Chunking|ACT]]: 动作分块 Transformer 基线

### 方法相关
- [[WAM]]: 核心架构范式
- [[Flow Matching]]: Video DiT 和 Motion DiT 的训练目标
- [[DiT]]: 扩散 Transformer 骨干
- [[FSQ]]: 离散运动 token 的量化方法
- [[SONIC]]: 全身运动 token 编解码器

### 硬件/数据相关
- [[Unitree G1]]: 实验平台
- [[EgoDex]]: Stage 1 预训练数据集（人类自我中心操作视频）

---

## 速查卡片

> [!summary] MotionWAM
> - **核心**: 双 DiT 视频-运动耦合架构，Video DiT 单次前向提取世界预测特征，Motion DiT 以此为条件生成统一全身运动潜变量
> - **方法**: 固定流时间步 $\tau_f \approx 1$ 提取 Video DiT 中间特征 → Motion DiT Flow Matching 预测 → SONIC 解码全身动作
> - **结果**: 9 项真实任务平均成功率 76.1%（+32% vs GR00T-N1.7），4.9 Hz 实时运行
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-10*
