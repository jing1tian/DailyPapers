---
title: "NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation"
method_name: "NavWAM"
authors: [Daichi Azuma, Taiki Miyanishi, Koya Sakamoto, Shuhei Kurita, Yaonan Zhu, Petr Khrapchenkov, Motoaki Kawanabe, Yusuke Iwasawa, Yutaka Matsuo]
year: 2026
venue: arXiv
tags: [world-action-model, visual-navigation, diffusion-policy, goal-conditioned, action-chunking, robot-navigation]
zotero_collection: Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2606.13494
created: 2026-06-13
---

# 论文笔记：NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | RIKEN AIP, University of Tokyo |
| 日期 | June 2026 |
| 项目主页 | [dachii-azm.github.io/navwam](https://dachii-azm.github.io/navwam/) |
| 对比基线 | [[NWM]]、[[OmniVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13494) / Code: 暂未发布 |

---

## 一句话总结

> NavWAM 将导航世界模型转化为闭环策略，通过统一的 [[Diffusion Transformer]] 在 9 帧潜在 Canvas 中联合预测未来观测、目标进度值和可执行动作块，消除了 CEM 外部规划器，推理延迟降低 1136 倍。

---

## 核心贡献

1. **世界-动作联合建模**: 在共享潜在序列（[[Latent Canvas|Canvas]]）中统一预测未来观测、目标进度值和动作块，消除传统世界模型方法所需的外部 [[Cross-Entropy Method|CEM]] 规划器
2. **Cosmos Predict2 适配**: 将 2B 参数的视频世界模型改造为导航策略，通过三阶段训练课程（HM3D 预训练 → 真实数据微调 → 领域微调）实现 Sim-to-Real 迁移
3. **极致推理效率**: 相比 NWM+CEM(N=120)，FLOPs 降低 3266 倍（14521 TF → 4.45 TF），延迟降低 1136 倍（233831ms → 205.7ms），峰值 GPU 显存降低 10.7 倍，且性能更优

---

## 问题背景

### 要解决的问题

目标条件视觉导航（Goal-Conditioned Visual Navigation）要求机器人在未知室内环境中仅凭当前 [[egocentric observation|自中心观测]] 和目标图像导航至指定位置。关键挑战在于：[[partial observability|部分可观测性]] 带来的长程规划困难，以及如何将"预测未来"的能力转化为可执行的连续控制动作。

### 现有方法的局限

- **直接策略**（GNM/ViNT/NoMaD/OmniVLA）：将观测和目标直接映射为动作，缺乏视觉预见性（visual foresight），在复杂场景泛化能力有限
- **导航世界模型**（[[NWM]]）：只预测未来观测，需在推理时用 [[Cross-Entropy Method|CEM]] 搜索动作，计算代价极高（~234 秒/动作），且预测与执行解耦导致误差积累
- **分离的预测与规划**：两模块之间的误差传播，难以端到端优化

### 本文的动机

导航世界模型本质上是视频预测问题，而动作预测（[[Diffusion Policy|扩散策略]]）也可被框架化为同一问题——只是把动作放到视频的"未来帧"位置上。因此，将动作、未来状态和目标进度值全部放入共享潜在 Canvas，让同一个扩散模型统一去噪，就能在单次前向传播中直接生成可执行策略，彻底消除规划循环。

---

## 方法详解

### 模型架构

NavWAM 采用 [[Diffusion Transformer]] 架构，基于预训练视频世界模型 [[Cosmos Predict2]]（2B 参数，Cosmos-Predict2 Video2World，480p，16fps）改造：

- **输入**: 目标图像 $g$ + 当前观测 $o_t$ + 机器人状态 $s_t$（位姿归一化 3 维向量）
- **Backbone**: [[Cosmos Predict2]]（2B，[[Causal VAE|因果 VAE]] + [[DiT]] 去噪器，分辨率 224×224）
- **核心模块**: [[Latent Canvas|潜在 Canvas]] — 将所有信息编码为 9 帧潜在序列，用统一 [[Diffusion Policy|扩散去噪]] 联合建模
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H-1}$（H=4 推理，H=16 预训练）+ 未来观测 $o_{t+H-1:t+H}$ + 目标进度值 $v_{t+H}$
- **总参数**: 2B（Cosmos Predict2 主干）
- **推理频率**: ~5Hz（RTX PRO 6000，bfloat16）

### 核心模块

#### 模块1: 潜在 Canvas（Latent Canvas）

**设计动机**: 利用 [[Causal VAE|因果 VAE]] 将图像和非图像数据统一编码为同一潜在空间，使 [[Diffusion Transformer]] 能无差别地去噪所有模态，从而将策略、世界模型、价值函数三种角色融入单一模型

**Canvas 9 帧布局**:

| 帧号 | 内容 | 维度 | 类型 |
|------|------|------|------|
| 0 | 空白（VAE 时间填充） | — | 已观测 |
| 1 | 当前状态 $s_t = [x_t/100, y_t/100, \psi_t/\pi]$ | $\mathbb{R}^3$ | 已观测 |
| 2 | 目标图像 $g$（或 $o_{t-1}$） | $3 \times 224 \times 224$ | 已观测 |
| 3 | 当前观测 $o_t$ | $3 \times 224 \times 224$ | 已观测 |
| 4 | 动作块 $a_{t:t+H-1}$ | $3H$ | **预测** |
| 5 | 未来状态 $s_{t+H}$ | $\mathbb{R}^3$ | **预测** |
| 6 | 未来观测 $o_{t+H-1}$ | $3 \times 224 \times 224$ | **预测** |
| 7 | 未来观测 $o_{t+H}$ | $3 \times 224 \times 224$ | **预测** |
| 8 | 目标进度值 $v_{t+H} \in [0,1]$ | 标量 | **预测** |

**具体实现**:
- 使用 [[Causal VAE|因果 VAE]]（时间压缩比 4:1，处理 33 原始帧→11 填充+8 潜在帧）将帧 0-8 编码为潜在表示
- 非图像数据（状态、动作、价值）渲染为与图像同尺寸的常数帧，再经 VAE 编码
- 动作块帧权重 $\lambda = 5$（相对于图像帧），强化动作预测信号
- 条件帧设噪声 $\sigma = 0$（无噪声），预测帧加噪 $\sigma \sim p(\sigma)$ 实现软屏蔽

#### 模块2: 三模式条件采样（Multi-Mode Conditioning）

**设计动机**: 通过随机屏蔽不同帧，使单一模型同时具备策略、世界模型、价值函数三种能力，互相强化

**三种模式（训练采样比例 50/25/25）**:

- **策略模式（50%）**: 已观测帧 0-3，预测帧 4-8（动作 + 未来观测 + 价值）— 推理时使用
- **世界模型模式（25%）**: 已观测帧 0-4（含动作），预测帧 5-8（未来观测 + 价值）
- **价值模式（25%）**: 已观测帧 0-7，仅预测帧 8（标量价值）

#### 模块3: 三阶段训练课程

**设计动机**: [[Curriculum Learning|课程学习]] 使模型先在仿真中学习视觉世界规律，再逐步适配真实数据分布

- **Phase 1 — HM3D 仿真预训练**: 802 场景 / 185K 轨迹，H=16，学习室内场景的物理规律和导航策略
- **Phase 2 — 真实数据联合微调**: RECON(11835) + SACSoN(2000) + SCAND(372) 共 ~14K 轨迹，H=4，迁移到真实世界分布
- **Phase 3 — 域内微调（可选）**: GO Stanford 3544 轨迹，用于特定场景的最终适配

---

## 关键公式

### 公式1: [[Direct Navigation Policy|直接导航策略]]（对比基线）

$$
\pi_{\theta}(a_{t:t+H-1} \mid o_t, g)
$$

**含义**: 传统直接策略——给定当前观测 $o_t$ 和目标图像 $g$，直接输出 $H$ 步动作块，不包含任何视觉预见性

**符号说明**:
- $a_{t:t+H-1}$: 时间步 $t$ 到 $t+H-1$ 的动作块
- $o_t$: 当前自中心观测图像
- $g$: 目标图像
- $H$: 动作块长度（推理时 $H=4$）

### 公式2: [[Navigation World Model|导航世界模型]]（对比基线）

$$
p_{\theta}(o_{t+H} \mid o_t, a_{t:t+H-1})
$$

**含义**: 传统导航世界模型——给定当前观测和动作序列，预测 $H$ 步后的未来观测，需外部 CEM 规划器评分选择动作

**符号说明**:
- $o_{t+H}$: $H$ 步后的预测未来观测
- $a_{t:t+H-1}$: 输入动作序列（推理时通过 CEM 搜索）

### 公式3: [[NavWAM]] 联合世界-动作预测（核心目标）

$$
p_{\theta}\!\left(a_{t:t+H-1}, s_{t+H}, o_{t+H-1:t+H}, v_{t+H} \mid o_t, g\right)
$$

**含义**: NavWAM 的核心建模目标——在统一模型中联合预测动作块、未来状态、未来观测序列和目标进度值，无需外部规划

**符号说明**:
- $a_{t:t+H-1}$: $H$ 步动作块
- $s_{t+H}$: $H$ 步后的未来机器人状态
- $o_{t+H-1:t+H}$: 未来两帧观测（连续时间步）
- $v_{t+H}$: 目标进度值 $\in [0, 1]$

### 公式4: [[Diffusion Policy|扩散训练损失]]

$$
\mathcal{L}_{\mathrm{diff}} = \mathbb{E}_{\sigma, \epsilon}\!\left[w(\sigma)\, \|x_0 - F_\theta(x_\sigma, \sigma, c)\|_2^2\right]
$$

**含义**: 标准扩散去噪目标，模型 $F_\theta$ 学习从加噪输入 $x_\sigma$ 直接预测干净数据 $x_0$，加权系数 $w(\sigma)$ 对不同噪声强度平衡损失

**符号说明**:
- $x_0$: Canvas 中待预测帧的干净潜在表示
- $x_\sigma = x_0 + \sigma \epsilon$: 加噪版本，$\epsilon \sim \mathcal{N}(0, I)$
- $\sigma$: 噪声强度，训练时 $\sigma_{\max}=200$，$\sigma_{\min}=10^{-2}$（推理时 $\sigma_{\max}=80$，$\sigma_{\min}=4$）
- $c$: 条件信息（已观测帧 0-3）
- $w(\sigma)$: EDM 噪声权重函数
- $F_\theta$: [[Cosmos Predict2]] 主干去噪器

### 公式5: [[Goal-Progress Value|目标进度值]]

$$
v_{t+H} = \mathrm{clip}\!\left(1 - \frac{\|p_{\mathrm{end}} - p_t\|_2}{d_{\max}},\ 0,\ 1\right)
$$

**含义**: 衡量机器人距目标的接近程度，归一化为 $[0,1]$（0=最远，1=到达）；作为辅助监督信号引导策略关注目标接近，对长时域预测（h=8）带来 27% ATE 改善

**符号说明**:
- $p_{\mathrm{end}}$: 目标位置（导航终点）
- $p_t$: 当前机器人位置（$H$ 步后的位置）
- $d_{\max}$: 训练数据轨迹长度的上百分位数，用于归一化
- 真实数据用欧氏距离，HM3D 仿真用测地线距离

### 公式6: [[Robot State Encoding|机器人状态编码]]

$$
s_t = \left[\frac{x_t}{100},\ \frac{y_t}{100},\ \frac{\psi_t}{\pi}\right] \in \mathbb{R}^3
$$

**含义**: 将机器人的 $(x, y, \psi)$ 位姿归一化编码为 3 维向量，与 VAE 图像帧在同一潜在空间对齐；Canvas 帧 1 中用此向量填充常数帧

**符号说明**:
- $x_t, y_t$: 机器人在地图坐标系下的 2D 位置（单位：cm，除以 100）
- $\psi_t$: 偏航角（单位：rad，除以 $\pi$ 归一化到 $[-1, 1]$）

### 公式7: [[Action Chunk|动作块]] 定义

$$
a_i = (\Delta x_i,\ \Delta y_i,\ \Delta\psi_i) \in \mathbb{R}^3, \quad i = t, \ldots, t+H-1
$$

**含义**: 每步动作为相对位移增量 $(\Delta x, \Delta y, \Delta\psi)$，H 步组成动作块；Canvas 帧 4 中以 $3H$ 通道的常数帧编码，再经 VAE 压缩

**符号说明**:
- $\Delta x_i, \Delta y_i$: 第 $i$ 步的 2D 位置增量（各数据集间距 0.12–0.38m）
- $\Delta\psi_i$: 第 $i$ 步的偏航角增量
- $H$: 动作块长度（推理 $H=4$，预训练 $H=16$）

---

## 关键图表

### Figure 1: 导航方法对比概览

![[NavWAM_fig1_teaser.png]]

**说明**: 三种导航范式对比。(a) 直接策略：$o_t, g \to a_t$，无视觉预见性；(b) 世界模型+规划：预测未来观测，[[Cross-Entropy Method|CEM]] 搜索动作（高计算代价）；(c) NavWAM：统一 Canvas 联合预测，无需外部规划。

### Figure 2: NavWAM 架构概览（Canvas 布局）

![[NavWAM_fig2_architecture.png]]

**说明**: 展示 9 帧潜在 Canvas 的完整布局。帧 0-3 为已观测（蓝色），帧 4-8 为预测目标（橙色）。[[Diffusion Transformer]] 去噪器统一处理所有帧，策略模式下单次前向传播输出动作块。

### Figure 3: 视觉一致性指标

![[NavWAM_fig3_consistency.png]]

**说明**: GO Stanford 数据集上各方法的视觉一致性（Subject Consistency）对比。NavWAM 零样本（0.668）显著优于 NWM（0.524），微调后 0.635，验证了联合训练改善了未来观测的视觉质量。

### Figure 4: 定性未来视图预测

![[NavWAM_fig4_qualitative.png]]

**说明**: NavWAM 在测试集上的未来帧预测示例（h=4 和 h=8）。模型能在不同场景下预测出语义一致的未来观测，h=4 时预测质量明显优于 h=8（时间更短，预测更准确）。

### Figure 5: 真实机器人导航轨迹

![[NavWAM_fig5_qualitative_realworld.png]]

**说明**: 四个室内场景（Office、Storage、Meeting Room、Hallway）下 NavWAM、NWM、OmniVLA 的导航轨迹对比。NavWAM 轨迹更平滑，能更可靠地到达目标（绿色星号），NWM 几乎完全失败。

### Figure 6: 执行过程中的未来视图预测

![[NavWAM_fig6_realworld_foresight.png]]

**说明**: 真实机器人执行过程中，NavWAM 实时预测的未来观测帧与实际观测帧对比。预测图像语义上与实际走廊/房间结构一致，验证了视觉预见性在真实部署中的有效性。

### Figure S1: 机器人平台（补充材料）

![[NavWAM_figS1_robot_platform.png]]

**说明**: 实验使用的 Direct Drive Tech Diablo 轮式机器人平台，搭载 Intel RealSense D455 RGB-D 相机和 NVIDIA Jetson AGX Orin 计算单元，3D 打印机架承载传感器。

### Table 1: GO Stanford 导航性能对比

| 方法 | ATE↓ (h=8) | RPE↓ (h=8) | 类型 |
|------|-----------|-----------|------|
| Cosmos Predict2 + CEM | 0.455 | 0.109 | 规划式 |
| NWM (N=120 CEM) | 0.453 | 0.107 | 规划式 |
| NavWAM（零样本） | 0.324 | 0.099 | 策略式 |
| **NavWAM w/ FT** | **0.192** | **0.070** | 策略式 |

**说明**: NavWAM 零样本即超越 NWM（CEM N=120），微调后 ATE 降低 57.6%（0.453→0.192），RPE 降低 34.6%，大幅领先所有基线。

### Table 2: 消融实验（监督目标对 ATE 的贡献，GO Stanford）

| 监督目标 | 推理方式 | ATE(h=4)↓ | ATE(h=8)↓ | RPE(h=4)↓ | RPE(h=8)↓ |
|---------|---------|----------|----------|----------|----------|
| 仅图像 | planning | 0.326 | 0.569 | 0.133 | 0.135 |
| 图像+动作+状态 | policy | 0.107 | 0.287 | 0.054 | 0.098 |
| **完整（+价值）** | **policy** | **0.076** | **0.192** | **0.037** | **0.070** |

**关键发现**: 价值监督对长时域（h=8）最关键——ATE 从 0.287→0.192（-33%）；直接策略推理远优于规划模式（0.107 vs 0.326 at h=4）。

### Table 3: 真实机器人导航（24 episodes，4 个室内场景）

| 方法 | Office (8) | Storage (6) | Meeting (6) | Hallway (4) | 总成功率 |
|------|-----------|-------------|-------------|------------|---------|
| NWM | 1/8 | 0/6 | 1/6 | 2/4 | 16.7% |
| OmniVLA | 4/8 | 4/6 | 3/6 | 3/4 | 58.3% |
| **NavWAM** | **6/8** | **6/6** | **4/6** | **3/4** | **79.2%** |

**关键发现**: NavWAM 在所有场景均领先，特别是 Storage 房间实现 100% 成功率；NWM 几乎失败（CEM 在实时系统中计算不可行）；NavWAM 以 2B 参数超越 7B 的 OmniVLA。

### Table 4: SiT 数据集直接策略对比（1400 episodes）

| 方法 | ATE(h=4)↓ | ATE(h=8)↓ | SR%(h=4)↑ | SR%(h=8)↑ |
|------|----------|----------|---------|---------|
| OmniVLA（7B） | 0.086 | 0.162 | 45.4% | 12.1% |
| **NavWAM（2B）** | **0.077** | **0.144** | **46.3%** | **15.9%** |

**说明**: NavWAM（2B）以更小模型规模超越 OmniVLA（7B），验证了联合世界-动作建模的参数效率优势。

### Table S4: 推理效率完整对比

| 方法 | FLOPs/action | 延迟 (ms) | 峰值 GPU (GB) | ATE(h=8)↓ |
|------|-------------|---------|-------------|---------|
| NWM (N=8) | 1,005 TF | 26,168 | 17.45 | 0.464 |
| NWM (N=32) | 3,901 TF | 69,737 | 44.32 | 0.460 |
| NWM (N=120) | 14,521 TF | 233,831 | 51.65 | 0.452 |
| NWM (N=240) | 28,993 TF | 469,320 | 52.53 | 0.470 |
| Cosmos+CEM (N=120) | 18,114 TF | 887,606 | 20.04 | 0.455 |
| **NavWAM（零样本）** | **4.45 TF** | **205.7** | **4.82** | **0.324** |
| **NavWAM w/ FT** | **4.45 TF** | **205.7** | **4.82** | **0.192** |

**关键发现**: NavWAM 比 NWM(N=120) 快 1136 倍（延迟），省 10.7 倍 GPU 显存，且性能更优；NWM 的 CEM 搜索在 N>8 后性能几乎饱和，说明外部规划器本身就是瓶颈。

### Table S5: 未来视图监督消融（GO Stanford h=8）

| 变体 | 未来图像监督 | ATE(h=4)↓ | ATE(h=8)↓ | RPE(h=4)↓ | RPE(h=8)↓ |
|------|------------|---------|---------|---------|---------|
| 无未来图像 | ✗ | 0.090 | 0.262 | 0.045 | 0.103 |
| **NavWAM 完整** | ✓ | **0.076** | **0.192** | **0.037** | **0.070** |

**说明**: 未来视图监督在长时域（h=8）带来 27% ATE 改善（0.262→0.192），是关键设计决策。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| HM3D | 802 场景 / 185,000 轨迹 | 仿真室内环境，测地线距离 | Phase 1 预训练 |
| RECON | 11,835 轨迹 | 真实户外/室内，间距 0.25m | Phase 2 微调 |
| SACSoN | 2,000 轨迹 | 真实校园，间距 0.255m | Phase 2 微调 |
| SCAND | 372 轨迹 | 真实室内，间距 0.38m | Phase 2 微调 |
| GO Stanford | 3,544 训练 / 30 测试 episodes | 室内办公，间距 0.12m | Phase 3 微调 / 离线评估 |
| SiT | 1,400 episodes（14 分段） | 结构化室内导航基准 | 直接策略对比（hold-out） |

### 实现细节

- **Backbone**: Cosmos-Predict2 2B（Video2World，480p→224×224）
- **优化器**: AdamW，学习率 $1 \times 10^{-4}$，余弦调度（线性预热）
- **Batch Size**: 8/GPU × 4 GPU = 有效批次 32
- **精度**: bfloat16 混合精度
- **训练噪声**: $\sigma_{\max}=200$，$\sigma_{\min}=10^{-2}$（Hybrid EDM 分布）
- **推理噪声**: $\sigma_{\max}=80$，$\sigma_{\min}=4$
- **动作损失权重**: $\lambda = 5$（相对于图像帧）
- **硬件**: 4× NVIDIA RTX PRO 6000（Blackwell，96GB）
- **推理频率**: ~5Hz（RTX PRO 6000 单卡，bfloat16）
- **机器人**: Direct Drive Tech Diablo 轮式平台，Intel RealSense D455 + Jetson AGX Orin，ROS 2 cmd_vel 接口

### 可视化结果

NavWAM 在四个真实室内场景中预测的未来帧与实际观测语义一致：走廊场景能预测前方拐角结构，储物间能预测货架空间。未来帧预测质量在 h=4 显著优于 h=8，符合时序预测规律。视觉一致性评分（Subject Consistency）0.668 优于 NWM 的 0.524。

---

## 批判性思考

### 优点
1. **优雅的统一框架**: 将策略、世界模型、价值函数三种角色融入单一扩散模型，设计简洁且有理论支撑
2. **极高推理效率**: 消除 CEM 搜索使延迟从 ~234 秒降至 ~0.2 秒，真正实现实时闭环控制（~5Hz）
3. **强大的 Sim-to-Real**: 仅用 ~14K 条真实轨迹微调，在 4 个真实场景达 79.2% 成功率
4. **参数效率高**: 2B 参数超越 7B 的 OmniVLA，说明联合建模比单纯扩大模型更有效

### 局限性
1. **评估规模有限**: 真实机器人仅 24 episodes，4 个环境，95% 置信区间 [59.5%, 90.8%]，统计置信度有限
2. **仅支持图像目标**: 不支持语言/物体目标，限制了实际应用场景
3. **室内导航专用**: 仅在结构化室内环境验证，户外、动态场景未测试
4. **价值引导采样未探索**: 论文提到价值函数可用于采样引导（Best-of-N），但未实现

### 潜在改进方向
1. **语言/物体目标**: 将 Canvas 中的目标帧替换为语言 embedding，扩展至 LLM 引导导航
2. **价值引导扩散采样**: 推理时用预测的价值函数引导去噪过程（Best-of-N），进一步提升导航成功率
3. **多机器人平台**: 在不同底盘（人形、四足）验证架构通用性

### 可复现性评估
- [ ] 代码开源（项目主页暂未发布代码）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（超参数、数据集规模均有说明）
- [x] 数据集可获取（HM3D、RECON、SACSoN、SCAND 均为公开数据集）

---

## 关联笔记

### 基于
- [[Cosmos Predict2]]: 2B 参数视频世界模型主干，NavWAM 在此基础上适配为导航策略
- [[Diffusion Policy]]: 扩散策略框架，NavWAM 将其扩展为多模态联合预测
- [[Action Chunking]]: 动作块预测机制，H=4 步一次性预测

### 对比
- [[NWM]]: 导航世界模型基线，NavWAM 消除了其依赖的 CEM 规划器
- [[OmniVLA]]: 7B 参数全模态 VLA 基线，NavWAM 以 2B 参数超越其性能

### 方法相关
- [[Latent Canvas]]: NavWAM 核心设计——将多模态信息编码为统一潜在帧序列
- [[Cross-Entropy Method]]: NavWAM 所替代的传统 CEM 规划方法
- [[Causal VAE]]: 将图像和非图像数据编码到共享潜在空间的关键组件
- [[Goal-Progress Value]]: 目标进度值辅助监督信号，对长时域预测关键
- [[DiT]]: NavWAM 使用的扩散变换器去噪主干
- [[Curriculum Learning]]: 三阶段训练课程策略

### 硬件/数据相关
- [[HM3D]]: Phase 1 仿真预训练数据集（Habitat-Matterport 3D，802 场景）
- [[RECON]]: Phase 2 真实机器人微调数据集

---

## 速查卡片

> [!summary] NavWAM: Navigation World Action Model
> - **核心**: 将导航世界模型转化为闭环策略，9 帧 Canvas 联合预测未来观测 + 动作 + 目标进度值
> - **方法**: [[Cosmos Predict2]] 主干 + [[Latent Canvas]] + 三阶段训练课程（HM3D→真实数据→领域微调）
> - **结果**: 真实机器人 79.2% 成功率（vs NWM 16.7%），推理延迟 205ms（vs NWM CEM 233 秒），FLOPs 降低 3266 倍
> - **代码**: 暂未发布（[项目主页](https://dachii-azm.github.io/navwam/)）

---

*笔记创建时间: 2026-06-13*
