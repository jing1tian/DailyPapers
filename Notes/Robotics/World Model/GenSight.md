---
title: "Seeing and Imagining as One: Unified Video Generation and 4D Perception for Embodied World Modeling"
method_name: "GenSight"
authors: [Haitang Feng, Zhuoman Liu, Zhennan Wang, Sheng Wang, Juelin Li, Chengzhi Zhao, Hexiang Wei, Lin Shao]
year: 2026
venue: arXiv
tags: [embodied-world-model, video-generation, dense-perception, video-diffusion, multi-task-learning, 4d-perception, robot-manipulation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.39232v1/
created: 2026-10-31
---

# 论文笔记：Seeing and Imagining as One: Unified Video Generation and 4D Perception for Embodied World Modeling

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开（arXiv 投稿）|
| 日期 | October 2026 |
| 项目主页 | [gensight-rbs.github.io](https://gensight-rbs.github.io) |
| 对比基线 | [[Cosmos]], [[Wan 2.2]], [[RynnWorld]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.39232) / Code: 未公开 |

---

## 一句话总结

> GenSight 用单个预训练[[Video DiT|视频扩散变换器]]同时完成未来视频生成与密集感知（深度、[[Embodiment Segmentation|本体分割]]、光流），三层统一设计——表示、骨干、预测空间——使"生成"与"感知"在一套参数内互相增强。

---

## 核心贡献

1. **统一时空表示**: 将 RGB、深度、[[Embodiment Segmentation|本体分割]]、首帧相对 2D 光流全部映射为 RGB 形式图像，经同一冻结视频自编码器编码，无需任何模态专用编解码器。
2. **共享骨干 + 轻量任务调制**: 单一 [[Video DiT]] 骨干，前 $k$ 层任务无关（task-agnostic prefix），后续层通过 [[AdaRMS|自适应 RMSNorm]] 零初始化调制实现任务感知，不破坏预训练初始化。
3. **统一速度预测空间**: 所有任务共享同一噪声样本，在整流流框架下同时回归视频速度场和各密集模态速度场；推理时只需单次前向通道即可从视频去噪轨迹中"读出"密集感知结果。

---

## 问题背景

### 要解决的问题

具身世界模型（Embodied World Model）需要同时具备**生成能力**（合成未来视频、支持规划与数据增强）和**感知能力**（输出深度、语义分割、光流，支持机器人控制与 3D 推理）。两类能力在现有方案中通常分属独立模块，相互割裂。

### 现有方法的局限

- **生成与感知分离**: 独立的视频生成模型（如 [[Wan 2.2]]、[[Cosmos]]）与独立的感知估计器（[[Depth Anything]]、[[SAM3]]、[[RAFT]]）串联，会传播伪影且缺乏跨模态约束。
- **模态专用头设计**: 现有联合方法通常为每个密集模态附加专用解码头，不共享主干，参数冗余，且预训练先验利用不充分。
- **缺乏统一预测空间**: 生成任务和感知任务在不同目标空间下优化（RGB velocity vs. 感知 loss），难以实现深层联合监督。

### 本文的动机

若将所有模态统一到同一时空表示和同一[[Rectified Flow|整流流]]预测框架，则生成任务的强先验（对场景结构、运动、光照的建模）天然为感知任务提供归纳偏置，反之感知监督也约束生成的几何一致性——两者在一套参数中协同涌现。

---

## 方法详解

### 模型架构

GenSight 采用**统一视频扩散变换器**架构：

- **输入**: 历史 RGB 帧 + 语言/本体条件 + 噪声视频帧
- **Backbone**: 36 层 [[Video DiT]]，隐层宽度 4096，由 Cosmos3-Nano 初始化；冻结 [[Wan 2.2]] VAE（压缩比 $(4,16,16)$）
- **核心模块**: 任务调制 [[AdaRMS|自适应 RMSNorm]]（任务感知前缀长度 $k=16$ 后生效）
- **输出**: 多步未来 RGB 视频 + 对齐的深度图 / [[Embodiment Segmentation|本体分割图]] / 2D 光流（生成模式），或单步密集感知（感知模式）
- **文本分词器**: Qwen3-VL-8B

### 核心模块

#### 模块 1: 统一模态表示

**设计动机**: 使所有密集模态均可通过同一冻结 VAE 处理，无须引入任何模态专用编解码器。

**具体实现**:
- **[[深度估计|深度]] $D$**: 对数深度经训练集 1–99 百分位归一化 → 量化为 768 级 → **RGB 温度计编码（Thermometer Code）**，解码时三通道相加。
- **[[Embodiment Segmentation|本体分割]] $S$**: 二值掩码复制到 RGB 三通道（0/255），解码后以 127 为阈值二值化。
- **[[Optical Flow|2D 光流]] $F$**: 流缩放到 1280 像素画布，低运动像素（最大位移 < 100px）置零，[[HSV光流编码]]（方向→色调，幅度→饱和度，明度固定最大），零/无效运动设为白色。
- **[[场景流|3D 场景流]] $F^{3D}$**: 由预测深度和 2D 光流通过反投影恢复（见公式 1–2）。

#### 模块 2: 共享骨干与轻量任务调制

**设计动机**: 利用预训练视频 DiT 强大先验的同时，以极少可训练参数实现任务感知。

**具体实现**:
- 骨干共 $L$ 层（默认 36）；前 $k$ 层（默认 16）**完全共享**，计算任务无关特征。
- 后 $L-k$ 层通过 [[AdaRMS|自适应 RMSNorm]] 进行任务调制：每层每任务学习尺度增量 $\Delta s$、偏置增量 $\Delta b$、残差门控增量 $\Delta g$（见公式 3），**零初始化**以保留预训练模型。
- 每个任务共享同一骨干权重，仅 $(\Delta s, \Delta b, \Delta g)$ 不同，参数极轻量。

#### 模块 3: 统一速度预测空间

**设计动机**: 使[[Multi-Task Learning|多任务]]训练在统一噪声空间下联合监督，让噪声项自然消去，只剩跨模态干净表示差异。

**具体实现**:
- 所有任务共享同一噪声样本 $\epsilon$，使用[[Rectified Flow|整流流]]插值构建噪声输入（见公式 4）。
- 视频速度目标与密集模态速度目标之差在代数上恰好等于两者干净潜在表示之差（见公式 5–7），噪声项消去。
- 推理时密集潜在表示可从视频去噪估计一步重建（见公式 8）。

---

## 关键公式

### 公式 1–2: [[场景流|3D 场景流]] 反投影恢复

$$
P_0 = D_0(p)\,K^{-1}\tilde{p}, \qquad P_t = D_t(p_t)\,K^{-1}\tilde{p}_t
$$

$$
F^{3D}_{0\to t}(p) = P_t - P_0
$$

**含义**: 利用预测深度和 2D 光流，将像素 $p$ 在首帧和第 $t$ 帧的反投影 3D 坐标相减，得到每像素 3D 运动向量。

**符号说明**:
- $p$: 首帧像素坐标（齐次坐标 $\tilde{p}$）
- $p_t$: 第 $t$ 帧对应像素坐标（由 2D 光流 $F$ 得到）
- $D_0(p),\, D_t(p_t)$: 首帧和第 $t$ 帧的预测深度值
- $K$: 相机内参矩阵
- $P_0, P_t$: 两帧中该像素的 3D 世界坐标

### 公式 3: [[AdaRMS|自适应 RMSNorm 任务调制]]

$$
\tilde{h} = (1+\Delta s)\odot \mathrm{RMSNorm}(h) + \Delta b
$$

$$
h' = h + (1+\Delta g)\odot \mathcal{F}(\tilde{h})
$$

**含义**: 在每个 DiT 层中，对归一化特征施加任务特定的缩放/偏移，并通过可学习门控 $\Delta g$ 调节残差贡献；所有增量零初始化，保证训练初始阶段等价于原预训练模型。

**符号说明**:
- $h$: 该层输入特征
- $\Delta s, \Delta b$: 任务特定的 [[RMSNorm]] 尺度和偏置增量（零初始化）
- $\Delta g$: 任务特定的残差门控增量（零初始化）
- $\mathcal{F}(\cdot)$: 该层的前馈/注意力子网络
- $h'$: 调制后输出特征

### 公式 4–6: [[Rectified Flow|整流流]]统一噪声输入与速度目标

$$
x_\sigma = (1-\sigma)z_V + \sigma\epsilon
$$

$$
v_V^\star = \epsilon - z_V, \qquad v_m^\star = \epsilon - z_m
$$

$$
v_m^\star - v_V^\star = z_V - z_m
$$

**含义**: 所有任务（视频 $V$ 和各密集模态 $m$）共享同一噪声样本 $\epsilon$，在整流流框架下各自的速度目标之差恰好等于干净潜在表示之差，**噪声项完全消去**，实现纯粹的跨模态监督。

**符号说明**:
- $\sigma \in [0, 1]$: 整流流时间步（噪声级别）
- $z_V$: 干净视频潜在表示
- $z_m$: 干净密集模态 $m$ 的潜在表示
- $\epsilon \sim \mathcal{N}(0, I)$: 共享噪声样本
- $v_V^\star, v_m^\star$: 视频和模态 $m$ 的目标速度

### 公式 7: 联合训练损失

$$
\mathcal{L} = w_V \mathcal{L}_V + \sum_{m \in \{S, D, F\}} w_m \mathcal{L}_m
$$

其中每个损失包含速度回归项和空间梯度匹配项：

$$
\mathcal{L}_m = \left\| v_\theta^m - v_m^\star \right\|_2^2 + \lambda \cdot \left\| \nabla v_\theta^m - \nabla v_m^\star \right\|_2^2
$$

**含义**: 视频和各密集模态（分割 $S$、深度 $D$、光流 $F$）的速度回归损失加权求和，密集模态额外附加空间梯度匹配项保留边缘细节；密集任务损失在前 800 步余弦预热。

**符号说明**:
- $w_V = 1$: 视频损失权重
- $w_S = w_D = w_F = 0.25$: 各密集模态损失权重
- $\lambda$: 空间梯度损失权重
- $\nabla$: 空间梯度算子

### 公式 8: 推理时密集潜在重建（生成模式）

$$
\hat{z}_m = \hat{z}_V - (\hat{v}_m - \hat{v}_V)
$$

**含义**: 在视频去噪轨迹达到密集模态读出噪声级 $\sigma_m^\star$ 时，用视频当前估计 $\hat{z}_V$ 减去两速度预测之差，无需对密集模态单独运行去噪轨迹，一步直接重建密集潜在表示。

**符号说明**:
- $\hat{z}_V$: 当前步视频干净潜在估计
- $\hat{v}_V, \hat{v}_m$: 模型预测的视频和模态 $m$ 的速度
- $\sigma_m^\star$: 模态 $m$ 的最优读出噪声级（深度≈0.8，分割≈0.4，光流≈0.9）

---

## 关键图表

### Figure 1: 系统总览（Teaser）

![Figure 1](https://arxiv.org/html/2610.39232v1/teaser.png)

**说明**: GenSight 两种推理模式概览。生成模式（上）：以历史帧为条件，生成多步未来视频并同步输出深度、[[Embodiment Segmentation|本体分割]]、2D 光流。感知模式（下）：以已观测视频为输入，单步预测对应的密集感知结果，同时利用深度 + 光流反投影得到 [[场景流|3D 场景流]]。

### Figure 2: 方法架构（Pipeline）

![Figure 2](https://arxiv.org/html/2610.39232v1/pipeline.png)

**说明**: 共享 [[Video DiT]] 骨干（前 $k$ 层任务无关，后续层经[[AdaRMS|自适应 RMSNorm]]任务调制）的架构示意，以及生成模式（多步去噪 + 密集读出）和感知模式（单步前向）两种推理路径。

### Figure 3: 预测目标空间对比

![Figure 3](https://arxiv.org/html/2610.39232v1/vel_space.png)

**说明**: 三种目标空间设计的对比：(a) 独立噪声 + 独立速度目标（基线），(b) 共享噪声 + 独立速度目标，(c) GenSight 的共享噪声 + **统一速度预测空间**（噪声项消去，仅剩跨模态干净表示差异）。

### Figure 4: 定性对比结果

![Figure 4](https://arxiv.org/html/2610.39232v1/qualitative.png)

**说明**: 在 RoboMIND 2.0（Franka 和 UR5 机械臂）上的定性对比。生成模式行：GenSight vs. [[Wan 2.2]]、[[Cosmos]]（视频质量更清晰，动态更真实）；感知模式行：深度、本体分割、光流各与参考标注对比，显示 GenSight 几何结构更精准。

### Figure 5: 跨本体生成（RoboTwin 2.0）

![Figure 5](https://arxiv.org/html/2610.39232v1/robotwin.png)

**说明**: 同一 RoboTwin 2.0 初始帧，分别以"机械臂"和"人手"为本体条件，GenSight 生成两种截然不同但动作语义一致的未来视频——验证本体条件控制能力。

### Figure 6: 消融实验

![Figure 6](https://arxiv.org/html/2610.39232v1/abla.png)

**说明**: 在 UR5 子集上的两组消融实验。左图：任务无关前缀长度 $k$（0, 4, 8, 16, 32）对视频生成（FVD/FID）和密集感知指标的影响——中间值（$k=16$）达最优权衡。右图：密集读出噪声级 $\sigma_m^\star$ 对各模态性能的影响——深度（≈0.8）、分割（≈0.4）、光流（≈0.9）各有独立最优值。

### Figure A: 3D 轨迹追踪定性结果

![Figure A](https://arxiv.org/html/2610.39232v1/tracking_3d.png)

**说明**: GenSight 预测的深度 + 2D 光流联合反投影得到 3D 点轨迹，与 [[Track4World]] 基线对比；GenSight 的轨迹在遮挡和机械臂高速运动区域更连贯。

### Figure B: RoboMIND 2.0 – Franka 机械臂定性结果

![Figure B](https://arxiv.org/html/2610.39232v1/franka.png)

**说明**: Franka 本体的完整生成帧序列（RGB）+ 感知输出（深度、分割、光流）与 ground truth 对比。

### Figure C: RoboMIND 2.0 – UR5 机械臂定性结果

![Figure C](https://arxiv.org/html/2610.39232v1/ur5.png)

**说明**: [[UR5]] 本体的生成 + 感知定性结果，覆盖多个未见任务。

### Figure D: RoboMIND 2.0 – Tianyi 机械臂定性结果

![Figure D](https://arxiv.org/html/2610.39232v1/tianyi.png)

**说明**: Tianyi 本体的生成 + 感知定性结果，验证跨机器人平台泛化性。

### Figure E: RoboMIND 2.0 – Tienkung 机械臂定性结果

![Figure E](https://arxiv.org/html/2610.39232v1/tienkung.png)

**说明**: [[TienKung]] 本体的生成 + 感知定性结果。

### Figure F: EgoTable（人手）定性结果

![Figure F](https://arxiv.org/html/2610.39232v1/egotable.png)

**说明**: 以人手为本体的 EgoTable 数据集上的生成 + 感知输出，展示 GenSight 对非机械臂本体的泛化能力。

### Figure G: RoboTwin 2.0 仿真定性结果

![Figure G](https://arxiv.org/html/2610.39232v1/robotwin_2.0.png)

**说明**: [[RoboTwin 2.0]] 仿真环境（AgileX 本体）上的完整生成 + 感知对比，进一步验证跨场景泛化。

### Table 1: 生成模式对比（RoboMIND 2.0 + RoboTwin 2.0）

**A. RoboMIND 2.0 测试集（600 episodes，24 tasks）**

| 方法 | FVD ↓ | FID ↓ | PSNR ↑ | LPIPS ↓ | Seg IoU ↑ | Depth AbsRel ↓ | Depth δ1 ↑ | Flow EPE ↓ | Flow nEPE ↓ |
|------|-------|-------|--------|---------|-----------|----------------|------------|------------|-------------|
| [[Wan 2.2]] | 69.37 | 21.90 | — | 0.2984 | — | — | — | — | — |
| [[Cosmos]] (Cosmos3-Nano) | 44.97 | 18.66 | — | 0.2584 | — | — | — | — | — |
| [[RynnWorld]] (RynnWorld4D) | — | — | — | — | — | — | — | — | — |
| **GenSight（本文）** | **14.55** | **16.12** | — | **0.2296** | — | **7.88%** | **93.17%** | — | — |

**B. RoboTwin 2.0 仿真（300 episodes，50 tasks）**

| 方法 | FVD ↓ | FID ↓ | PSNR ↑ | LPIPS ↓ | Flow EPE ↓ | Flow nEPE ↓ | Flow AAE ↓ |
|------|-------|-------|--------|---------|------------|-------------|------------|
| [[RynnWorld]] (RynnWorld4D) | 较高 | 较高 | 较低 | 较高 | **较低（raw）** | 较高 | 较高 |
| **GenSight（本文）** | **最低** | **最低** | **最高** | **最低** | 较高（raw） | **最低** | **最低** |

> 注：RynnWorld4D 在原始 EPE 上略低，但 GenSight 在 nEPE 和 AAE（对参考运动归一化的指标）上更优，作者认为后两者在参考运动较小时更具区分意义。

**表格说明**: GenSight 在所有视频生成指标（FVD/FID/LPIPS）上大幅超越 Wan 2.2 和 Cosmos3-Nano，并在深度感知指标（AbsRel/δ1）上显著领先。

### Table 2: 感知模式对比

**分割（IoU ↑）**

| 方法 | Seg IoU |
|------|---------|
| [[SAM3]] | 73.5 |
| GenCeption | 67.5 |
| **GenSight** | **88.7** |

**深度（RMSE ↓ / AbsRel ↓ / δ1 ↑）**

| 方法 | RMSE ↓ | AbsRel ↓ | δ1 ↑ |
|------|--------|----------|------|
| [[Depth Anything]] (Depth Anything 3) | 较高 | 较高 | 较低 |
| [[VGGT-Ω]] | 13.11 | 5.9% | 94.5% |
| **GenSight** | **4.76** | **2.4%** | **99.1%** |

**2D 光流（EPE ↓）**

| 方法 | EPE ↓ |
|------|-------|
| [[RAFT]] | 11.2 |
| [[Track4World]] | 6.8 |
| **GenSight** | **5.2** |

**3D 场景流（ADE/FDE ↓）**

| 方法 | 是否需要 GT 深度 | ADE ↓ | FDE ↓ |
|------|----------------|-------|-------|
| RAFT-3D | 需要 GT 深度 | **最低** | **最低** |
| [[Track4World]] | 不需要 | 较高 | 较高 |
| **GenSight** | 不需要 | **次优（无 GT 深度方法中最优）** | — |

**表格说明**: 感知模式下，GenSight 在分割 IoU（+15.2 vs. SAM3）、深度（RMSE 缩小 63.7% vs. VGGT-Ω）、2D 光流（EPE 降低 23.5% vs. Track4World）上全面领先；3D 场景流在不使用 GT 深度的方法中最优。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboMIND]] 2.0 | UR5, Franka, Tianyi, Tienkung 多机器人 | 真实机器人多本体操作数据 | 训练 + 测试 |
| EgoTable（in-house） | 以人类手部桌面操作为主 | 自采第一视角人手数据 | 训练 |
| [[RoboTwin 2.0]] | 仿真，AgileX 本体 | 50 任务，300 episodes 测试 | 测试 |
| **合计训练集** | **221,044 episodes（≈1,050 小时）** | — | — |

- **评估集**: RoboMIND held-out OOD 任务（训练任务的 5%）+ RoboMIND 官方测试集（600 episodes，24 tasks）+ RoboTwin 2.0 仿真测试集（300 episodes，50 tasks）。

### 实现细节

- **Backbone**: 36 层 Video DiT，隐层宽度 4096，由 Cosmos3-Nano 初始化
- **VAE**: 冻结 [[Wan 2.2]] VAE，时空压缩比 $(4, 16, 16)$
- **文本分词器**: Qwen3-VL-8B
- **优化器**: Fused AdamW（$\beta=(0.9, 0.95)$）
- **学习率**: 峰值 $8\times10^{-5}$，余弦衰减至 $1\times10^{-5}$，共 8,000 步
- **梯度裁剪**: 1.0
- **序列打包**: 每设备 12,288 token 预算
- **计算资源**: 64× NVIDIA A800 GPU，约 85 小时（≈5,400 GPU-hours）
- **消融协议**: 在 UR5 训练集子集上训练至 3,000 步，评估 8 个未见任务（200 episodes）

### 可视化结果

- **生成质量**: GenSight 生成的 Franka/UR5 操作视频中机械臂运动轨迹更连贯，物体形变更物理合理，与 Wan 2.2 和 Cosmos3-Nano 相比伪影显著减少（Figure 4）。
- **密集感知一致性**: 感知模式输出的深度边缘与机械臂/物体轮廓吻合，分割掩码二值化后几乎无噪声泡。
- **3D 追踪**: 与 Track4World 对比，GenSight 在机械臂高速运动和遮挡区域的 3D 点轨迹更平滑、漂移更小（Figure A）。
- **跨本体条件生成**: 同一初始帧，机械臂条件 vs. 人手条件产生视觉截然不同但运动语义一致的未来视频（Figure 5）。

---

## 批判性思考

### 优点

1. **三层统一设计的一致性**: 表示层（同一 VAE）、架构层（共享主干 + 轻量调制）、预测空间层（共享噪声）三层完全统一，内部一致性高，设计优雅。
2. **推理时零额外成本**: 密集感知不需要额外去噪轨迹，与视频生成共享同一前向通道，实用性强。
3. **多本体泛化**: 在四种真实机器人 + 人手 + 仿真场景上均验证，泛化能力扎实。
4. **感知性能超越专用模型**: 在[[深度估计|深度]]（RMSE 降低 63.7%）、分割 IoU（+15.2）上大幅超越 VGGT-Ω 和 SAM3 等专用感知模型，验证了生成先验对感知的实质性增益。

### 局限性

1. **无动作预测或闭环控制**: GenSight 当前输出视频和密集感知，不输出机器人动作，尚未形成完整的 VLA 闭环控制系统。
2. **密集监督依赖外部估计器**: 深度标注来自传感器补全，分割/光流来自 off-the-shelf 估计器（如 [[SAM3]]、[[RAFT]]），噪声和偏差可能传入训练数据。
3. **计算代价较高**: 5,400 GPU-hours 的训练成本限制了学术界复现。
4. **缺乏闭环和下游任务验证**: 尚未在机器人在线规划或 IL/RL 下游任务中验证生成质量对控制性能的实际提升。

### 潜在改进方向

1. **接入动作预测头**: 将动作预测模块（如 flow-based action diffusion）集成到统一速度预测空间，形成完整具身世界动作模型。
2. **自监督密集标注**: 摆脱外部感知估计器，用模型自身的循环一致性约束在线精化密集标注质量。
3. **在线闭环验证**: 在仿真或真实机器人上用 GenSight 做 model-based planning，验证感知辅助的实际决策收益。
4. **Scaling 研究**: 探索 backbone 规模（模型大小、数据量）对生成质量和感知性能的 scaling law。

### 可复现性评估

- [ ] 代码开源（未发布）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（论文和附录描述充分）
- [x] 数据集可获取（RoboMIND 2.0、RoboTwin 2.0 公开可用）

---

## 关联笔记

### 基于

- [[Cosmos]]: GenSight 的 Video DiT 骨干由 Cosmos3-Nano 初始化
- [[Wan 2.2]]: 冻结 VAE 编解码器来源
- [[Rectified Flow]]: 训练和推理的核心整流流框架
- [[Video Diffusion Model]]: GenSight 所属的模型类别

### 对比

- [[RynnWorld]]: RynnWorld4D，主要对比基线，GenSight 在视频生成所有指标上超越
- [[Wan 2.2]]: 视频生成基线，GenSight FVD 降低 79%（14.55 vs. 69.37）
- [[Cosmos]]: 视频生成基线，GenSight FVD 降低 68%（14.55 vs. 44.97）
- [[VGGT-Ω]]: 深度感知基线，GenSight RMSE 降低 63.7%
- [[SAM3]]: 分割基线，GenSight IoU 高出 15.2 点
- [[Track4World]]: 光流 + 3D 追踪基线，GenSight 2D EPE 降低 23.5%
- [[RAFT]]: 光流基线，GenSight EPE 降低 53.6%

### 方法相关

- [[Video DiT]]: 核心骨干架构
- [[AdaRMS]]: 任务调制机制
- [[RMSNorm]]: 归一化层基础组件
- [[Rectified Flow]]: 统一速度预测空间
- [[Flow Matching]]: 整流流训练框架
- [[Multi-Task Learning]]: 密集模态联合训练策略
- [[Optical Flow]]: 密集感知模态之一
- [[深度估计]]: 密集感知模态之一
- [[Embodiment Segmentation]]: 密集感知模态之一（机器人/手部分割）
- [[场景流]]: 由深度 + 光流反投影推导的 3D 运动表示
- [[HSV光流编码]]: 光流可视化编码方式
- [[Dense Prediction]]: 输出类型总称
- [[World Model]]: GenSight 所属的具身世界模型范式

### 硬件/数据相关

- [[RoboMIND]]: 主要训练 + 测试数据集（含 UR5、Franka、Tianyi、Tienkung）
- [[RoboTwin 2.0]]: 仿真测试基准
- [[UR5]]: 训练数据中的机器人本体之一

---

## 速查卡片

> [!summary] GenSight: Unified Video Generation and 4D Perception
> - **核心**: 单个 Video DiT 同时做未来视频生成和密集感知（深度/分割/光流），统一表示 + 共享骨干 + 共享噪声空间
> - **方法**: 整流流统一速度预测空间，AdaRMS 零初始化任务调制（$k=16$ 前缀），无模态专用头
> - **结果**: RoboMIND FVD 14.55（vs. Cosmos 44.97），感知 Depth δ1 99.1%（vs. VGGT-Ω 94.5%），Seg IoU 88.7（vs. SAM3 73.5）
> - **代码**: 未公开（项目页: gensight-rbs.github.io）

---

*笔记创建时间: 2026-10-31*
