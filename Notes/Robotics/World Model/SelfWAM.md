---
title: "SelfWAM: A Self-Grounded Unified World Action Model for Fast Robot Control"
method_name: "SelfWAM"
authors: [Bikang Pan, Fan Liu, Haotao Lu, Jingya Wang, Ye Shi]
year: 2026
venue: arXiv
tags: [world-action-model, robot-manipulation, video-diffusion, self-grounding, action-conditioning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.00725v1
created: 2026-08-05
---

# 论文笔记：SelfWAM

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开 |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[Flash-WAM]] / [[GigaWorldPolicy0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.00725) / Code — |

---

## 一句话总结

> SelfWAM 在 FastWAM 的两流 MoT 架构上引入"干净动作条件化"与"机器人自我掩码预测"，让世界模型真正对动作敏感，同时保持推理时仅运行动作流的高效性。

---

## 核心贡献

1. **[[Clean Action Conditioning|干净动作条件化]]**: 将无噪声动作副本（扩散时间步 $t=0$）注入视觉 token 的 cross-attention，但屏蔽于动作预测 query，避免目标泄漏，同时使视频预测真正感知动作语义。
2. **[[Robot Self-Mask Prediction|机器人自我掩码预测]]**: 以机器人分割掩码作为辅助视频预测目标，复用同一视频骨干（仅改 prompt），强迫模型聚焦于动作引发的身体运动，提升 LPIPS 32.5%、PSNR 12.1%、FVD-I3D 24.8%。
3. **零推理开销**: 训练时增加两个信息流，但推理时两者均可省略，动作延迟仅增加 0.97%（320.7ms → 323.9ms），显存不变（13.959 GB）。

---

## 问题背景

### 要解决的问题

现有 [[World Action Model]] 在训练时引入了未来视频预测分支，以期为动作策略学习更丰富的世界表示；但这种视频预测并未对特定动作敏感——即同一场景下不同动作应产生不同未来，然而实验发现 [[Fast-WAM]] 等模型的未来视频预测对动作扰动几乎不变。

### 现有方法的局限

- [[Fast-WAM]]：两流 [[Mixture-of-Transformers|MoT]] 训练有效，但视频 token 无法区分不同动作输入，视频分支沦为单纯场景建模而非"动作后果模型"
- 其他 WAM（GigaWorld-Policy、[[GeoSem-WAM]] 等）：或引入额外几何监督，或引入掩码 token，但均未解决"视频预测是否对动作敏感"这一根本问题
- 训练时给视频 token 注入动作信息的直觉方案：直接将带噪动作 token 暴露给视频 query 会污染动作预测（目标泄漏），因为视频 token 可借此间接访问未去噪动作标签

### 本文的动机

将动作分成两份：带噪版本继续参与动作去噪，干净版本（$t=0$，零噪声）单独提供给视频 query 作为条件。这样视频预测感知了"要执行的动作是什么"，而动作预测 query 被严格屏蔽于干净动作副本之外，不产生泄漏。同时，让模型预测机器人自我掩码（身体遮罩）迫使模型关注动作引发的空间运动，提供比原始 RGB 更精确的动作反事实监督。

---

## 方法详解

### 模型架构

**SelfWAM** 在 [[Fast-WAM]] 基础上扩展，采用 [[Mixture-of-Transformers|两流 MoT]] 架构，骨干为 [[Wan 2.2-5B]] 视频扩散 Transformer：

- **输入**: 语言指令 $l$ + 三视角观测 $o_t$（拼成 384×320 T 形合图） + 本体感知 $s_t$ + 带噪动作 chunk $\tilde{A}$ + 带噪未来视频 latent $\tilde{V}$
- **额外输入（训练专用）**: 干净动作 $A$（$t=0$，无噪声副本）+ 未来自我掩码视频目标 $M$
- **Backbone**: WAN 2.2-5B Video DiT
- **核心模块**: [[Clean Action Conditioning]] + [[Robot Self-Mask Prediction]] + [[Mixture-of-Transformers|MoT 不对称注意力]]
- **输出（推理）**: 32 步 [[Action Chunking|动作块]] $A_{t:t+32}$
- **总参数**: WAN 2.2-5B 级别（~50亿参数量）

### 核心模块

#### 模块 1: Clean Action Conditioning（干净动作条件化）

**设计动机**: 让未来视频预测利用 [[Action Chunking|动作块]] 的精确语义，将视频分支变成真正的"动作后果模型"，同时防止目标泄漏至动作预测 query。

**具体实现**:
- 将 $A$（即带噪动作在扩散时间步 $t=0$ 的版本）单独编码，与对应带噪 token 共享时序位置编码，建立语义对齐
- 注意力规则：
  - 视频 query $\mathcal{Q}_{\tilde{V}}$ 可访问 $\{O,\, \tilde{V},\, A\}$（干净动作可见）
  - 动作 query $\mathcal{Q}_{\tilde{A}}$ 只可访问 $\{O,\, \tilde{A}\}$（干净动作**不可见**）
- 数值验证：干净/带噪路径在各层激活的 NRMSE 最大误差约 $10^{-5}$，余弦距离约 $6\times10^{-11}$，确认两路径完全独立

#### 模块 2: Robot Self-Mask Prediction（机器人自我掩码预测）

**设计动机**: RGB 预测可用场景先验绕开真实动作影响；机器人身体掩码仅包含动作引起的位移，提供更纯净的动作感知训练信号，即[[Robot Self-Mask Prediction|自我接地（Self-Grounding）]]。

**具体实现**:
- 使用 [[RobotSeg]] 模型离线生成机器人分割掩码作为训练标签
- 复用视频骨干，仅替换文本 prompt：
  - RGB 实例：`"A video recorded from a robot's point of view executing the following instruction: [task]"`
  - Self-Mask 实例：`"A robot-arm mask video recorded from a robot's point of view executing the following instruction: [task]"`
- 训练时 RGB 与 Self-Mask 实例按 9:1 采样
- Self-Mask 实例不携带带噪动作 token，因此不优化动作损失

#### 模块 3: 双推理模式

| 推理模式 | 输入流 | 延迟 | 用途 |
|----------|--------|------|------|
| 仅动作 | $O,\, \tilde{A}$ | 323.9 ms | 部署（**常用**） |
| 动作 + 视频 | $O,\, \tilde{A},\, \tilde{V}$ | 677.5 ms | 验证 / 数据合成 |

---

## 关键公式

### 公式 1: [[World Action Model|WAM 联合建模目标]]

$$
\mathcal{L}_{\text{WAM}} = \mathbb{E}_{(o,l,o',a,m) \sim \mathcal{D}} \left[ -\log p(o', m, a \mid o, l) \right]
$$

**含义**: 联合建模未来 RGB $o'$、自我掩码 $m$ 和动作块 $a$ 的对数似然。

**符号说明**:
- $o$: 当前多视角观测
- $l$: 语言指令
- $o'$: 未来 RGB 视频（8 帧，时序下采样自 32 步）
- $m$: 机器人自我掩码视频
- $a$: 32 步动作块

---

### 公式 2: [[Clean Action Conditioning|干净动作条件化]] 加权损失

$$
\mathcal{L} = \lambda_{\text{act}} \cdot \mathcal{L}_{\text{act}} + \lambda_{\text{video}} \cdot \mathcal{L}_{\text{rgb/mask}}
$$

**含义**: 动作损失与视频（RGB + 自我掩码）损失等权求和，$\lambda_{\text{act}} = \lambda_{\text{video}} = 1$。

**符号说明**:
- $\mathcal{L}_{\text{act}}$: 动作去噪扩散损失
- $\mathcal{L}_{\text{rgb/mask}}$: 视频去噪扩散损失（含 RGB 实例与 Self-Mask 实例）
- $\lambda_{\text{act}},\, \lambda_{\text{video}}$: 等权系数（均为 1）

---

### 公式 3: [[Mixture-of-Transformers|MoT 不对称注意力掩码]]

$$
\text{Attention}(\mathcal{Q}_x,\, K,\, V;\; \mathcal{M}_x), \quad \mathcal{M}_x \in \{0, -\infty\}^{N \times N}
$$

**含义**: 对 query 类型 $x \in \{\tilde{V}, \tilde{A}\}$ 使用不同掩码矩阵 $\mathcal{M}_x$，实现"视频 query 可见干净动作，动作 query 不可见"的不对称信息流。

**符号说明**:
- $\mathcal{Q}_{\tilde{V}}$: 视频 token query，访问 $\{O,\, \tilde{V},\, A\}$
- $\mathcal{Q}_{\tilde{A}}$: 动作 token query，访问 $\{O,\, \tilde{A}\}$
- $A$: 干净动作（$t=0$，零噪声），仅用于训练
- $\mathcal{M}_x$: 对应 query 类型的二值掩码

---

## 关键图表

### Figure 1: SelfWAM 整体概览

![Figure 1](https://arxiv.org/html/2608.00725v1/x1.png)

**说明**: 对比动作无关视频预测（上）与 SelfWAM 动作感知视频预测（下）。给定完全不同的动作（向左 vs 向右），基线模型预测相同未来，而 SelfWAM 依据 [[Clean Action Conditioning|干净动作条件化]] 生成方向一致的机器人自我掩码移动，与真实运动轨迹吻合。

### Figure 2: 架构与注意力设计

![Figure 2](https://arxiv.org/html/2608.00725v1/x2.png)

**说明**: 三个子图。(a) 模态专属 [[Mixture-of-Transformers|MoT]] 结构；(b) 不对称注意力设计——干净动作 $A$ 向视频 query 开放但对动作 query 屏蔽；(c) 两种推理模式（仅动作 / 动作 + 视频）。

### Figure 3: 真实机器人四任务执行

![Figure 3](https://arxiv.org/html/2608.00725v1/x3.png)

**说明**: [[ALOHA]] 机械臂在四项抓放任务（杯子放架、笔插杯、鼠标放垫、纸球入桶）上的代表性闭环轨迹，SelfWAM 平均成功率 95%，高于 FastWAM（85%）和 π₀.₅（90%）。

### Figure 4: 方向性动作扰动分析

![Figure 4](https://arxiv.org/html/2608.00725v1/x4.png)

**说明**: 在关节空间施加 ±0.10 rad 偏移（前 8 步），动作无关基线预测自我掩码完全不变；SelfWAM 预测掩码随扰动方向性移动，与真实关节运动一致，验证 [[Clean Action Conditioning]] 的动作敏感性。

### Figure S1: 时序采样与多视角合图

![Figure S1](https://arxiv.org/html/2608.00725v1/x5.png)

**说明**: 32 步动作块对应的 8 帧视频序列时序下采样方案，三路相机视图拼合为 384×320 T 形合图。

### Figure S2: 连续动作插值

![Figure S2](https://arxiv.org/html/2608.00725v1/x6.png)

**说明**: 在静止保持动作（0%）与示教动作（100%）之间做插值，基线模型未来预测几乎不变，SelfWAM 随插值比例平滑过渡，定性验证动作连续响应能力。

### Figure S3: 干净/带噪路径一致性

![Figure S3](https://arxiv.org/html/2608.00725v1/x7.png)

**说明**: 逐 Transformer 层测量干净动作路径与带噪路径的 NRMSE 和余弦距离，最大误差分别约 $10^{-5}$ 和 $6 \times 10^{-11}$，数值验证两路径信息完全隔离。

---

### Table 1: RoboTwin 2.0 仿真成功率

| Method | Clean | Random | Average |
|--------|-------|--------|---------|
| π₀ | 65.92% | 58.40% | 62.16% |
| π₀.₅ | 82.74% | 76.76% | 79.75% |
| Motus | 88.66% | 87.02% | 87.84% |
| GigaWorld-Policy | 86.36% | 85.04% | 85.70% |
| FastWAM | 91.82% | 91.86% | 91.84% |
| **SelfWAM** | **92.16%** | **93.08%** | **92.62%** |

**说明**: [[RoboTwin 2.0]] 基准 50 项双臂操作任务（Clean + Random 布局），每设置 100 次 rollout。SelfWAM 超越 FastWAM 0.78 个百分点，在 Random 布局下更明显（+1.22%），说明更强的动作感知有助于鲁棒泛化。

---

### Table 2: 真实机器人成功率（10 次 rollout/任务）

| Task | SelfWAM | FastWAM | π₀.₅ |
|------|---------|---------|------|
| Brown Cup on Stand | 100% | 80% | 90% |
| Black Pen in Cup | 90% | 90% | 90% |
| Black Mouse on Pad | 100% | 80% | 90% |
| Paper Ball in Bin | 90% | 90% | 90% |
| **Average** | **95%** | **85%** | **90%** |

**说明**: [[ALOHA]] 平台真实抓放任务，SelfWAM 在"杯子放架"和"鼠标放垫"两项精细定位任务上达到 100%，领先 FastWAM 20 个百分点。

---

### Table 3: 未来视频预测质量

| Model | LPIPS↓ | PSNR↑ | FVD-I3D↓ |
|-------|--------|-------|----------|
| FastWAM | 0.0636 | 29.24 | 45.92 |
| SelfWAM | **0.0429** | **32.79** | **34.55** |

**说明**: [[Clean Action Conditioning]] 使感知相似度（LPIPS）提升 32.5%，像素保真度（PSNR）提升 12.1%，视频分布质量（FVD-I3D）提升 24.8%，证明视频分支成为真正的动作后果模型。

---

### Table 4: 消融实验（RoboTwin 2.0 Average）

| 配置 | Average | 说明 |
|------|---------|------|
| FastWAM（基线） | 91.84% | 无干净动作条件化，无自我掩码 |
| + 干净动作条件化 | 90.80% | 引入 CA 但无自我掩码，略降 |
| + 干净动作条件化 + 自我掩码 | **92.62%** | 完整 SelfWAM |

**关键发现**: 单独引入 [[Clean Action Conditioning]] 因为改变了视频预测的信息结构，短期可能轻微扰动动作策略；加入 [[Robot Self-Mask Prediction]] 后不仅恢复基线，更进一步提升，两个模块相辅相成。

---

### Table 5: 推理效率对比

| 模式 | FastWAM | SelfWAM | 增量 |
|------|---------|---------|------|
| 仅动作（ms） | 320.7 | 323.9 | +0.97% |
| 动作 + 视频（ms） | 668.4 | 677.5 | +1.36% |
| 显存峰值（GB） | 13.959 | 13.959 | 0% |

**关键发现**: 推理时两个训练专属分支（干净动作 + 视频流）均可省略，性能开销可忽略不计。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboTwin 2.0]] | 27,500 demo（2,500 clean + 25,000 random） | 50 项双臂操作，仿真 | 训练 + 评估 |
| AgileX ALOHA（真机） | 17,307 episodes，9.5M transitions | 4 抓放任务，三视角（高位 + 双腕） | 训练 + 评估 |
| Self-Mask 标签 | 与 RGB 数据对齐 | RobotSeg 离线生成机器人分割掩码 | 辅助训练目标 |

### 实现细节

- **Backbone**: [[Wan 2.2-5B]] WAN 2.2 视频扩散 Transformer
- **输入分辨率**: 384×320（三视角 T 形合图）
- **动作 horizon**: 32 步，视频预测 8 帧（时序下采样）
- **优化器**: [[AdamW]]，学习率 $1\times10^{-4}$
- **训练轮数**: 5 epochs
- **硬件**: 64× A800 GPU，约 30 小时
- **RGB : Self-Mask 采样比**: 9:1
- **动作执行**: 每 24 步重规划（action chunk 32 步，执行前 24 步）
- **去噪步数**: 推理时 10 步

### 可视化结果

真实机器人实验中，SelfWAM 在杯子放置和鼠标放垫两项对精细末端位置敏感的任务上达到 100% 成功率，FastWAM 仅 80%，体现了动作感知视频预测对精细操作策略的实质增益。

---

## 批判性思考

### 优点
1. **机制设计简洁优雅**: 干净/带噪动作分流的思路巧妙规避目标泄漏，数值验证充分（图 S3 数量级精度）
2. **推理零开销**: 额外训练信息不带来部署代价，工程友好
3. **自我掩码的动机充分**: 消融实验定量证明掩码监督与动作条件化协同效应，非简单叠加
4. **评估全面**: 覆盖仿真基准、真实机器人、视频预测质量、动作敏感性（方向扰动 + 插值 + WorldArena 指标）

### 局限性
1. **自我掩码依赖离线分割模型**: 需要 RobotSeg 预先生成标签，增加数据准备复杂度；分割质量直接影响掩码监督信号
2. **未建模接触力与遮挡**: 动作后果中隐式物理交互（接触力、被遮挡物体的状态）仍未显式建模
3. **仿真提升有限**: RoboTwin 2.0 上仅提升 0.78%，可能已接近当前 WAM 范式瓶颈
4. **机构/数据未公开**: 论文未披露机构信息，代码/数据集未开源，可复现性存疑

### 潜在改进方向
1. 引入接触力预测或深度图作为额外自接地目标，进一步丰富物理先验
2. 探索自动分割（如 SAM2）替代专用 RobotSeg，降低数据准备门槛
3. 将干净动作条件化机制推广至更长 horizon 或多任务规划场景

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（优化器、轮数、GPU 数量均有说明）
- [x] 数据集可获取（RoboTwin 2.0 公开，真机数据未公开）

---

## 关联笔记

### 基于
- [[Fast-WAM]]: 直接前驱，SelfWAM 在其两流 MoT 上扩展
- [[GeoSem-WAM]]: 类似思路，通过几何与语义辅助分支增强视频预测
- [[World Action Model]]: 范式来源

### 对比
- [[GigaWorldPolicy0.5]]: 仿真基准 Table 1 对比方法，同为近期 WAM
- [[Flash-WAM]]: 高效 WAM 基线

### 方法相关
- [[Clean Action Conditioning]]: 本文核心创新 1
- [[Robot Self-Mask Prediction]]: 本文核心创新 2
- [[Mixture-of-Transformers]]: 架构基础
- [[Wan 2.2-5B]]: 视频骨干

### 硬件/数据相关
- [[ALOHA]]: 真实机器人平台（AgileX ALOHA）
- [[RoboTwin 2.0]]: 仿真评估基准
- [[Action Chunking]]: 动作表示方式

---

## 速查卡片

> [!summary] SelfWAM
> - **核心**: 干净动作条件化 + 机器人自我掩码预测，让 WAM 视频分支真正感知动作
> - **方法**: WAN 2.2-5B 两流 MoT + 不对称注意力（视频 query 见干净动作，动作 query 不见）
> - **结果**: RoboTwin 2.0 平均 92.62%（+0.78% vs FastWAM），真机 95%；推理延迟仅 +0.97%
> - **代码**: 未公开

---

*笔记创建时间: 2026-08-05*
