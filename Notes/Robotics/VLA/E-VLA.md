---
title: "E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes"
method_name: "E-VLA"
authors: [Jiajun Zhai, Hao Shi, Shangwei Guo, Kailun Yang, Kaiwei Wang]
year: 2026
venue: ECCV 2026
tags: [vla, event-camera, robustness, low-light, motion-blur, robot-manipulation, multi-modal-fusion]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2604.04834
created: 2026-07-02
---

# 论文笔记：E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Zhejiang University, Chongqing University |
| 日期 | April 2026（ECCV 2026 接收） |
| 项目主页 | https://github.com/JJayzee/E-VLA |
| 对比基线 | [[SmolVLA]]、[[OpenVLA]]、[[RetinexNet]] |
| 链接 | [arXiv](https://arxiv.org/abs/2604.04834) / [Code](https://github.com/JJayzee/E-VLA) |

---

## 一句话总结

> E-VLA 将[[事件相机]]的运动与结构线索通过轻量级事件适配器融入 VLA 流水线，在极暗光（20 lux）下使 Pick-Place 成功率从 0% 提升至 90%，实现对低光与运动模糊场景的鲁棒操控。

---

## 核心贡献

1. **首个 Event-VLA 融合框架**: 直接利用[[事件相机]]流中的运动与结构线索，无需重建 RGB 图像，保留语义感知能力
2. **双路融合策略**: 参数量为 0 的 Overlay 融合 + 13.3M 参数的[[层次化事件适配器|Hierarchical Event Adapter]]，覆盖不同算力预算
3. **同步 RGB-事件-动作数据集**: 基于 [[DAVIS 事件相机|DAVIS346]] 与 [[LeRobot|SO100]] 机械臂收集 724 条遥操作轨迹，含多光照等级标注

---

## 问题背景

### 要解决的问题

现有 [[VLA|Vision-Language-Action Model]] 依赖高质量 RGB 帧进行感知与决策。在极低光照（< 40 lux）或长曝光运动模糊（> 300ms）场景下，图像信号严重退化，导致策略完全失效（Pick-Place 成功率降至 0%）。

### 现有方法的局限

- **图像增强方法**（[[RetinexNet]]、Retinexformer、EvLight）：需要额外网络（0.56M–22.73M 参数，48–534 GFLOPs）；增强结果在低光极端情况下仍无法为下游策略提供可靠特征
- **事件重建方法**（E2VID）：重建的伪 RGB 图像在极低光照下产生伪影，将信息损失转嫁到另一模态；连续递推重建存在时序偏差，实验结果平均仅 35.8%
- **VLA 本身**: 预训练于大规模文本-图像数据，缺乏对[[事件相机]]异步脉冲数据的原生支持

### 本文的动机

[[事件相机]]在极低光照与高速运动场景下仍能可靠输出亮度变化信号（在 20 lux 时每秒约 87,000 个事件，与 200 lux 时相近），而 RGB 摄像头信号几乎消失。因此，与其重建退化图像，不如**直接将事件流编码为可供预训练 [[ViT]] 编码器处理的稠密灰度帧**，以最小开销增强 VLA 鲁棒性。

---

## 方法详解

### 模型架构

![Figure 1: E-VLA 总览](https://arxiv.org/html/2604.04834v2/x1.png)

**说明**: 左侧展示遥操作数据采集平台（[[DAVIS 事件相机|DAVIS346]] + [[LeRobot|SO100]]），中间对比 RGB-only VLA 在低光/模糊场景的失败与 E-VLA 的成功，右侧展示跨光照等级的成功率对比曲线。

E-VLA 以 [[SmolVLA]]（0.5B 参数）为基础 [[VLA|Vision-Language-Action]] 骨干，扩展支持[[事件相机]]输入：

- **输入**: 语言指令 $l$ + RGB 帧 $I \in \mathbb{R}^{H \times W \times 3}$ + 事件流 $\mathcal{E}$ + 本体状态 $s_t$
- **Backbone**: 冻结的 [[SigLIP]] 视觉编码器 + 冻结 LLM 骨干
- **事件分支**: [[层次化事件适配器|Event Adapter]] / Overlay 融合
- **输出**: [[Action Chunking|动作块]] $a_{t:t+40}$（30 Hz，预测约 1.33 秒）
- **总参数**: 0.5B（+ 13.3M 适配器，约 2.8% 开销）

### 核心模块

#### 模块1: 近计数事件窗口（Recent-Count Windowing）

**设计动机**: 机器人运动速度不稳定，导致事件率波动大，固定时长窗口在低速时事件稀疏、高速时过于密集；近计数窗口自适应地保留最新 N 个事件，分布更稳定。

**具体实现**: 截取最近 $N = 2000$ 个事件，采用极性无关的计数累积（Accumulate Count）生成稠密灰度帧，再通过去马赛克 $\mathcal{D}$ 转为三通道：

#### 模块2: Overlay 融合（参数量为 0）

**设计动机**: 最轻量的事件集成方式，直接在 RGB 图像上覆盖最近事件极性，保持预训练编码器 token 分布。

**具体实现**: 对每个有事件的像素位置，用最新事件的极性颜色覆盖 RGB 值；无事件的像素保持原 RGB 不变。

#### 模块3: 层次化事件适配器（Hierarchical Event Adapter）

**设计动机**: 通过在[[ViT]]编码器的中间层注入事件特征，实现渐进式多层次融合，性能优于 Overlay 但仍保持轻量（13.3M 参数，20.4 GFLOPs）。

**具体实现**:
- 事件帧经独立的 [[ViT]] patch 嵌入（16×16，与图像共享权重）生成事件 token 序列 $E^{(0)}$
- 4 个堆叠[[Transformer]]块处理事件序列，在 SigLIP 编码器的第 3、6、9、12 层注入：
  - 事件特征 $E^{(l)}$ 经 MLP 融合模块与图像特征 $F^{(l)}$ 拼接
  - 融合后的图像特征继续传入下一 SigLIP 层
- 融合后视觉 token 与语言 token、状态 token 拼接后输入冻结 LLM

---

## 关键公式

### 公式1: [[事件相机|事件流表示]]

$$
\mathcal{E}=\{e_{i}\}_{i=1}^{M}, \quad e_{i}=(x_{i},y_{i},t_{i},p_{i})
$$

**含义**: 每个事件是一个四元组，记录触发像素坐标、时间戳（微秒级）和极性（亮度增减）。

**符号说明**:
- $x_i, y_i$: 像素坐标
- $t_i$: 微秒级时间戳
- $p_i \in \{+1, -1\}$: 亮度变化极性

### 公式2: [[近计数事件窗口|Recent-Count 窗口]]

$$
\mathcal{E}_{t_e}=\{e_i \mid t_i \leq t_e\}, \quad \mathcal{W}_I=\{e_k\}_{k=|\mathcal{E}_{t_e}|-N+1}^{|\mathcal{E}_{t_e}|}
$$

**含义**: 截取曝光时刻 $t_e$ 之前最新的 $N$ 个事件构成窗口 $\mathcal{W}_I$。

**符号说明**:
- $t_e$: 当前 RGB 帧的曝光时刻
- $N$: 窗口大小（最优为 2000）

### 公式3: [[极性无关事件累积|事件帧生成]]

$$
\tilde{E}(x,y)=\sum_{e_i \in \mathcal{W}_I}\delta(x-x_i,y-y_i), \quad E=\mathcal{D}(\text{Norm}(\tilde{E}))
$$

**含义**: 对窗口内所有事件按像素坐标做极性无关计数，归一化后去马赛克得到三通道事件帧。

**符号说明**:
- $\delta(\cdot)$: Dirac delta 函数（统计命中像素的事件数）
- $\text{Norm}(\cdot)$: 归一化到 $[0,1]$
- $\mathcal{D}(\cdot)$: 去马赛克操作（灰度→三通道）

### 公式4: [[Overlay 融合|Overlay 事件叠加]]

$$
I^o(x,y)=\begin{cases}I(x,y), & |\mathcal{E}_{(x,y)}|=0 \\ \boldsymbol{c}(p_j), & j=\operatorname*{argmax}_i t_i, e_i \in \mathcal{E}_{(x,y)}\end{cases}
$$

**含义**: 对每个像素，若该位置有事件则用最新事件的极性颜色覆盖 RGB，否则保留原值。

**符号说明**:
- $\mathcal{E}_{(x,y)}$: 位于 $(x,y)$ 的事件集合
- $\boldsymbol{c}(p_j)$: 极性 $p_j$ 对应的颜色编码（正极性/负极性分别映射不同颜色）

### 公式5: [[层次化事件适配器|Hierarchical Event Adapter 融合]]

$$
E^{(l+1)}=\mathcal{G}_{l+1}(E^{(l)}), \quad F^{(l+1)}=\mathcal{F}_{l+1}(\text{Fuse}(F^{(l)},E^{(l)}))
$$

**含义**: 事件特征 $E^{(l)}$ 经[[Transformer]]块 $\mathcal{G}$ 更新，同时通过 $\text{Fuse}$（拼接 + MLP）将事件信息注入图像特征 $F^{(l)}$，再由 SigLIP 层 $\mathcal{F}$ 继续处理。

**符号说明**:
- $E^{(l)}$: 第 $l$ 层的事件 token 序列（维度 384）
- $F^{(l)}$: 第 $l$ 层的图像 token 序列（维度 768）
- $\mathcal{G}_{l+1}$: 第 $l+1$ 个[[Transformer]]事件块
- $\mathcal{F}_{l+1}$: 第 $l+1$ 个 SigLIP 编码器层
- 注入层位置: $l \in \{3, 6, 9, 12\}$

---

## 关键图表

### Figure 2: E-VLA 整体框架

![Figure 2: E-VLA 框架概览](https://arxiv.org/html/2604.04834v2/x2.png)

**说明**: 展示两种融合策略的对比。**Hierarchical Event Adapter** 将事件特征注入冻结 [[SigLIP]] 编码器的中间层；**Overlay** 直接在 RGB 图像上叠加事件帧。融合后的视觉 token 与语言 token、状态 token 拼接，经冻结 LLM 骨干条件化 Action Expert 输出机器人动作。

### Figure 3: E-VLA 系统流程图

![Figure 3: E-VLA 系统流程](https://arxiv.org/html/2604.04834v2/x3.png)

**说明**: 展示从[[事件相机]]原始数据到机器人动作的完整推理流水线。事件流经缓冲、聚合为[[近计数事件窗口|Recent-Count 表示]]后，与 RGB 特征在[[层次化事件适配器]]中对齐融合，最终经 LLM 与 Action Expert 生成分块动作，通过动作缓冲区流式执行。

### Figure 4: 数据集与平台展示

![Figure 4: 数据集与遥操作平台](https://arxiv.org/html/2604.04834v2/x4.png)

**说明**: 左侧展示基于 [[LeRobot|SO100]] 机械臂与 [[DAVIS 事件相机|DAVIS346]] 的遥操作平台（侧视图与俯视图）；右侧展示数据集统计与事件率随光照变化曲线——即使 RGB 图像信号随光照急剧下降，事件率仍维持约 87K 事件/秒，证明[[事件相机]]的鲁棒性。

### Figure 5: 不同光照下的视觉输入对比

![Figure 5: 不同光照下视觉输入定性对比](https://arxiv.org/html/2604.04834v2/x5.png)

**说明**: 定性展示在不同光照等级下 RGB 图像（退化严重）、Overlay 融合（部分保留结构）与 Event Adapter 输出（结构清晰）的对比，直观说明[[事件相机]]在低光场景的优势。

### Figure S6: 不同光照下的灰度分布（附录）

![Figure S6: 灰度分布CDF](https://arxiv.org/html/2604.04834v2/x6.png)

**说明**: 对数横轴的灰度累积分布函数（CDF），展示从 200 lux 到 20 lux 图像信号的急剧压缩，量化说明 RGB 摄像头在极低光照下的信息退化程度。

### Figure S7: E2VID 重建结果（附录）

![Figure S7: E2VID重建质量](https://arxiv.org/html/2604.04834v2/x7.png)

**说明**: 展示不同 E2VID 实现在 40 lux 下的连续帧重建结果，揭示重建方法存在时序伪影，说明直接利用事件流优于事件重建。

### Figure S8: 图像 Dropout 率消融（附录）

![Figure S8: Dropout率消融](https://arxiv.org/html/2604.04834v2/x8.png)

**说明**: 对 20%~80% 图像 Dropout 率的消融实验，最终选用 50% dropout 作为防止模型走捷径（依赖 RGB）的最佳策略。

### Figure S9: 运动模糊场景示例（附录）

![Figure S9: 运动模糊场景](https://arxiv.org/html/2604.04834v2/x9.png)

**说明**: (a) Pick-Place 正常场景序列；(b) 300ms 曝光运动模糊；(c) 1000ms 曝光严重运动模糊，直观展示长曝光带来的视觉退化。

---

### Table 1: 低光照 Pick-Place 成功率（%）

| 方法 | 75 lux | 40 lux | 35 lux | 30 lux | 25 lux | 20 lux | 平均 |
|------|--------|--------|--------|--------|--------|--------|------|
| Image（10ms exp.） | 100 | 80 | 70 | 35 | 0 | 0 | 47.5 |
| Image + RetinexNet | 100 | 100 | 85 | 80 | 25 | 10 | 66.7 |
| Image + Retinexformer | 100 | 80 | 80 | 75 | 20 | 10 | 60.8 |
| Image + EvLight | 100 | 95 | 95 | 75 | 45 | 10 | 70.0 |
| Image + E2VID | 80 | 60 | 55 | 10 | 5 | 5 | 35.8 |
| Ours Overlay | 100 | 100 | 85 | 75 | 65 | 60 | 80.8 |
| **Ours Event Adapter** | **100** | **100** | **95** | **90** | **90** | **90** | **94.2** |

**关键发现**: Event Adapter 在 20 lux 时达到 90% 成功率，而 RGB-only 基线完全失败（0%）；平均成功率 94.2% vs 47.5%，提升近 2 倍。

### Table 2: 分布外泛化（OOD，仅在 200 lux 正常光照训练，测试低光照）

| 方法 | 条件 | 100 lux | 75 lux | 40 lux | 30 lux | 20 lux | 平均 |
|------|------|---------|--------|--------|--------|--------|------|
| Image 基线 | ID | 100 | 100 | 80 | 35 | 0 | — |
| | OOD | 100 | 65 | 30 | 0 | 0 | 39.0 |
| Image + Retinexformer | OOD | 100 | 70 | 50 | 10 | 0 | 46.0 |
| Image + EvLight | OOD | 80 | 50 | 50 | 20 | 5 | 40.5 |
| Ours Overlay | OOD | 100 | 95 | 55 | 10 | 10 | 54.0 |
| **Ours Event Adapter** | **OOD** | **100** | **85** | **75** | **70** | **45** | **75.0** |

**关键发现**: 仅在正常光照下训练，Event Adapter OOD 平均成功率 75%，比 RGB-only 基线（39%）高出 36 个百分点，展示出色的分布外泛化能力。

### Table 3: 运动模糊成功率（%，Pick-Place / Sorting）

| 方法 | 100ms | 200ms | 300ms | 400ms | 500ms | 1000ms | 平均 |
|------|-------|-------|-------|-------|-------|--------|------|
| Image P&P | 100 | 85 | 40 | 30 | 15 | 0 | 45.0 |
| Image Sorting | 77.5 | 75 | 52.5 | 32.5 | 20 | 5 | 43.75 |
| Overlay P&P | 100 | 100 | 90 | 60 | 60 | 20 | 71.7 |
| Overlay Sorting | 95 | 95 | 72.5 | 70 | 55 | 32.5 | 70.0 |
| **Adapter P&P** | **100** | **100** | **85** | **85** | **50** | **25** | **74.2** |
| Adapter Sorting | 95 | 85 | 85 | 72.5 | 45 | 32.5 | 69.2 |

**关键发现**: 在极端 1000ms 曝光下，Event Adapter Pick-Place 成功率 25% vs 基线 0%；中等模糊（300-500ms）提升尤为显著。

### Table 4: 事件窗口策略消融（Pick-Place）

| 窗口类型 | 75 lux | 40 lux | 35 lux | 30 lux | 25 lux | 20 lux | 平均 |
|---------|--------|--------|--------|--------|--------|--------|------|
| 5 ms 固定时长 | 80 | 70 | 65 | 55 | 50 | 45 | 60.8 |
| 20 ms 固定时长 | 100 | 85 | 85 | 80 | 75 | 70 | 82.5 |
| 40 ms 固定时长 | 100 | 90 | 90 | 75 | 50 | 45 | 75.0 |
| 500 事件近计数 | 100 | 100 | 60 | 50 | 50 | 45 | 67.5 |
| **2000 事件近计数** | **100** | **100** | **95** | **90** | **90** | **90** | **94.2** |
| 4000 事件近计数 | 100 | 100 | 85 | 85 | 75 | 70 | 85.8 |

**关键发现**: N=2000 的[[近计数事件窗口]]最优；近计数窗口整体优于固定时长窗口，因机器人运动速度不均匀导致固定时长下事件密度不稳定。

### Table 5: 训练策略消融（30 lux Pick-Place 成功率）

| 训练策略 | 动作阶段 | 事件阶段 | 联合阶段 | 权重共享 | 成功率 |
|---------|---------|---------|---------|---------|--------|
| Action→Joint | ✓ | ✗ | ✓ | ✓ | 75.0 |
| Event→Joint | ✗ | ✓ | ✓ | ✓ | 50.0 |
| Action→Event→Joint（无共享） | ✓ | ✓ | ✓ | ✗ | 80.0 |
| **完整方案（Ours）** | **✓** | **✓** | **✓** | **✓** | **90.0** |

**关键发现**: 三阶段渐进训练（Action→Event→Joint）+权重共享缺一不可；跳过事件专门训练阶段导致感知-动作对齐失败（75%）。

### Table S9: 计算效率对比

| 方法 | 参数量（M） | FLOPs（G） |
|------|------------|-----------|
| RetinexNet | 0.56 | 48.5 |
| Retinexformer | 1.61 | 23.5 |
| EvLight | 22.73 | 533.7 |
| E2VID | 10.71 | 171.5 |
| Overlay（Ours） | 0.00 | ~0 |
| **Event Adapter（Ours）** | **13.31** | **20.4** |

**关键发现**: Event Adapter 在所有基于事件的方法中 FLOPs 最低（20.4G），而性能最强；Overlay 完全无参数，适合资源极度受限场景。

### Table S11: 事件表示方法对比（Pick-Place 平均成功率）

| 表示方法 | 75 lux | 40 lux | 35 lux | 30 lux | 25 lux | 20 lux | 平均 |
|---------|--------|--------|--------|--------|--------|--------|------|
| Voxel Grid | 90 | 80 | 85 | 80 | 60 | 35 | 71.7 |
| Time Surface | 95 | 85 | 80 | 80 | 75 | 80 | 82.5 |
| Accum. Sum（极性感知） | 90 | 85 | 80 | 80 | 75 | 65 | 79.2 |
| **Accum. Count（极性无关）** | **100** | **100** | **95** | **90** | **90** | **90** | **94.2** |

**关键发现**: 极性无关的计数累积最优，因其产生的稠密灰度分布与 SigLIP 预训练图像分布最接近，利用了预训练视觉知识。

### Table S12: Sorting 与 Stacking 任务低光照成功率（%）

| 方法 | 任务 | 75 lux | 40 lux | 35 lux | 30 lux | 25 lux | 20 lux |
|------|------|--------|--------|--------|--------|--------|--------|
| Image | Sorting | 100.0 | 85.0 | 62.5 | 30.0 | 0.0 | 0.0 |
| | Stacking | 60 | 45 | 35 | 15 | 0 | 0 |
| Overlay | Sorting | 95 | 80 | 72.5 | 70 | 67.5 | 50 |
| | Stacking | 55 | 60 | 45 | 40 | 35 | 30 |
| **Adapter** | **Sorting** | **97.5** | **85** | **82.5** | **82.5** | **80.0** | **70.0** |
| | Stacking | 60 | 55 | 50 | 55 | 45 | 40 |

**关键发现**: Sorting 任务在低光下颜色辨识困难（事件只编码亮度变化不含颜色），Adapter 仍维持 70% vs 基线 0%；Stacking 因遮挡问题提升有限。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| E-VLA Dataset（自建） | 724 条轨迹，~339k 帧 | RGB + 事件 + 动作同步；3 个任务；多光照等级（200/100/75/40 lux） | 训练与测试 |
| Pick-Place | 310 条轨迹 | 三色物体（红/黄/绿），随机配置 | 主要评测任务 |
| Sorting | 244 条轨迹 | 颜色分类（低光下颜色辨识挑战） | 颜色感知测试 |
| Stacking | 170 条轨迹 | 积木堆叠（遮挡挑战） | 遮挡鲁棒性测试 |

### 实现细节

- **基础 VLA**: [[SmolVLA]]（0.5B 参数），[[SigLIP]] 视觉编码器 + 冻结 LLM
- **事件编码器**: 独立 ViT，patch size 16×16，维度 384
- **融合注入层**: SigLIP 第 3、6、9、12 层
- **三阶段训练**:
  - Stage 1（动作对齐）: lr 2×10⁻⁴, batch 256, 20k iter
  - Stage 2（事件预热）: lr 5×10⁻⁴, batch 128, 10k iter
  - Stage 3（联合精调）: lr 1×10⁻⁴, batch 128, 10k iter
- **图像 Dropout**: 50%（联合训练阶段，防止模型忽略事件模态）
- **[[Action Chunking]]**: 40 步，30 Hz（~1.33 秒）
- **训练硬件**: NVIDIA A800 GPU
- **推理硬件**: NVIDIA AGX Orin（边缘部署）
- **机器人**: [[LeRobot|SO100]] 6-DoF，[[DAVIS 事件相机|DAVIS346]] 腕部单目摄像头

### 可视化结果

低光照下 E-VLA Event Adapter 完整执行 Pick-Place 的可视化（Figure 5）：RGB 帧几乎全黑无法辨识物体，而事件帧仍清晰显示物体轮廓与运动边缘，结合[[层次化事件适配器]]成功引导抓取与放置动作。

---

## 批判性思考

### 优点

1. **任务导向设计**: 直接利用事件流特征而非重建 RGB，避免信息二次损失，策略更简洁
2. **计算效率突出**: Event Adapter 仅 13.3M 参数、20.4 GFLOPs，远低于 EvLight（22.73M，533.7G），可在边缘设备（AGX Orin）实时运行
3. **OOD 泛化强**: 仅在正常光照训练即可泛化到极低光照（75%），具备实用价值
4. **系统开放完整**: 开源代码、数据集与遥操作平台，可复现性高

### 局限性

1. **颜色语义缺失**: [[事件相机]]仅编码亮度变化，抓取后物体被遮挡时颜色信息消失，Sorting 颜色分类任务受限
2. **单目遮挡问题**: 腕部摄像头在堆叠操作中被抓持物体遮挡，Stacking 性能提升有限
3. **静态场景盲区**: 场景静止时事件率为零，模型退化为纯 RGB 策略（但此时 RGB 本身通常可用）
4. **数据规模限制**: 仅 724 条轨迹，复杂场景泛化能力待验证

### 潜在改进方向

1. **主动事件感知**: 通过预测性运动触发事件，而非被动等待机器人运动
2. **多模态颜色融合**: 在低光下用近红外或事件极性颜色编码补充颜色语义
3. **双目事件相机**: 解决单目遮挡问题，同时引入深度信息
4. **扩展到更多 VLA**: 验证 Event Adapter 在 [[π0.5]]、[[OpenVLA]] 等更大模型上的迁移性

### 可复现性评估

- [x] 代码开源（https://github.com/JJayzee/E-VLA）
- [x] 数据集开源（RGB-Event-Action Dataset）
- [x] 训练细节完整（三阶段参数全公开）
- [x] 评测基准明确（实物机器人，固定测试配置）

---

## 关联笔记

### 基于

- [[SmolVLA]]: E-VLA 的基础 VLA 骨干（0.5B 参数）
- [[SigLIP]]: 冻结视觉编码器，Event Adapter 注入点
- [[π0.5]]: 附录中验证事件适配器在更强骨干上的迁移性（Table S8，95.0% 平均成功率）

### 对比

- [[EventVLA]]: 同期事件-VLA 融合工作，动作条件化事件融合
- [[RetinexNet]]: 低光增强基线，计算更轻但策略性能弱（66.7%）

### 方法相关

- [[事件相机]]: 核心传感器，异步脉冲视觉感知
- [[DAVIS 事件相机]]: 具体硬件 DAVIS346，同步 RGB+事件
- [[层次化事件适配器]]: 本文提出的核心模块（待创建概念）
- [[近计数事件窗口]]: 本文提出的事件窗口策略（待创建概念）
- [[Action Chunking]]: 动作块预测机制

### 硬件/数据相关

- [[LeRobot]]: SO100 机械臂平台
- [[Jetson Orin]]: 边缘推理硬件（AGX Orin 变体）

---

## 速查卡片

> [!summary] E-VLA (ECCV 2026)
> - **核心**: 首个将[[事件相机]]直接集成到 VLA 流水线的框架，无需重建 RGB
> - **方法**: 近计数事件帧 + Overlay 融合 / 层次化事件适配器（13.3M 参数）
> - **结果**: Pick-Place 在 20 lux 从 0% 提升至 90%；OOD 泛化平均 75% vs 基线 39%
> - **代码**: https://github.com/JJayzee/E-VLA

---

*笔记创建时间: 2026-07-02*
