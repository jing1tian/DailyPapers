---
title: "GeoMem-VLA: Internalizing 3D Scene Understanding into Temporal Memory for Vision-Language-Action Models"
method_name: "GeoMem-VLA"
authors: [Qinfan Zhang, Yifan Luo, Haoyuan Shi, Guanlin Liu, Han Dong, Haozhe Shan, Yong Dai, Xiancong Ren, Xiaozhu Ju]
year: 2026
venue: arXiv
tags: [vla, 3d-scene-understanding, temporal-memory, robot-manipulation, flow-matching, geometry-supervision, bimanual]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2610.29866v1
created: 2026-10-23
---

# 论文笔记：GeoMem-VLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | X-Humanoid、北航、中国人民大学、中科大、港中深、复旦大学 |
| 日期 | October 2026 |
| 项目主页 | [geomem-vla.github.io](https://geomem-vla.github.io/) |
| 对比基线 | [[Spatial Forcing]]、[[MemoryVLA]]、[[StarVLA]]、[[3DThinkVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.29866) / [Code](https://github.com/GeoMem-VLA/GeoMem-VLA) / [HuggingFace](https://huggingface.co/rxc205/GEOMEM-VLA) |

---

## 一句话总结

> GeoMem-VLA 在 [[VLA]] 内部引入可学习的 [[GeoPlan Token]]，由冻结的几何基础模型 [[VGGT]] 监督，将 3D 场景理解以紧凑的几何感知 [[时序记忆|Geometry-Aware Episode Memory]] 的形式积累，推理时无需 3D 教师，在 LIBERO 达到 98.9%、RoboTwin 达到 91.17% SOTA。

---

## 核心贡献

1. **几何监督空间表示 (Geometry-Supervised Latents)**: 引入 $K$ 个可学习 [[GeoPlan Token]]，经 VLM 编码后投影到几何工作空间，由 [[VGGT]] 提供点图（point map）和末端执行器（EEF）轨迹作为 [[Flow Matching]] 监督目标，训练结束后移除教师。
2. **因果几何记忆 (Geometry-Aware Episode Memory)**: 三路独立记忆流（视觉×2 + 几何×1）通过 [[Causal Retrieval Memory|因果检索]] 融合历史观测，配合 [[Memory Consolidation|记忆合并]] 机制防止冗余积累，每集重置。
3. **跨平台真实机器人验证**: 在 Astribot S1 和 AgileX Cobot Magic 2 五项真实任务中平均 78.0%，超越 LingBot-VLA 2.0、π0.5 和 OpenVLA-OFT。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型主要依赖当前帧的 2D 观测进行决策，缺乏对三维空间的显式推理能力，且无法利用跨时间步的历史观测，在遮挡、长时程操作等场景中表现受限。

### 现有方法的局限

- **逐帧 3D 方法**（如 Spatial Forcing、[[3DThinkVLA]]）：每帧独立推断 3D，不保留历史；推理需要 3D 预测器，增加延迟。
- **纯记忆方法**（如 [[MemoryVLA]]、EventVLA）：记忆中缺乏几何结构信息，无法支持精确的空间推理。
- **3D 重建方法**（NeRF、3DGS 集成）：重建代价高，难以实时更新。

### 本文的动机

机器人操作需要的是**任务相关、持续更新的几何摘要**，而非每帧完整的 3D 重建。通过在 VLM 的 latent space 内用 [[GeoPlan Token]] 学习紧凑几何表示，配合因果记忆跨时间步积累，可以在**推理时完全移除 3D 教师**。

---

## 方法详解

### 模型架构

GeoMem-VLA 基于 **StarVLA/[[GR00T]]** 框架扩展，采用 [[Diffusion Transformer (DiT)|Flow-Matching Diffusion Transformer]] 架构：

- **输入**: 多视角 RGB $\mathbf{o}_t$、本体感知状态 $\mathbf{s}_t$、语言指令 $\ell$
- **VLM Backbone**: [[Qwen3-VL]]-4B（冻结）
- **几何分支**: 由 [[VGGT]] 教师网络（训练时）监督 [[GeoPlan Token]] 学习几何表示
- **记忆模块**: 三路独立记忆流（视觉流 $j=1,2$、几何流 $j=3$），通过 [[Causal Retrieval Memory|因果跨注意力检索]]
- **动作专家**: GR00T-style [[Flow Matching]] Diffusion Transformer
- **输出**: [[Action Chunking|动作块]] $\widehat{\mathbf{A}}_t \in \mathbb{R}^{H \times d_a}$

### 核心模块

#### 模块1: GeoPlan 几何监督潜变量

**设计动机**: 在 VLM latent 空间内直接学习 3D 几何摘要，避免推理时调用外部 3D 预测器

**具体实现**:
- 在 VLM 输入序列末尾追加 $K$ 个可学习 [[GeoPlan Token]]
- 取 VLM 最后一层在这 $K$ 个标记位置的状态 $\mathbf{z}_t$
- 经线性投影得到几何工作空间表示 $\mathbf{c}_t = \mathrm{Proj}(\mathbf{z}_t) \in \mathbb{R}^{K \times D_g}$
- 以冻结的 [[VGGT]] 输出（当前帧点图 + 未来 EEF 轨迹）为目标，用辅助 [[Flow Matching]] 头监督 $\mathbf{c}_t$
- **推理时**：移除 [[VGGT]] 教师和几何解码头，仅保留 [[GeoPlan Token]] 的 VLM 通路

#### 模块2: 几何感知时序记忆

**设计动机**: 跨时间步积累场景状态和几何信息，支持遮挡恢复和长时程规划

**具体实现**:
- 三路独立记忆库 $\mathcal{M}^j$（$j=1,2$：视觉观测流；$j=3$：几何 token 流）
- [[Causal Retrieval Memory|因果检索]]：每路通过带时间戳编码的跨注意力，仅检索**同一集内当前时刻之前**的条目
- **检索结果融合**：几何检索结果 $\widetilde{\mathbf{c}}_t$ 与视觉 token $\widetilde{\mathbf{p}}_t$ 通过门控交叉注意力融合（见附录 B，公式9）
- [[Memory Consolidation|记忆合并]]：当记忆库满时，找余弦相似度最高的相邻条目对，取均值合并，保留差异性状态

#### 模块3: 事件感知训练

**设计动机**: 训练时模拟跨时间步的记忆检索场景，增强历史利用能力

**具体实现**:
- 从同一集内采样时间上分散的帧，保持时间顺序和因果访问
- 每个训练样本包含当前帧和历史帧的记忆上下文

---

## 关键公式

### 公式1: [[VLA|VLA 策略]] 输出

$$
\widehat{\mathbf{A}}_t = \pi_\theta(\mathbf{o}_t, \mathbf{s}_t, \ell)
$$

**含义**: 策略网络由多视角观测 $\mathbf{o}_t$、本体状态 $\mathbf{s}_t$ 和语言指令 $\ell$ 预测动作块

**符号说明**:
- $\widehat{\mathbf{A}}_t \in \mathbb{R}^{H \times d_a}$: 预测动作块，$H$ 为动作 horizon，$d_a$ 为动作维度
- $\mathbf{o}_t$: 多视角 RGB 观测
- $\mathbf{s}_t$: 本体感知状态
- $\ell$: 语言指令

### 公式2: [[Flow Matching|动作流匹配损失]]

$$
\mathcal{L}_{\text{action}} = \mathbb{E}_{\rho, \bm{\epsilon}^a}\left[\left\| v_\psi^{\text{act}}(\mathbf{A}_{t,\rho}, \rho, \mathbf{C}_t) - (\mathbf{A}_t - \bm{\epsilon}^a) \right\|^2\right]
$$

**含义**: 训练动作速度场 $v_\psi^{\text{act}}$，在噪声轨迹 $\mathbf{A}_{t,\rho}$ 和条件 $\mathbf{C}_t$ 下预测真实流方向

**符号说明**:
- $\mathbf{A}_{t,\rho} = (1-\rho)\bm{\epsilon}^a + \rho\mathbf{A}_t$: 流插值点
- $\rho$: 从 Beta 衍生调度采样的流时间
- $\bm{\epsilon}^a \sim \mathcal{N}(0, I)$: 高斯噪声
- $\mathbf{C}_t$: 视觉-语言上下文

### 公式3: [[GeoPlan Token|GeoPlan 投影]]

$$
\mathbf{c}_t = \mathrm{Proj}(\mathbf{z}_t) \in \mathbb{R}^{K \times D_g}
$$

**含义**: 将 VLM 最后一层在 $K$ 个 GeoPlan 标记处的状态 $\mathbf{z}_t$ 投影到几何工作空间

**符号说明**:
- $\mathbf{z}_t$: VLM 最后一层 GeoPlan 标记的输出状态
- $K$: GeoPlan token 数量（超参数）
- $D_g$: 几何工作空间维度

### 公式4: [[GeoPlan Token|几何监督目标拼接]]

$$
\mathbf{Y}_t = [\mathbf{P}_t;\, \mathbf{W}_t]
$$

**含义**: 将当前点图和未来 EEF 轨迹拼接为几何监督目标

**符号说明**:
- $\mathbf{P}_t \in \mathbb{R}^{P^2 \times 3}$: [[VGGT]] 提供的点图（置信度加权池化到 $P \times P$ 网格）
- $\mathbf{W}_t \in \mathbb{R}^{n_e \times 3}$: 未来 $n_e$ 步 EEF 航点（转换到当前点图坐标系）

### 公式5: [[Flow Matching|几何流匹配损失]]

$$
\mathcal{L}_{\text{geo}} = \sum_{m \in \{\text{scene},\, \text{eef}\}} \lambda_m\, \mathbb{E}_{\tau, \epsilon}\left[\left\| v_\phi(\mathbf{Y}_{t,\tau}, \tau, \mathbf{c}_t) - (\mathbf{Y}_t - \epsilon) \right\|^2\right]
$$

**含义**: 辅助几何流匹配损失，分别对场景点图和 EEF 轨迹部分加权监督

**符号说明**:
- $\lambda_m$: 各部分权重（$\lambda_\text{scene}=0.5$，$\lambda_\text{eef}=3.0$）
- $\tau \sim \mathcal{U}[0,1]$: 流时间
- $\mathbf{Y}_{t,\tau} = (1-\tau)\epsilon + \tau\mathbf{Y}_t$: 流插值点
- $v_\phi$: 几何速度场（训练完成后移除）

### 公式6: [[GeoPlan Token|总训练损失]]

$$
\mathcal{L} = \mathcal{L}_{\text{action}} + \alpha\,\mathcal{L}_{\text{geo}}
$$

**含义**: 动作预测损失与几何监督辅助损失的加权和

**符号说明**:
- $\alpha$: 几何监督强度系数
- $\mathcal{L}_{\text{action}}$: 动作流匹配损失
- $\mathcal{L}_{\text{geo}}$: 几何辅助流匹配损失

### 公式7: [[Causal Retrieval Memory|因果记忆检索]]

$$
\widetilde{\mathbf{x}}^j_t = R_j\!\left(\mathbf{x}^j_t,\, \mathcal{M}^j_{<t};\, t\right)
$$

**含义**: 第 $j$ 路检索函数仅从当前集内时刻 $t$ 之前的记忆库中检索，保证因果性

**符号说明**:
- $\mathbf{x}^j_t$: 当前时刻第 $j$ 路的查询特征
- $\mathcal{M}^j_{<t}$: 第 $j$ 路在时刻 $t$ 之前的记忆条目集合
- $t$: 时间戳编码，注入检索注意力

### 公式8: [[Memory Consolidation|记忆合并策略]]

$$
r_j^\star = \arg\max_r\, \cos\!\left(\mathrm{vec}(\mathbf{m}^j_r),\, \mathrm{vec}(\mathbf{m}^j_{r+1})\right)
$$

$$
\mathbf{m}^j_{r^\star} \leftarrow \frac{\mathbf{m}^j_{r^\star} + \mathbf{m}^j_{r^\star+1}}{2}
$$

**含义**: 当记忆库溢出时，找余弦相似度最高的相邻条目对并取均值合并，保留状态差异性

**符号说明**:
- $\mathbf{m}^j_r$: 第 $j$ 路第 $r$ 个记忆条目
- $\cos(\cdot,\cdot)$: 向量化后的余弦相似度

### 公式9: 几何-视觉门控融合

$$
\mathbf{u}_t = \mathrm{CrossAttn}(\widetilde{\mathbf{p}}_t,\, \widetilde{\mathbf{c}}_t,\, \widetilde{\mathbf{c}}_t)
$$

$$
\mathbf{h}_t = \mathbf{u}_t + \mathrm{FFN}(\mathrm{LN}(\mathbf{u}_t))
$$

$$
\mathbf{g}_t = \sigma\!\left(W_g[\mathbf{h}_t;\, \widetilde{\mathbf{p}}_t] + \mathbf{b}_g\right)
$$

$$
\mathbf{p}'_t = \mathbf{g}_t \odot \widetilde{\mathbf{p}}_t + (1-\mathbf{g}_t) \odot \mathbf{h}_t
$$

**含义**: 用几何检索结果 $\widetilde{\mathbf{c}}_t$ 作为 key/value 对视觉 token $\widetilde{\mathbf{p}}_t$ 做跨注意力，再通过门控决定几何信息注入比例

**符号说明**:
- $\widetilde{\mathbf{p}}_t$: 视觉流检索结果
- $\widetilde{\mathbf{c}}_t$: 几何流检索结果
- $\mathbf{g}_t \in [0,1]^d$: 软门控系数
- $\sigma$: Sigmoid 激活

### 公式10: 几何重建诊断 RMSE

$$
e_{3D} = \sqrt{\frac{1}{N}\sum_{i=1}^N \left\|\widehat{\mathbf{p}}_i - \mathbf{p}_i\right\|_2^2}
$$

**含义**: 离线评估预测点图与 VGGT 真值点图之间的均方根误差

**符号说明**:
- $N$: 对应点数（场景 256，EEF 100）
- $\widehat{\mathbf{p}}_i$: 第 $i$ 个预测点
- $\mathbf{p}_i$: VGGT 提供的真值点，EEF RMSE 以 mm 报告

---

## 关键图表

### Figure 1: GeoMem-VLA 总览

![Figure 1](https://arxiv.org/html/2610.29866v1/Teaser_Geo.png)

**说明**: 展示遮挡场景示例、[[GeoPlan Token]] 记忆流水线结构，以及在 LIBERO 和 RoboTwin benchmark 上的性能对比。左侧说明了为何 2D 观测不足以应对遮挡，右侧展示 [[时序记忆|Geometry-Aware Episode Memory]] 如何积累历史信息。

### Figure 2: GeoMem-VLA 架构图

![Figure 2](https://arxiv.org/html/2610.29866v1/GeoMemVLA_v4.png)

**说明**: 完整模型架构。VLM 编码图像、语言和 [[GeoPlan Token]]；[[VGGT]] 教师（仅训练时）提供点图和 EEF 轨迹监督；三路记忆流融合历史；动作专家（[[Flow Matching]] DiT）预测动作块。推理时移除蓝色虚框中的教师和几何解码头。

### Figure 3: 几何重建与记忆可视化（RoboTwin）

![Figure 3](https://arxiv.org/html/2610.29866v1/case_figure.png)

**说明**: 在 RoboTwin "open microwave" 和 "stack three blocks" 任务上的离线诊断。展示 RGB 输入、[[VGGT]] 目标点图、GeoMem-VLA 预测点图、未来 EEF 轨迹（演示 vs. 预测），以及最终记忆保留情况。说明模型确实学到了空间几何信息。

### Figure 4: 真实机器人任务成功率

![Figure 4](https://arxiv.org/html/2610.29866v1/task.png)

**说明**: 五项真实任务（每项 10 次 rollout）成功率对比。GeoMem-VLA 在 "Separate file pockets" 达到 100%，平均 78.0%，优于 LingBot-VLA 2.0、π0.5 和 OpenVLA-OFT。

### Figure 6: RoboTwin 几何重建与记忆分析

![Figure 6](https://arxiv.org/html/2610.29866v1/figures/robotwin_cases_overview.png)

**说明**: 更详细的几何重建分析，涵盖教师几何、重建点图、预测轨迹、检索注意力热图，以及保留事件对齐情况，验证记忆模块的有效性。

### Figure 7 / 8: Astribot S1 成功 / 失败案例

| 成功案例 | 失败案例 |
|----------|----------|
| ![Figure 7](https://arxiv.org/html/2610.29866v1/figures/real_world_cases/astribot_success.png) | ![Figure 8](https://arxiv.org/html/2610.29866v1/figures/real_world_cases/astribot_failure.png) |

**说明**: Astribot S1 机器人在拾取竖放棍子、分离文件夹和转移杯子任务中的成功/失败 rollout 对比。失败主要源于末端执行器接触精度不足。

### Figure 9 / 10: AgileX Cobot Magic 2 成功 / 失败案例

| 成功案例 | 失败案例 |
|----------|----------|
| ![Figure 9](https://arxiv.org/html/2610.29866v1/figures/real_world_cases/agilex_success.png) | ![Figure 10](https://arxiv.org/html/2610.29866v1/figures/real_world_cases/agilex_failure.png) |

**说明**: AgileX 双臂机器人在盖布后寻找苹果（遮挡任务）和重复拾取方块任务的成功/失败对比。遮挡任务依赖时序记忆积累布料位移的几何信息。

### Figure 11 / 12: 真实机器人几何记忆分析

| Astribot | AgileX |
|----------|--------|
| ![Figure 11](https://arxiv.org/html/2610.29866v1/figures/real_world_cases/astribot_overview.png) | ![Figure 12](https://arxiv.org/html/2610.29866v1/figures/real_world_cases/agilex_overview.png) |

**说明**: 三项 Astribot / 两项 AgileX 真实任务的几何重建和记忆保留可视化，说明模型在真实机器人上同样获得了有意义的 3D 几何表示。

---

### Table 1: LIBERO Benchmark 成功率 (%)

| 方法 | Spatial | Object | Goal | Long | Average |
|------|---------|--------|------|------|---------|
| OpenVLA (FT) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| TraceVLA | 84.6 | 85.2 | 75.1 | 54.1 | 74.8 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| 4D-VLA | 88.9 | 95.2 | 90.9 | 79.1 | 88.6 |
| DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GeoPredict | 98.0 | 98.2 | 95.7 | 94.0 | 96.5 |
| [[MemoryVLA]] | 98.4 | 98.4 | 96.4 | 93.4 | 96.7 |
| CronusVLA | 97.3 | 99.6 | 96.9 | 94.0 | 97.0 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| GeoVLA | 98.4 | 99.0 | 96.6 | 96.6 | 97.7 |
| AVA-VLA | 99.2 | 99.6 | 97.9 | 96.2 | 98.2 |
| [[Spatial Forcing]] | 99.4 | 99.6 | 98.8 | 96.0 | 98.5 |
| [[3DThinkVLA]] | **100.0** | **100.0** | 98.8 | 95.8 | 98.7 |
| **GeoMem-VLA** | 99.6 | 99.8 | **99.6** | **96.6** | **98.9** |

**关键发现**: GeoMem-VLA 平均 98.9% 为当前最高（超 [[Spatial Forcing]] 0.4pp），Goal 套件最高 99.6%，Long 套件 96.6% 并列最高。

---

### Table 2: RoboTwin 2.0 Benchmark 成功率 (%)

| 方法 | Clean | Randomized | Average |
|------|-------|-----------|---------|
| X-VLA | 70.00 | 39.00 | 54.50 |
| Track4Action | 80.44 | 81.48 | 80.96 |
| ABot-M0 | 86.06 | 85.08 | 85.57 |
| Qwen-VLA-Instruct | 86.10 | 87.20 | 86.65 |
| LingBot-VLA | 88.56 | 86.68 | 87.62 |
| StarVLA-OFT | 88.18 | 88.32 | 88.25 |
| [[StarVLA]]-GR00T | 88.00 | 88.50 | 88.25 |
| JoyAI-RA 0.1 | 90.48 | 89.28 | 89.88 |
| HyVLA-0.5 | 90.90 | 90.10 | 90.50 |
| **GeoMem-VLA** | **91.10** | **91.24** | **91.17** |

**关键发现**: GeoMem-VLA 在 50 项双臂任务 Clean 和 Randomized 两个设置下均达到最高，平均 91.17%（较 HyVLA-0.5 提升 0.67pp）。

---

### Table 3: LIBERO 消融实验 (%)

| 配置 | Geo | Mem | Spatial | Object | Goal | Long | Average |
|------|-----|-----|---------|--------|------|------|---------|
| Full | ✓ | ✓ | 99.6 | 99.8 | 99.6 | 96.6 | 98.9 |
| w/o 几何分支 | ✗ | ✓ | 98.0 | 99.6 | 98.8 | 95.2 | 97.9 |
| w/o 时序记忆 | ✓ | ✗ | 99.4 | 99.6 | 99.4 | 94.2 | 98.2 |
| [[StarVLA]]-GR00T | ✗ | ✗ | 97.8 | 98.8 | 97.4 | 92.0 | 96.5 |

**关键发现**: 移除几何分支损失 1.0pp（Long 降 1.4pp），移除记忆损失 0.7pp（Long 降 2.4pp）；两者对 Long 长时程任务的贡献最显著。

---

### Table 4: RoboTwin 2.0 消融实验 (%)

| 配置 | Geo | Mem | Clean | Randomized | Average |
|------|-----|-----|-------|-----------|---------|
| Full | ✓ | ✓ | 91.10 | 91.24 | 91.17 |
| w/o 几何分支 | ✗ | ✓ | 89.68 | 89.74 | 89.71 |
| w/o 时序记忆 | ✓ | ✗ | 89.84 | 90.52 | 90.18 |
| [[StarVLA]]-GR00T | ✗ | ✗ | 88.00 | 88.50 | 88.25 |
| StarVLA-OFT | ✗ | ✗ | 88.18 | 88.32 | 88.25 |

**关键发现**: 几何分支带来约 +1.5pp，记忆模块带来约 +1.0pp；两者组合有轻微正向交互。

---

### Table 5: 训练超参数配置

| 设置 | LIBERO | RoboTwin | Astribot | AgileX |
|------|--------|---------|---------|--------|
| 动作维度 | 7 | 14 | 16 | 7 |
| 状态维度 | 8 | 14 | 16 | 7 |
| Horizon H | 8 | 50 | 50 | 50 |
| 训练步数 | 300,000 | 200,000 | 20,000 | 20,000 |
| 梯度累积 | 4 | 8 | 1 | 2 |
| 教师输入分辨率 | 336 | 336 | 336 | 336 |
| Scene 损失权重 | 0.5 | 0.5 | 0.5 | 0.5 |
| EEF 损失权重 | 3.0 | 3.0 | 3.0 | 3.0 |
| 记忆容量 | 16 | 8 | 16 | 16 |
| 采样组大小 | 32 | 16 | 32 | 32 |

---

### Table 7: 几何重建诊断 RMSE（200K checkpoint）

| 任务 | 设置 | 点图 RMSE（归一化）| EEF RMSE (mm) |
|------|------|-----------------|--------------|
| Open microwave | Clean | 0.076 | 3.69 |
| Open microwave | Randomized | 0.094 | 3.74 |
| Stack three blocks | Clean | 0.070 | 3.07 |
| Stack three blocks | Randomized | 0.064 | 3.64 |

**说明**: 这是离线诊断（非闭环），验证 [[GeoPlan Token]] 确实学到了合理的 3D 场景信息和 EEF 轨迹预测（EEF 误差约 3-4mm）。

---

## 实验

### 数据集/Benchmark

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 4 套×10 任务，50 trials/task | 桌面操作，Spatial/Object/Goal/Long 四套 | LIBERO 评估 |
| RoboTwin 2.0 | 50 项双臂任务，100 trials/task×2 设置 | 双臂操作，Clean/Randomized 两种设置 | RoboTwin 评估 |
| Astribot S1 | 3 项真实任务，10 rollouts/task | 真实机器人，精细操作 | 真实环境验证 |
| AgileX Cobot Magic 2 | 2 项真实任务，10 rollouts/task | 双臂机器人，遮挡场景 | 真实环境验证 |

### 实现细节

- **VLM**: Qwen3-VL-4B（冻结）
- **几何教师**: [[VGGT]]（仅训练时，推理移除）
- **动作专家**: GR00T-style Flow-Matching Diffusion Transformer
- **GeoPlan Token 数量** $K$: 未在正文中明确，可从附录推断
- **硬件**: 未详述（可推断为多 GPU）

### 真实机器人结果

| 任务 | 平台 | 成功率 |
|------|------|--------|
| Pick and stand sticks | Astribot S1 | 90% |
| Separate file pockets | Astribot S1 | 100% |
| Transfer cup across rack | Astribot S1 | 70% |
| Cover and find apple under towel | AgileX Cobot Magic 2 | 80% |
| Pick n times cube | AgileX Cobot Magic 2 | 50%（推算） |
| **平均** | — | **78.0%** |

---

## 批判性思考

### 优点
1. **推理时零额外 3D 开销**：训练完成后移除 [[VGGT]] 教师，几何信息完全内化到 [[GeoPlan Token]]，推理速度不受影响。
2. **几何与记忆的有机结合**：将空间理解和时序积累统一到一个框架，而非两个独立模块叠加。
3. **跨平台验证充分**：同时在 LIBERO（单臂桌面）、RoboTwin（双臂仿真）和真实机器人三类环境验证，泛化性有保障。

### 局限性
1. **真实机器人评估规模小**：每任务仅 10 次 rollout，统计显著性不足；实验数量（5 项任务）也相对有限。
2. **无专门局限性章节**：论文没有专门讨论 failure modes 或 limitations，对遮挡率、操作速度等边界条件的分析缺失。
3. **代码未公开**：承诺发表后公开，但目前无法复现。
4. **消融实验深度有限**：未分析 $K$（GeoPlan token 数量）、记忆容量、$\alpha$ 系数等关键超参的敏感性。

### 潜在改进方向
1. 探索在线更新 [[VGGT]] 教师（半冻结）以适应未见场景
2. 将记忆容量和合并策略设计为自适应的，而非固定容量 + 手工合并
3. 与语言指令联合的记忆选择机制（当前检索与语言无关）

### 可复现性评估
- [ ] 代码开源（承诺发表后开源）
- [ ] 预训练模型（HuggingFace 有模型但需验证）
- [x] 训练细节基本完整（附录 Table 5）
- [x] 数据集可获取（LIBERO/RoboTwin 均为公开 benchmark）

---

## 关联笔记

### 基于
- [[StarVLA]]: 基础架构，GeoMem-VLA 在其上扩展几何分支和记忆模块
- [[GR00T]]: 动作专家架构来源（Flow-Matching DiT）
- [[VGGT]]: 几何教师，提供点图和未来 EEF 轨迹监督

### 对比
- [[Spatial Forcing]]: LIBERO 最近强基线（98.5% vs 98.9%）
- [[MemoryVLA]]: 无几何信息的纯记忆 VLA 对比
- [[3DThinkVLA]]: 逐帧 3D 思考 VLA，LIBERO 部分指标持平

### 方法相关
- [[GeoPlan Token]]: 本文核心创新，几何监督可学习 token
- [[Flow Matching]]: 动作预测和几何监督的统一技术框架
- [[Causal Retrieval Memory]]: 因果时序记忆检索机制
- [[Memory Consolidation]]: 记忆库溢出时的合并策略
- [[Action Chunking]]: 输出形式

### 硬件/数据相关
- [[Astribot S1]]: 真实机器人平台

---

## 速查卡片

> [!summary] GeoMem-VLA (arXiv 2610.29866)
> - **核心**: 用 GeoPlan Token 将 3D 几何理解内化为时序记忆，推理时无需 3D 教师
> - **方法**: VGGT 监督 GeoPlan Token + 三路因果记忆（含记忆合并）+ Flow Matching 动作预测
> - **结果**: LIBERO 98.9%、RoboTwin 91.17%、真实机器人 78.0% 均达到 SOTA
> - **代码**: [GitHub](https://github.com/GeoMem-VLA/GeoMem-VLA) / [HuggingFace](https://huggingface.co/rxc205/GEOMEM-VLA)

---

*笔记创建时间: 2026-10-23*
