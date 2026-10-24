---
title: "FocusVLA: Future-Intent Guided Dynamic View Routing for Vision-Language-Action Models"
method_name: "FocusVLA"
authors: [Huixin Xu, Ziya Yu, Peng Li]
year: 2026
venue: CoRL 2026
tags: [vla, multi-view-perception, view-routing, intent-prediction, robot-manipulation]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2610.31423v1
created: 2026-10-24
---

# 论文笔记：FocusVLA: Future-Intent Guided Dynamic View Routing for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未披露 |
| 日期 | October 2026 |
| 项目主页 | 无 |
| 对比基线 | [[Pi05]], [[Pi0]], [[RDT]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.31423) / Code: 未开源 |

---

## 一句话总结

> FocusVLA 通过预测未来交互意图并将其转化为动态视角路由权重，解决多视角 [[VLA]] 模型中固定视觉融合导致的"视角依赖惯性"问题，在 RoboTwin 2.0 上比 [[Pi05]] 提升 10.10 个百分点。

---

## 核心贡献

1. **发现视角依赖惯性**: 量化了多视角 [[VLA]] 模型在任务不同阶段对腕部相机的过度依赖，定义 [[Dominance Margin]] 和局部依赖变化度两个度量
2. **Future Intent Bridge 模块**: 基于历史动作和机器人状态预测未来 $H=30$ 步的交互意图，生成 chunk 级意图 token 和逐步路由权重
3. **Action-Side Dynamic Routing**: 在动作生成侧将全局/局部视觉特征按意图引导的权重动态融合，作为非侵入式插件适配现有 [[VLA]] 框架

---

## 问题背景

### 要解决的问题

多视角 [[VLA]] 策略通常将第三人称（全局）相机和腕部（局部）相机的特征固定融合后输入模型，导致模型在整个任务执行过程中对视角的依赖比例不随任务阶段动态调整。

### 现有方法的局限

- 以 [[Pi05]] 为代表的方法将所有相机帧平等地拼入上下文，不区分任务阶段（规划期 vs. 操作期）的差异化视觉需求
- BFA 等方法进行了一定程度的视角融合，但仍采用 per-step 或固定的分配比例，且使用离线 VLM 标注标签，不具备实时预测能力
- 作者通过遮蔽实验证明：在 Stack Blocks Three 任务规划阶段遮蔽局部视图会使成功率从 85% 跌至 20%，而遮蔽全局视图仅跌至 65%，说明当前模型存在错误的静态视角优先

### 本文的动机

机器人操作任务在不同阶段对视角的需求本质上不同：规划和定位阶段需要全局视图以把握空间关系，精确抓取和对齐阶段需要局部腕部视图以获取细粒度信息。作者认为若能预测何时进行交互（"意图"），就能提前调整视角路由权重，从而消除视角惯性。

---

## 方法详解

### 模型架构

FocusVLA 采用**非侵入式插件**架构，在不修改视觉编码器、语言模型和动作主干的前提下，向现有 [[Pi05]] 框架注入两个模块：

- **输入**: 语言指令 $l$ + 全局相机观测 $o^g_t$ + 局部腕部相机观测 $o^l_t$ + 机器人状态 $s_t$ + 历史动作序列 $\{a_{t-T}, ..., a_{t-1}\}$
- **Backbone**: [[Pi05]] 的视觉语言动作流匹配主干（保持冻结）
- **核心模块**: [[Future Intent Prediction|Future Intent Bridge]] 用于预测意图并生成路由权重；[[Dynamic View Routing|Action-Side Dynamic Routing]] 用于在动作生成侧动态融合视觉特征
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H}$，$H=30$
- **新增参数**: 70.13M 参数（相对于基础模型为插件级别增量）

### 核心模块

#### 模块1：Future Intent Bridge

**设计动机**: 利用[[Cross-Attention]]机制，从历史轨迹和语言指令中预测未来交互意图窗口，使路由权重具备前瞻性。

**具体实现**:
- 编码机器人状态 $s_t$ 和前 $T=30$ 步历史动作 $\{a_{t-T}, ..., a_{t-1}\}$
- 以语言特征为 Query，视觉特征与历史动作特征为 Key/Value，执行[[Cross-Attention]]操作
- 输出 $H=30$ 步的逐步意图概率序列 $I_t = [i^{(1)}, ..., i^{(H)}]$，$i^{(k)} \in [0, 1]$
- 同时生成一个 chunk 级意图 token，注入动作生成主干

**标签构建** (无需人工标注):
- 若轨迹包含夹爪关闭信号：检测 $\Delta g_t < -0.03$ 持续 $\geq 4$ 帧的窗口
- 否则：基于末端执行器运动速度的阈值（高速阈值 $p_{70}$，低速阈值 $p_{45}$），识别低速交互段
- 额外处理按压类和换手类动作的特殊情形

#### 模块2：Action-Side Dynamic Routing

**设计动机**: 在动作生成侧而非视觉特征提取侧进行融合，使路由权重直接影响动作预测，避免早期融合导致的信息丢失。

**具体实现**:
- 动作特征作为 Query，分别与全局视觉特征和局部视觉特征进行[[Cross-Attention]]，得到 $c_g^{(k)}$ 和 $c_l^{(k)}$
- 路由权重 $\alpha_g^{(k)}, \alpha_l^{(k)}$ 由意图概率 $i^{(k)}$ 通过有界线性映射生成，限制在 $(0.25, 0.75)$ 区间（最大 3:1 比例）
- 加权融合后通过残差连接和轻量级精化模块注入动作主干

---

## 关键公式

### 公式1：[[Dominance Margin|显性边距]]

$$
D_t = |a_{\text{global},t} - a_{\text{local},t}|
$$

**含义**: 衡量每一步模型对全局视图和局部视图的注意力差异绝对值，$D_t$ 越大表示模型的视角依赖越不均衡。

**符号说明**:
- $a_{\text{global},t}$: 第 $t$ 步模型对全局视图 token 的平均注意力权重
- $a_{\text{local},t}$: 第 $t$ 步模型对局部视图 token 的平均注意力权重

### 公式2：局部依赖变化度

$$
V = \text{Var}(\Delta L_t), \quad L_t = a_{\text{local},t}
$$

**含义**: 局部视图注意力随时间步的方差，衡量模型是否能根据任务阶段灵活调整视角依赖。失败轨迹通常展现出更低的 $V$。

**符号说明**:
- $L_t$: 第 $t$ 步局部视角注意力权重
- $\Delta L_t = L_t - L_{t-1}$: 相邻步的差分

### 公式3：[[Dynamic View Routing|动态视角融合]]

$$
c^{(k)} = \alpha_g^{(k)} \cdot c_g^{(k)} + \alpha_l^{(k)} \cdot c_l^{(k)}
$$

**含义**: 第 $k$ 步的融合视觉上下文，由全局和局部动作侧[[Cross-Attention]]输出按意图路由权重加权求和。

**符号说明**:
- $c_g^{(k)}$: 第 $k$ 步全局视图的动作侧 [[Cross-Attention]] 输出
- $c_l^{(k)}$: 第 $k$ 步局部视图的动作侧 [[Cross-Attention]] 输出
- $\alpha_g^{(k)}, \alpha_l^{(k)} \in (0.25, 0.75)$: 由意图概率 $i^{(k)}$ 映射的路由权重

### 公式4：夹爪关闭事件检测

$$
\Delta g_t < -0.03 \text{ 且持续 } \geq 4 \text{ 帧}
$$

**含义**: 识别夹爪关闭事件以标记交互窗口起始，$g_t$ 为第 $t$ 帧夹爪开合度。

### 公式5：速度阈值

$$
v_{\text{high}} = p_{70}(v), \quad v_{\text{low}} = p_{45}(v)
$$

**含义**: 分别用速度分布的第 70 和第 45 百分位数定义高速和低速阈值，用于无夹爪信号时识别低速交互段。

### 公式6：融合路径长度

$$
L_{\text{seg}} = \sum_{t \in \text{seg}} \max\left(\|p_t^L - p_{t-1}^L\|, \|p_t^R - p_{t-1}^R\|\right)
$$

**含义**: 计算一段轨迹内左右末端执行器的最大逐帧位移之和，用于区分有效交互段和无效短段。

### 公式7：自适应路径上限

$$
L_{\text{cap}} = \max\left(0.02, \ 2 \times p_{80}(L_{\text{seg}})\right)
$$

**含义**: 基于数据自适应地过滤过短的交互段噪声，避免对异常长段过拟合。

### 公式8：二值意图标签

$$
y^{(k)} = \mathbb{1}[k \in \text{interaction interval}]
$$

**含义**: 将交互窗口内的未来步标记为正样本（$y=1$），其余为负样本（$y=0$）。

### 公式9：[[View-Dependency Inertia|类别平衡意图损失]]

$$
\mathcal{L}_{\text{intent}} = -\left[\lambda_+ \cdot y \log(i) + \lambda_- \cdot (1-y) \log(1-i)\right]
$$

**含义**: 正负样本不均衡的加权 BCE 损失，$\lambda_+ = 15$（正类权重），$\lambda_- = 1$（负类权重）。

**符号说明**:
- $\lambda_+ = 15$: 正类（交互期）权重，缓解类别不均衡
- $\lambda_-$: 负类权重（默认为 1）
- $i$: 模型预测的意图概率

### 公式10：总损失

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{act}} + \lambda_{\text{intent}} \cdot \mathcal{L}_{\text{intent}}
$$

**含义**: 动作回归损失与意图预测损失联合优化，$\lambda_{\text{intent}} = 1$。

**符号说明**:
- $\mathcal{L}_{\text{act}} = \|V_t - U_t\|^2_2$: 流匹配动作损失（预测速度场与真实场之差）
- $\mathcal{L}_{\text{intent}}$: 类别平衡意图 BCE 损失
- $\lambda_{\text{intent}} = 1$: 意图损失权重

---

## 关键图表

### Figure 1：Overview / 系统概览

![Figure 1](https://arxiv.org/html/2610.31423v1/overview.png)

**说明**: FocusVLA 的整体概览。左侧展示静态多视角融合导致"[[View-Dependency Inertia|视角依赖惯性]]"的问题，右侧展示 Future Intent Bridge 预测意图后通过 [[Dynamic View Routing|Action-Side Dynamic Routing]] 动态调整全局/局部视角权重的解决方案。

### Figure 3：Motivation Analysis / 动机分析

![Figure 3](https://arxiv.org/html/2610.31423v1/focusvla-analysis.png)

**说明**: 基于 [[Pi05]] 的视角依赖惯性定量分析。展示失败轨迹相较于成功轨迹具有更大的 [[Dominance Margin]] 和更小的局部依赖变化度，证明静态融合的视角切换能力不足。

### Figure 4：Architecture / FocusVLA 架构

![Figure 4](https://arxiv.org/html/2610.31423v1/methods.png)

**说明**: FocusVLA 的完整模块架构。左侧为 Future Intent Bridge，包含状态/历史动作编码、语言引导[[Cross-Attention]]和意图序列预测；右侧为 Action-Side Dynamic Routing，展示全局/局部特征的逐步加权融合。

### Figure 5：Real-World Setup / 真实场景评估

![Figure 5](https://arxiv.org/html/2610.31423v1/realworld.png)

**说明**: 真实机器人（Piper 机械臂 + 两台 RealSense D435i）的实验设置和 6 个测试任务（每任务 30 次试验）。FocusVLA 在所有 6 项任务上均超越 [[Pi05]]。

### Figure 6：Spatial Adaptability / 空间泛化性

![Figure 6](https://arxiv.org/html/2610.31423v1/spatial.png)

**说明**: 在未见过的物体摆放位置下的执行稳定性对比。FocusVLA 相比静态融合基线在空间布局变化时表现更稳健，体现出意图引导路由对空间推理的辅助作用。

### Figure 7：Stage-Aware Routing Visualization / 阶段感知路由可视化

![Figure 7](https://arxiv.org/html/2610.31423v1/visualize.png)

**说明**: Stack Blocks Three 任务的路由权重随时间步变化曲线。定位/规划阶段全局视图权重高，抓取/对齐阶段局部腕部视图权重提升，堆叠阶段趋于平衡，验证了意图引导路由的阶段感知能力。

### Table 1：遮蔽消融实验（Stack Blocks Three）

| 设置 | 成功率 (%) |
|------|----------|
| 无遮蔽（基线） | 85.0 |
| 规划阶段遮蔽全局视图 | 65.0 |
| 规划阶段遮蔽局部视图 | 20.0 |

**说明**: 验证任务不同阶段对不同视角的依赖程度不同。遮蔽局部视图影响远大于遮蔽全局视图，印证了"视角依赖惯性"问题的真实性。

### Table 2：RoboTwin 2.0 主实验结果（50 任务，每任务 100 次）

| 模型 | 短时域 | 中时域 | 中长时域 | 长时域 | 超长/灵巧 | **总体平均** |
|------|-------|-------|---------|-------|---------|------------|
| RDT | 49.90 | 14.60 | 28.80 | 29.50 | 51.00 | 34.76 |
| π0 | 56.20 | 28.40 | 38.30 | 45.50 | 66.90 | 47.06 |
| π0.5 | 63.10 | 54.90 | 72.00 | 77.10 | 67.00 | 66.82 |
| **FocusVLA** | **74.00** | **67.80** | **80.70** | **85.60** | **76.50** | **76.92** |

**说明**: FocusVLA 在所有时域分组上均超越 [[Pi05]]，总体平均成功率提升 **+10.10 pp**，在长时域任务上提升最为显著（+8.50 pp）。

### Table 3：消融实验（9 个任务，2 seed，每条件 100 次）

| 变体 | 成功率 (%) |
|------|----------|
| **FocusVLA（完整）** | **73.7** |
| Learned Static（学习静态融合权重） | 67.5 |
| Reactive-Vision（基于当前帧视觉） | 57.4 |
| Time + State（时间步 + 状态，无未来预测） | 58.5 |
| E2E Future（无意图监督，端到端） | 55.7 |
| Routing Only（无意图 token） | 63.3 |
| Random Masking（RDT 风格） | 54.6 |

**关键发现**: 意图监督是最关键因素（E2E Future 无监督时掉 18 pp）；chunk 级意图 token 和逐步路由权重互补（去掉 token 从 73.7 降至 63.3）；仅学习静态权重（67.5）也有提升，但远不如动态路由。

### Table 4：BFA 对比控制实验（4 任务，100 次）

| 变体 | 成功率 (%) |
|------|----------|
| FocusVLA 完整 | 79.5 |
| Feature-side fusion（BFA 融合位置） | 67.0 |
| BFA-style labels（VLM 离线标注）| 72.0 |
| Reactive (H_r=1)（仅预测 1 步） | 55.5 |
| Reactive w/o token | 49.8 |
| Routing Only | 71.5 |
| Reactive-Vision | 61.3 |

**关键发现**: 与 BFA 的主要差异在于融合位置（动作侧 vs. 特征侧，+12.5 pp）和标签构建（轨迹标签 vs. VLM 离线标注，+7.5 pp），两者均贡献显著。

### Table 5：意图标签质量评估

| 标签方案 | 人工标注 F1 (%) | 保留意图 F1 (%) | 保留意图 AUROC (%) |
|---------|---------------|---------------|-----------------|
| FocusVLA 轨迹标签 | 82.9 | 78.4 | 91.2 |
| BFA-style VLM 标注 | 65.1 | — | — |

**关键发现**: 轨迹导出标签与人工标注一致性（F1=82.9%）显著高于 VLM 离线标注（65.1%）；推理时注入 10-20% 标签噪声仅使成功率变化 ≤4.5 pp，说明模型具有一定的鲁棒性。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboTwin 2.0]] | 50 任务 × 50 演示 = 2500 轨迹 | 全面覆盖不同时域长度和灵巧度的机器人操作任务 | 训练+测试（每任务 100 次评估） |
| 真实 Piper 机械臂 | 6 任务 × 30 试验 = 180 次 | 双 RealSense D435i 相机，真实世界任务 | 测试 |

### 实现细节

- **Backbone**: [[Pi05]]（视觉语言动作流匹配，冻结）
- **优化器**: 使用 [[Pi05]] 标准训练设定
- **意图损失权重**: $\lambda_{\text{intent}} = 1$，正类权重 $\lambda_+ = 15$
- **意图预热**: 前 5K 步仅训练意图模块，5K-15K 步通过 teacher forcing 退火（0.77 → 0.3），推理时关闭
- **Batch Size**: 未指明
- **训练轮数**: 100K steps
- **硬件**: 8×A100 80GB GPU
- **推理延迟**: 每 chunk 从 79.59 ms 增至 92.78 ms（+13.19 ms，+16.6%，batch_size=1）

### 可视化结果

在 Stack Blocks Three 任务中，路由权重可视化（Figure 7）直接印证了意图预测的有效性：全局视图在定位规划阶段主导，局部视图在精确抓取对齐阶段主导，堆叠阶段两者趋于平衡——与直觉高度一致。

---

## 批判性思考

### 优点
1. **问题定义清晰**: 首次系统量化 VLA 多视角融合中的"视角依赖惯性"，[[Dominance Margin]] 和时序变化度是可复现的分析工具
2. **非侵入式设计**: 仅增加 70M 参数，可直接适配 [[Pi05]] 等现有框架，工程落地成本低
3. **标签构建自动化**: 轨迹导出意图标签无需人工/VLM 标注，且与人工标注一致性更高（F1: 82.9% vs. 65.1%）

### 局限性
1. **意图表示单一**: 仅用二值交互意图驱动路由，未建模子目标、物体级交互状态或语言接地的精细意图
2. **泛化性未验证**: 实验仅在 [[Pi05]] 骨干上验证，跨机器人形态和不同相机配置的适用性未探索
3. **消融与主实验不可比**: Table 3/4 使用不同任务子集和步数，无法直接与 Table 2 主结果对比

### 潜在改进方向
1. 引入子目标层级或物体级交互状态作为更丰富的意图表示
2. 扩展到更多骨干模型（[[RDT]]、OpenVLA 等）和跨形态机器人验证泛化性
3. 探索用连续路由权重替代二值意图标签的端到端学习方案

### 可复现性评估
- [ ] 代码开源（论文未提供）
- [ ] 预训练模型（论文未提供）
- [x] 训练细节完整（标签构建流程、超参数均在附录中详述）
- [x] 数据集可获取（[[RoboTwin 2.0]] 为公开 benchmark）

---

## 关联笔记

### 基于
- [[Pi05]]: 直接基于 π0.5 构建，复用其视觉编码器、语言模型和流匹配动作主干
- [[Pi0]]: π0.5 的前身，作为对比基线之一
- [[RDT]]: 另一个多视角 VLA 基线，提供 Random Masking 对比变体

### 对比
- [[Pi05]]: 主要基线，FocusVLA 在 RoboTwin 2.0 上超越 +10.10 pp
- [[Pi0]]: FocusVLA 超越 +29.86 pp
- [[RDT]]: FocusVLA 超越 +42.16 pp

### 方法相关
- [[Cross-Attention]]: Future Intent Bridge 和 Action-Side Routing 的核心运算
- [[Action Chunking]]: FocusVLA 输出 $H=30$ 步动作块，与 chunk 级意图 token 对应
- [[Future Intent Prediction]]: 核心新概念，从历史轨迹预测未来交互意图
- [[Dynamic View Routing]]: 核心新概念，意图驱动的逐步视角融合权重调整
- [[View-Dependency Inertia]]: 论文定义的关键问题概念

### 硬件/数据相关
- [[RoboTwin 2.0]]: 主要评估 benchmark（50 任务）
- Piper 机械臂 + RealSense D435i: 真实世界实验平台

---

## 速查卡片

> [!summary] FocusVLA (CoRL 2026)
> - **核心**: 多视角 VLA 存在"视角依赖惯性"，需动态调整全局/局部视角权重
> - **方法**: Future Intent Bridge 预测未来 30 步意图 → Action-Side Routing 加权融合视觉特征
> - **结果**: RoboTwin 2.0 上比 π0.5 提升 +10.10 pp（76.92% vs. 66.82%）；真实场景 85.0% vs. 71.7%
> - **代码**: 未开源

---

*笔记创建时间: 2026-10-24*
