---
title: "SCOUT: Planning under Occlusion via Object-Centric World Model Rollouts"
method_name: "SCOUT"
authors: [Amir Hossain Raj, Xuesu Xiao]
year: 2026
venue: arXiv
tags: [occlusion-aware-planning, object-centric-representation, world-model, monte-carlo-tree-search, robotic-manipulation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2611.01600v1
created: 2026-11-04
---

# 论文笔记：SCOUT: Planning under Occlusion via Object-Centric World Model Rollouts

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | George Mason University |
| 日期 | November 2026 |
| 项目主页 | — |
| 对比基线 | [[π0.5]], [[Long-VLA]], [[OWMM-Agent]], [[ParticleFormer]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.01600) |

---

## 一句话总结

> SCOUT 在遮挡密集的货架场景中，通过对象中心场景重建、LLM 动作提议与世界模型辅助的 MCTS 规划，显著超越 VLA 和三维世界模型基线。

---

## 核心贡献

1. **结构化隐场景表示（GHOST）**: 从稀疏正面 RGB-D 视图重建遮挡区域的对象中心槽位，构建包含遮挡关系、可见性与可达性的关系规划状态。
2. **动作条件世界模型**: 轻量化 Transformer 预测每个对象的位移、运动标记与有效性，支持在规划时对未来状态进行快速模拟。
3. **LLM 引导的 MCTS 规划**: 以 [[Qwen2.5]] 生成候选动作作为先验概率，配合 MCTS 深度-2 滚动评估，以 receding horizon 方式执行长时程操作规划。

---

## 问题背景

### 要解决的问题

机器人在拥挤货架中操作物体时，前景物体遮挡目标和空闲空间，导致反应式策略无法感知隐藏约束，而穷举搜索又过于昂贵。

### 现有方法的局限

- **VLA 策略**（π0.5、Long-VLA）：端到端反应式，无法规划超过直接可见范围的遮挡问题。
- **具身智能体**（OWMM-Agent）：缺乏显式三维世界模型，对遮挡场景的几何推理有限。
- **三维世界模型**（ParticleFormer）：作为过渡模型使用时，在离散对象级别的精度不及 SCOUT 的专用世界模型。

### 本文的动机

在 Belief Space 中规划需要两个核心组件：(1) 对遮挡几何的结构化表示，(2) 预测动作对遮挡关系影响的世界模型。SCOUT 将两者结合并以 LLM 提议作为搜索先验，解决长时程遮挡感知规划问题。

---

## 方法详解

### 模型架构

SCOUT 采用 **模块化规划** 架构，输入为稀疏正面 RGB-D 观测 $o$ 和目标 $g$：

- **场景重建**: [[GHOST]] 对象中心槽位表示 → 关系规划状态 $x_t$
- **动作提议**: [[Qwen2.5|LLM（Qwen2.5-7B-Instruct）]] 生成候选 $(I_c, d_c)$ 对
- **规划搜索**: [[MCTS]] + 动作条件 [[World Model|世界模型]] 滚动评估
- **动作空间**: 离散对象级原语 $a = (i, u)$，$u \in \{\text{shift\_left, shift\_right, shift\_back, pull\_forward}\}$

![Figure 1: SCOUT 整体流水线](https://arxiv.org/html/2611.01600v1/pipeline_v4.png)

**说明**: 从 RGB-D 观测 → GHOST 重建 → 关系规划状态 → LLM 提议 + MCTS 评估 → 多步规划执行。

### 核心模块

#### 模块1: GHOST 对象中心场景表示

**设计动机**: 利用 [[Slot Attention|槽位注意力（Slot Attention）]] 机制将多视角 RGB-D 观测融合为对象中心槽位，同时为遮挡区域学习隐藏槽位，重建完整三维场景。

**具体实现**:
- 每个槽位 $s_i^t$ 包含三维中心 $\mu_i$、三维尺寸 $\sigma_i$ 和存在概率 $p_i$
- 可见槽位来自各视角的 [[SAM]] SAM3D 先验融合
- 固定数量的**学习型隐藏槽位**表示遮挡对象（架构：128-dim、4 层 Slot Transformer，16 个隐藏槽位）
- Slot Transformer 细化槽位后，通过高斯细化渲染占据体积和关系图

![Figure 2: GHOST 场景表示流水线](https://arxiv.org/html/2611.01600v1/GHOST.png)

**说明**: 多视角 RGB-D → 对象中心槽位融合 → 槽位交互细化 → 统一三维场景表示。

#### 模块2: 关系规划状态

**设计动机**: 将连续槽位表示投影为规划所需的结构化关系图，捕捉遮挡关系、可见性与可达路径。

**具体实现**:
- 关系规划状态 $x_t = P(z_t, g)$ 编码目标、遮挡关系、可见性和粗略空间关系
- **遮挡测试（Eq 5）**: 对象 $i$ 遮挡目标 $t$，当且仅当 $x_i + 10^{-3} < x_t$ 且 YZ 截面重叠 > 5%
- **可见性（Eq 6）**: 基于 YZ 截面遮挡程度计算
- **可达性（Eq 7）**: 基于侧面和背面空闲空间计算

**Figure 3**: 关系规划状态示意（从对象中心槽位导出）。

#### 模块3: 动作条件世界模型

**设计动机**: 实现每个动作对场景状态影响的快速预测，支持 MCTS 中的批量滚动评估（单次前向传播 4.8 ms）。

**具体实现**:
- 模型 $f_\phi(z_t, a_t)$ 预测每个对象的位移 $\Delta\mu_i$、运动标记 $m_i$ 和有效性标记 $v_i$
- 位置更新：$\mu_i^{t+1} = \mu_i^t + m_i \cdot \Delta\mu_i$，尺寸和存在概率保持不变
- **两阶段训练**: 先在仿真转换上训练，再在 GHOST 重建上微调；幻觉槽位监督为零运动
- 架构：2 个 Transformer block，4 头，128-dim

**Figure 4**: 世界模型引导的规划示意（MCTS 深度-2 递归前向预测）。

#### 模块4: LLM 引导的候选动作提议

**设计动机**: 以 [[Qwen2.5|Qwen2.5-7B-Instruct]] 对拥挤场景进行语义理解，生成结构化候选动作作为 [[PUCT]] 的先验概率，减少搜索空间。

**具体实现**:
- LLM 输出提议 $c = (I_c, d_c)$，每个提议配对一组对象索引与粗略方向
- 候选集 $\mathcal{A}(c) = \{(i, d_c) : i \in I_c\} \cup \mathcal{A}_{aux}$，$\mathcal{A}_{aux}$ 为确定性几何回退动作
- LLM 推理延迟：中位数 0.42 s/步

---

## 关键公式

### 公式1: [[遮挡感知规划|规划目标]]

$$
\max_{\{a_t\}} \mathbb{E}\left[r(\tau) \mid o, g, T\right]
$$

**含义**: 在预算 $T$ 步内，给定 RGB-D 观测 $o$ 和目标 $g$，最大化期望任务成功奖励。

**符号说明**:
- $a_t$：$t$ 时刻的动作（对象索引 + 方向原语）
- $r(\tau)$：轨迹 $\tau$ 的任务成功奖励
- $T$：最大步数预算（实验中为 12）

### 公式2: [[MCTS|遮挡测试（Blocker Test）]]

$$
\text{blocks}(i, t) = \mathbb{I}\left[x_i + 10^{-3} < x_t\right] \cdot \mathbb{I}\left[\text{overlap}_{YZ}(i, t) > 0.05\right]
$$

**含义**: 判断对象 $i$ 是否在深度方向遮挡目标 $t$，要求 $i$ 在 $t$ 前方且 YZ 投影重叠超过 5%。

**符号说明**:
- $x_i, x_t$：对象 $i$ 和目标 $t$ 的深度坐标（x 轴）
- $\text{overlap}_{YZ}$：YZ 平面的投影重叠比例

### 公式3: [[遮挡感知规划|成功判断条件]]

$$
\text{Reveal}: \quad \text{vis} \geq 0.6,\; \text{acc} \geq 0.6,\; B = 0
$$

$$
\text{Place}: \quad \mu_{target} \in \mathcal{R}_{goal}
$$

**含义**: Reveal 任务需目标可见度、可达性均 ≥ 0.6 且遮挡计数为零；Place 任务需目标中心进入目标区域。

**符号说明**:
- $\text{vis}$：可见度，基于 YZ 截面遮挡程度
- $\text{acc}$：可达性，基于侧面和背面空闲空间
- $B$：当前遮挡对象计数
- $\mathcal{R}_{goal}$：目标放置区域

### 公式4: [[World Model|世界模型损失]]

$$
\mathcal{L} = 1.0\,\mathcal{L}_{disp} + 0.5\,\mathcal{L}_{moved} + 0.25\,\mathcal{L}_{valid} + 0.5\,\mathcal{L}_{halluc}
$$

**含义**: 世界模型以加权多任务损失训练，第二阶段在 GHOST 重建上微调时增加幻觉惩罚项。

**符号说明**:
- $\mathcal{L}_{disp}$：遮掩的 Smooth-L1 位移损失（权重 1.0）
- $\mathcal{L}_{moved}$：运动对象 BCE 损失（权重 0.5）
- $\mathcal{L}_{valid}$：有效性 BCE 损失（权重 0.25）
- $\mathcal{L}_{halluc}$：幻觉槽位惩罚，仅第二阶段（权重 0.5）

### 公式5: [[PUCT|PUCT 动作选择]]

$$
a^* = \arg\max_a \left[Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1+N(s,a)}\right], \quad c_{puct} = 1.25
$$

**含义**: MCTS 节点选择时，以 PUCT 公式平衡价值估计 $Q$ 与 LLM 先验 $P$ 和访问次数上置信界。

**符号说明**:
- $Q(s,a)$：当前状态-动作对的价值估计
- $P(s,a)$：LLM 提议为该动作赋予的先验概率
- $N(s)$：当前状态的总访问次数
- $N(s,a)$：动作 $a$ 的访问次数
- $c_{puct} = 1.25$：探索系数

### 公式6: [[遮挡感知规划|滚动评分函数]]

$$
V(\tau) = \Delta_{vis} + \Delta_{acc} - 0.25\,B_{end} - 0.10\,\max(0, \Delta_{block}) - 1.0\,\mathbb{I}_{invalid}
$$

**含义**: 对深度-2 MCTS 滚动轨迹 $\tau$ 进行综合评分，同时奖励可见性/可达性提升，惩罚遮挡增加与无效动作。

**符号说明**:
- $\Delta_{vis}$：滚动前后可见度变化量（越大越好）
- $\Delta_{acc}$：滚动前后可达性变化量（越大越好）
- $B_{end}$：滚动结束时的遮挡对象计数（惩罚系数 0.25）
- $\Delta_{block}$：遮挡计数增量（惩罚系数 0.10，仅惩罚正增加）
- $\mathbb{I}_{invalid}$：是否包含无效动作的指示变量（惩罚系数 1.0）

---

## 关键图表

### Figure 1: SCOUT 整体流水线

![Figure 1: SCOUT pipeline](https://arxiv.org/html/2611.01600v1/pipeline_v4.png)

**说明**: 从稀疏 RGB-D 观测经 GHOST 重建到 LLM 提议与 MCTS 规划的完整流程。

### Figure 2: GHOST 场景重建流水线

![Figure 2: GHOST pipeline](https://arxiv.org/html/2611.01600v1/GHOST.png)

**说明**: 多视角 [[RGB-D]] 观测通过 [[SAM]] SAM3D 先验生成初始槽位，Slot Transformer 细化后渲染统一三维场景表示。

### Figure 5: 四种任务类型示例

![Figure 5: Task families](https://arxiv.org/html/2611.01600v1/tasks.png)

**说明**: 四类任务：Reveal（将目标物体移出遮挡）、Reveal_Multi（多目标）、Place（将物体放入目标区域）、Place_Multi（多目标），评估需 8-16 个 [[YCB]] / ProcTHOR / RoboCasa 资产。

### Figure 7: 失败原因分解

![Figure 7: Failure analysis](https://arxiv.org/html/2611.01600v1/failure.png)

**说明**: 失败类型分布：隐场景重建误差 7.6%、过渡预测误差 5.2%、几何有效性误差 4.8%、搜索选择误差 1.6%。同时展示三类典型失败案例。

### Figure 9: 真实机器人实验

![Figure 9: Real-world rollouts](https://arxiv.org/html/2611.01600v1/figures/demo.png)

**说明**: 在 [[Boston Dynamics Spot]] 腕部相机上执行 Reveal 和 Place 任务的真实场景滚动演示。

### Figure 10: 稀疏 RGB-D 输入样本

![Figure 10: Sparse RGB-D observations](https://arxiv.org/html/2611.01600v1/samples_app.png)

**说明**: 每行为同一拥挤货架的不同正面视角，含 RGB 图、深度图和分割掩码，是 GHOST 仅有的输入。

### Figure 11: 真实世界数据采集装置

![Figure 11: Real-world setup](https://arxiv.org/html/2611.01600v1/real_robot_app.png)

**说明**: Boston Dynamics Spot 利用腕部相机观察拥挤货架，右图为采集到的 RGB 图与深度图对。

### Table 1: 仿真主要结果（成功率 %）

| 方法 | Reveal | Reveal_Multi | Place | Place_Multi | 平均 |
|------|--------|--------------|-------|-------------|------|
| π0.5 | — | — | — | — | ≤72.8 |
| Long-VLA | — | — | — | — | ≤72.8 |
| OWMM-Agent | — | — | — | — | ≤72.8 |
| ParticleFormer | — | — | — | — | 72.8 |
| **SCOUT（ours）** | **86.8** | **~78** | **88.2** | **~79** | **80.8** |

**说明**: SCOUT 在所有四类任务上均超过最强基线约 8 个百分点；多目标任务提升约 9 个点。

### Table 1（附）: 滚动指标（平均步数 / 无效率）

| 任务 | 平均步数 | 无效动作率 |
|------|---------|-----------|
| Reveal | 6.3 | 3.2% |
| Reveal_Multi | 8.7 | 5.8% |
| Place | 6.0 | 4.1% |
| Place_Multi | 9.1 | 6.4% |

### Table 2: 泛化实验成功率（%）

| 设置 | SCOUT | 最强基线 |
|------|-------|---------|
| 分布内（In-distribution） | 80.8 | 72.8 |
| 新对象（Novel objects） | 75.4 | 67.6 |
| 新视角（Novel views） | 76.8 | 69.3 |

**说明**: SCOUT 在所有泛化设置下均保持约 7-8 点的领先优势。

### Table 3: 消融实验成功率（%）

| 配置 | 平均成功率 | 分析 |
|------|-----------|------|
| 单视角输入 | 69.7 | -11.1 多视角融合关键 |
| 无关系投影 | 68.1 | -12.7 结构化状态重要 |
| 无滚动搜索 | 70.6 | -10.2 MCTS 规划必要 |
| **完整 SCOUT** | **80.8** | — |

**关键发现**: 三大组件（多视角融合、关系状态、MCTS 滚动）缺一不可，各自贡献约 10 个百分点。

---

## 实验

### 数据集/仿真环境

| 环境/数据集 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| [[ManiSkill2]] 货架仿真 | 500 rollout × 3 seed | 8-16 对象，0.36×0.90×0.40 m | 训练 + 测试 |
| [[YCB]] | 77 个对象 | 标准操作基准对象 | 仿真对象资产 |
| ProcTHOR | 97 个对象 | 程序生成家居对象 | 仿真对象资产 |
| RoboCasa | 16 个对象 | 厨房场景对象 | 仿真对象资产 |
| Boston Dynamics Spot 真实货架 | 20 次试验 | 腕部 RGB-D 相机，含眩光噪声 | 真实世界验证 |

### 实现细节

- **GHOST 架构**: 128-dim Slot Transformer，4 层，16 个隐藏槽位
- **世界模型**: 2 个 Transformer block，4 头，128-dim
- **训练 Phase 1**: 8 epoch，lr $3\times10^{-4}$（仿真转换）
- **训练 Phase 2**: 4 epoch，lr $10^{-4}$（GHOST 重建微调）
- **MCTS**: 32 次迭代，深度 2，分支因子 6
- **LLM**: Qwen2.5-7B-Instruct，temperature 0.2，top-p 0.9
- **相机**: 16 候选视角，最多选 5 个
- **推理延迟**: LLM 中位数 0.42 s，MCTS 0.68 s/步，世界模型 4.8 ms/forward

### 真实世界结果

在 Boston Dynamics Spot 机器人的 20 次试验中，SCOUT 成功 **12/20（60%）**，覆盖 Reveal 和 Place 任务。失败原因包括夹爪滑落和眩光导致重建噪声引发的规划错误。

---

## 批判性思考

### 优点

1. **结构化遮挡推理**: 显式建模遮挡关系，使规划对可见性动态变化有直接响应，而非依赖端到端黑盒策略。
2. **计算高效的世界模型**: 4.8 ms 单次前向使 MCTS 在 0.68 s/步内完成，满足实际交互速率要求。
3. **模块化可替换**: GHOST、世界模型、LLM 提议三个模块彼此独立，每个模块均可单独升级。

### 局限性

1. **依赖重建精度**: 重建误差直接传播到规划，眩光等噪声源对真实场景性能影响明显（真实 60% vs 仿真 80%）。
2. **离散动作空间**: 仅限四种方向原语，无移出（remove）动作，无法处理接触丰富的连续控制。
3. **静态场景假设**: 规划期间假设场景不变，无法处理动态干扰或并行操作。
4. **无方向表示**: 对象姿态未纳入模型，限制姿态敏感任务的性能。

### 潜在改进方向

1. 引入物理感知世界模型（含接触力），改善复杂接触动力学预测。
2. 扩展连续动作空间，支持自适应轨迹生成。
3. 引入在线自适应，使重建在执行过程中持续细化。

### 可复现性评估

- [ ] 代码开源（论文未提及）
- [ ] 预训练模型（论文未提及）
- [x] 训练细节完整（附录 A 有详细描述）
- [x] 数据集可获取（ManiSkill2 + YCB 均为公开）

---

## 关联笔记

### 基于

- [[GHOST]]: 对象中心场景重建核心组件
- [[SAM]]: 提供 SAM3D 分割先验
- [[Slot Attention]]: 槽位机制的基础概念

### 对比

- [[π0.5]]: VLA 策略基线，无显式规划
- [[Long-VLA]]: 长时程 VLA 基线
- [[OWMM-Agent]]: 具身智能体基线
- [[ParticleFormer]]: 三维粒子世界模型，作为 SCOUT 过渡模型的替代基线

### 方法相关

- [[MCTS]]: 核心规划算法
- [[PUCT]]: MCTS 节点选择策略
- [[World Model]]: 动作条件世界模型概念
- [[Qwen2.5]]: LLM 动作提议生成
- [[遮挡感知规划|Occlusion-Aware Planning]]: 核心问题域

### 硬件/数据相关

- [[ManiSkill2]]: 仿真平台
- [[YCB]]: 操作对象数据集
- [[Boston Dynamics Spot]]: 真实机器人平台

---

## 速查卡片

> [!summary] SCOUT: Planning under Occlusion via Object-Centric World Model Rollouts
> - **核心**: 对象中心场景重建 + 世界模型 MCTS 规划，解决拥挤货架遮挡操作
> - **方法**: GHOST 槽位重建 → 关系状态 → LLM 提议 + MCTS 深度-2 滚动
> - **结果**: 仿真平均 80.8%（vs 基线 72.8%），真实机器人 60%（20 次试验）
> - **代码**: 未公开

---

*笔记创建时间: 2026-11-04*
