---
title: "Temporal GRPO: Beyond Trajectory-Level Credit in Vision-Language-Action Reinforcement Learning"
method_name: "TemporalGRPO"
authors: [Yao Zhou, Hang Gao, Fengge Wu, Changwen Zheng, Wenwen Qiang]
year: 2026
venue: arXiv
tags: [vla, reinforcement-learning, temporal-credit-assignment, grpo, long-horizon-manipulation, robot-manipulation, stage-conditioned-learning]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.13026
created: 2026-08-15
---

# 论文笔记：Temporal GRPO

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开（论文中未明确列出机构） |
| 日期 | August 2026 |
| 项目主页 | N/A |
| 对比基线 | [[SimpleVLA-RL]] / [[OpenVLA-OFT]] / [[Shared-Prefix GRPO]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.13026) / Code N/A |

---

## 一句话总结

> Temporal GRPO 将 VLA 强化学习的优势估计从"轨迹级别"细化到"任务阶段级别"，通过阶段内相对比较解决信用混叠问题，在 RoboTwin 2.0 上以 75.8% 成功率超越 SimpleVLA-RL 7 个百分点。

---

## 核心贡献

1. **问题诊断——轨迹级别信用混叠**: 首次系统性阐述[[轨迹级别信用混叠]]现象：[[GRPO]] 以单一标量优势覆盖整个轨迹所有动作，导致早期阶段已成功完成的动作被后期失败惩罚压制。
2. **阶段构建——Stage Compiler**: 利用冻结的 [[RynnBrain-4B]] 从任务指令和初始观测生成语义候选阶段，再通过 Stage Compiler 编译为带先决条件依赖的可检测信用阶段序列。
3. **时序信用重建——[[阶段条件优势]]**: 仅在进入同一阶段的轨迹间做组内相对比较，将阶段优势专属分配给对应动作区间，实现精准的[[时序信用分配]]而无需修改 VLA 架构或引入额外价值网络。

---

## 问题背景

### 要解决的问题

[[VLA（视觉-语言-动作模型）]] 的结果驱动 RL 后训练（GRPO 类方法）对长时序操作任务效果欠佳。原因在于：一次任务通常包含多个阶段（到达→抓取→搬运→放置），当一条轨迹在第 3 阶段失败时，第 1、2 阶段已经正确完成的动作依然会被分配与"一开始就失败"的轨迹完全相同的负优势，这种现象即[[轨迹级别信用混叠]]。

### 现有方法的局限

- **Trajectory-GRPO**: 以整条轨迹的最终结果为信号，无法区分"失败于何阶段"，对早期成功动作的惩罚抑制了有价值的子技能。
- **Stage-Reward GRPO**: 将阶段进度转化为阶段奖励标量后仍做轨迹级分配，信用仍混叠于全序列。
- **SimpleVLA-RL**: 结果驱动的代表性方法，在长时序任务上提升有限（68.8% 宏平均成功率）。

### 本文的动机

若能将"比较单元"从整条轨迹收窄到"同一阶段内的轨迹子集"，则优势估计更精准、梯度信号更干净。只要能可靠地检测每条轨迹何时完成各阶段，就可将[[GRPO]] 的组内相对优势思想应用于阶段级别。

---

## 方法详解

### 模型架构

Temporal GRPO 是一种**训练框架**（非新架构），以[[OpenVLA-OFT]] SFT 检查点为基础策略，通过以下四步重构信用分配：

- **输入**: 语言指令 $l$ + 初始观测 $o_0$ + 仿真器特权状态（仅训练时）
- **阶段生成**: [[RynnBrain-4B]] 语义阶段生成 → Stage Compiler 阶段编译
- **对齐**: 关系检测器基于特权仿真器状态定位每条轨迹各阶段完成时刻
- **优化目标**: [[阶段条件优势]] 替换原始 GRPO 的轨迹级别优势
- **输出**: 单一统一 [[VLA（视觉-语言-动作模型）|VLA 策略]]，推理时不依赖阶段检测器或特权状态

### 核心模块

#### 模块 1: 任务阶段构建（Task-Conditioned Stage Generation）

**设计动机**: [[轨迹级别信用混叠]]的根本原因是缺乏阶段感知，因此需要先从任务指令中提取可检测的语义阶段。

**具体实现**:
- **语义阶段生成**: 冻结 [[RynnBrain-4B]] 接收任务指令 $l$ 和初始观测 $o_0$，输出候选语义阶段列表
- **Stage Compiler 编译**:
  - 规范化阶段顺序，构建先决条件依赖（$m_1 \to m_2 \to \cdots \to m_K$）
  - 将自然语言描述翻译为可检测的完成条件
  - 消解冗余边界；最终阶段 $m_K$ 与原始任务成功条件对齐，保证局部信用与全局目标一致

#### 模块 2: 轨迹到阶段对齐（Rollout-to-Stage Alignment）

**设计动机**: 跨轨迹比较需要"等进度"作为前提，原始时间步不足以对齐，需要语义级进度锚点。

**具体实现**:
- 关系检测器在训练时读取**特权仿真器状态**，逐步评估每个阶段的完成条件
- 引入稳定性约束：避免瞬态接触或视觉抖动触发误报
- 阶段完成时刻定义：$T_{i,k} = \min\{t \mid m_k \text{ 在轨迹 } i \text{ 中稳定完成}\}$
- 成功阶段的动作区间：$B_{i,k} = (T_{i,k-1}, T_{i,k}]$
- 失败阶段（无法完成 $m_k$）的动作区间：$B_{i,k} = (T_{i,k-1}, T_i]$（轨迹剩余部分）

#### 模块 3: 阶段条件优势重建（Stage-Conditioned Advantage Reconstruction）

**设计动机**: 只在"进入同一阶段"的轨迹间做相对比较，才能获得无混叠的信用信号。

**具体实现**:
- 阶段参与变量 $V_{i,k}$：轨迹 $i$ 成功完成前置阶段 $m_{k-1}$ 才置为 1，否则排除在外（不计为失败）
- 二值阶段结果 $R_{i,k}$：完成 $m_k$ 为 1，未完成为 0
- 阶段内统计量 $\mu_k, \sigma_k$ 仅在参与该阶段的轨迹上计算
- 阶段级别优势 $\widehat{A}_{i,k}$ 专属分配给动作区间 $B_{i,k}$，最终得到时序变化的优势序列

---

## 关键公式

### 公式 1: [[轨迹级别信用混叠|传统 GRPO 轨迹级别优势]]

$$
\widehat{A}_{i,t} = \widehat{A}_i = \frac{R_i - \operatorname{mean}(\{R_j\}_{j=1}^G)}{\operatorname{std}(\{R_j\}_{j=1}^G) + \epsilon} \quad \forall t
$$

**含义**: 所有时刻共享同一标量优势——这正是导致[[轨迹级别信用混叠]]的根源。

**符号说明**:
- $R_i$: 轨迹 $i$ 的最终任务奖励（0 或 1）
- $G$: 同一任务的并行采样轨迹数（组大小）
- $\epsilon$: 数值稳定常数

---

### 公式 2: [[GRPO|原始 GRPO 优化目标]]

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|a_i|}\sum_{t=1}^{|a_i|} \min\!\left(r_{i,t}(\theta)\widehat{A}_i,\ \operatorname{clip}(r_{i,t}(\theta), 1-\varepsilon_L, 1+\varepsilon_H)\widehat{A}_i\right)\right]
$$

**含义**: 以 PPO-clip 形式约束策略更新步长，但优势信号仍为轨迹级别。

**符号说明**:
- $r_{i,t}(\theta) = \pi_\theta(a_{i,t}|s_{i,t}) / \pi_{\theta_{old}}(a_{i,t}|s_{i,t})$: 策略概率比
- $\varepsilon_L, \varepsilon_H$: 截断范围的下界和上界

---

### 公式 3: [[时序信用分配|阶段参与变量]]

$$
V_{i,k} = \begin{cases}1, & k=1 \text{ 或轨迹 } i \text{ 完成阶段 } m_{k-1}\\0, & \text{否则（不计入 }m_k\text{ 的统计）}\end{cases}
$$

**含义**: 只有实际进入某阶段的轨迹才参与该阶段的优势估计，避免将"尚未到达"与"到达后失败"混为一谈。

---

### 公式 4: [[时序信用分配|阶段结果与阶段条件优势]]

$$
R_{i,k} = \begin{cases}1, & \text{轨迹 } i \text{ 完成阶段 } m_k\\0, & \text{未完成}\end{cases}
$$

$$
\mu_k = \frac{\sum_{i=1}^G V_{i,k} R_{i,k}}{\sum_{i=1}^G V_{i,k}}, \quad \sigma_k = \sqrt{\frac{\sum_{i=1}^G V_{i,k}(R_{i,k}-\mu_k)^2}{\sum_{i=1}^G V_{i,k}}}
$$

$$
\widehat{A}_{i,k} = \frac{R_{i,k} - \mu_k}{\sigma_k + \epsilon}
$$

**含义**: 阶段 $k$ 的优势仅在参与该阶段（$V_{i,k}=1$）的轨迹内归一化计算，实现精确的组内相对比较。

**符号说明**:
- $\mu_k, \sigma_k$: 阶段 $k$ 内参与轨迹的成功率均值与标准差

---

### 公式 5: [[阶段条件优势|时序优势分配]]

$$
\widehat{A}_{i,t} = \sum_{k=1}^{K} \mathbb{I}[t \in B_{i,k}]\, \widehat{A}_{i,k}
$$

**含义**: 时刻 $t$ 的优势由其所属阶段的阶段优势决定，构建出随时间变化的分段优势序列——这是对传统 GRPO 均一优势的核心突破。

**符号说明**:
- $B_{i,k}$: 轨迹 $i$ 中产生阶段 $k$ 结果的动作区间
- $K$: 任务阶段总数

---

### 公式 6: [[阶段条件优势|Temporal GRPO 优化目标]]

$$
\mathcal{J}_{Temporal\text{-}GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|a_i|}\sum_{t=1}^{|a_i|} \min\!\left(r_{i,t}(\theta)\widehat{A}_{i,t},\ \operatorname{clip}(r_{i,t}(\theta), 1-\varepsilon_L, 1+\varepsilon_H)\widehat{A}_{i,t}\right)\right]
$$

**含义**: 与标准 GRPO 目标形式完全相同，唯一区别在于用时序变化的 $\widehat{A}_{i,t}$ 替换了固定的 $\widehat{A}_i$，从而无需修改 VLA 架构即可注入阶段级别信用。

---

### 公式 7: [[时序信用分配|阶段完成时刻与动作区间]]

$$
T_{i,k} = \min\{t \mid m_k \text{ 在轨迹 } i \text{ 中稳定完成}\}, \quad T_{i,0} = 0
$$

$$
B_{i,k}^{\text{成功}} = (T_{i,k-1},\, T_{i,k}], \quad B_{i,k}^{\text{失败}} = (T_{i,k-1},\, T_i]
$$

**含义**: 成功完成的阶段取完成前的动作区间；失败阶段则将轨迹剩余所有动作归入该区间，接收对应的负优势。

---

### 公式 8: [[时序信用分配|阶段变化量评估指标]]

$$
\Delta p_k = 100\!\left[\Pr_{\tau \sim \pi_{\theta^+}}(R_k(\tau)=1 \mid V_k(\tau)=1) - \Pr_{\tau \sim \pi_\theta}(R_k(\tau)=1 \mid V_k(\tau)=1)\right]
$$

**含义**: 比较 Temporal GRPO 训练前后，同一阶段 $k$ 在参与轨迹中的条件完成率变化，用于验证"信用集中在首个发散阶段"的假设。

---

## 关键图表

### Figure 1: Temporal GRPO 方法总览

![Figure 1: Temporal GRPO 总览](https://arxiv.org/html/2608.13026v1/Main.png)

**说明**: 左侧为轨迹级别信用混叠示意——两条失败轨迹（分别在阶段 2 和阶段 3 失败）被分配相同的负优势；右侧为 Temporal GRPO 的[[阶段条件优势]]分配——各阶段优势专属其对应动作区间，信用精准定位到首个失败阶段。

---

### Figure 2: 样本效率分析（RoboTwin 2.0）

**说明**: Temporal GRPO 与 Trajectory-GRPO 在 RoboTwin 2.0 所有任务和长/超长任务上的训练曲线（3 个独立种子均值）。Temporal GRPO 在长时序任务上的提升尤为显著（约 6.5 个百分点），且收敛速度更快，验证了阶段级信用对长时序操作的特殊价值。

---

### Figure 3: 阶段级信用分配控制实验（LIBERO-Long）

**说明**: 以"首个发散阶段 $m_d$"对齐 LIBERO-Long 任务，对比 Temporal GRPO 与 Trajectory-GRPO 在各阶段的 $\Delta p_k$ 变化。Temporal GRPO 的改善集中在 $m_d$ 阶段（首个发散点），前置阶段接近零变化；Trajectory-GRPO 则在前置阶段出现负变化，印证了[[轨迹级别信用混叠]]的破坏性。

---

### Table 1: RoboTwin 2.0 任务成功率（%）

| 方法 | 短时序 | 中时序 | 长/超长 | 宏平均 |
|------|--------|--------|---------|--------|
| π₀ | 45.5 | 58.8 | 43.3 | 49.2 |
| RDT-1B | 24.5 | 47.8 | 27.8 | 33.3 |
| OpenVLA-OFT (SFT) | 21.3 | 47.1 | 46.5 | 38.3 |
| Trajectory-GRPO | 37.8±1.8 | 52.6±1.6 | 48.7±1.9 | 46.4±1.3 |
| TGRPO | 43.9±1.6 | 58.4±1.5 | 54.1±1.7 | 52.1±1.1 |
| Stage-Reward GRPO | 52.7±1.4 | 64.2±1.3 | 60.8±1.5 | 59.2±1.0 |
| SimpleVLA-RL | 64.9±1.2 | 72.5±1.0 | 69.0±1.3 | 68.8±0.9 |
| **Temporal GRPO** | **73.2±0.9** | **79.0±0.8** | **75.2±1.1** | **75.8±0.7** |

**关键发现**: Temporal GRPO 宏平均 75.8%，超过 SimpleVLA-RL（68.8%）7.0 个百分点，在所有时序长度上一致领先 6.2–8.3 点，长时序任务优势最明显。

---

### Table 2: LIBERO-Long 消融实验（%）

| 变体 | 成功率 |
|------|--------|
| **Temporal GRPO（完整）** | **99.1±0.4** |
| w/o Stage Compiler | 96.8±0.7 |
| Stage-Reward GRPO | 94.7±0.9 |
| w/o entered-stage gating | 92.5±1.1 |
| w/o same-stage grouping | 90.6±1.3 |
| Trajectory-GRPO | 88.4±1.5 |

**关键发现**: 每个组件缺失均带来显著下降。其中 same-stage grouping（同阶段分组）和 entered-stage gating（已进入阶段门控）是最核心的两个设计，合计贡献约 8.5 个百分点提升。

---

## 实验

### 数据集

| 数据集 | 特点 | 用途 |
|--------|------|------|
| [[RoboTwin 2.0]] | 涵盖短/中/长/超长时序操作任务，多样化机器人操作场景 | 主要评测（成功率 + 样本效率） |
| [[LIBERO]] | 阶段依赖关系清晰，适合受控分析 | 消融实验 + 阶段级信用机制验证 |

### 实现细节

- **基础策略**: [[OpenVLA-OFT]] SFT 检查点（公开发布版本），作为各方法的统一初始化
- **阶段生成模型**: 冻结 [[RynnBrain-4B]]（仅推断使用，不参与 RL 训练）
- **对比控制**: Trajectory-GRPO、Stage-Reward GRPO、Temporal GRPO 共享相同探索和优化超参数，仅信用分配规则不同
- **评估**: 每组实验 3 个独立随机种子，报告均值 ± 标准差

### 可视化结果

LIBERO-Long 受控实验从"首个发散阶段"维度对齐多条任务，直观展示：Temporal GRPO 的改善集中在首个发散阶段（$\Delta p_{m_d}$ 显著正），前置阶段几乎不受影响；Trajectory-GRPO 则前置阶段出现退化（负 $\Delta p_k$），直接证明了[[轨迹级别信用混叠]]的存在及其危害。

---

## 批判性思考

### 优点

1. **问题定义精准**: [[轨迹级别信用混叠]]的概念清晰、实验验证充分（LIBERO-Long 的 $\Delta p_k$ 分析是有力的消融证据）。
2. **架构无侵入**: 不修改 VLA 模型结构，不引入额外价值网络，仅通过重构优势信号即可提升性能，工程友好性高。
3. **一致性提升**: 在 RoboTwin 2.0 所有时序长度上均有 6–8 个百分点的稳定提升，不存在只在特定场景生效的局限。

### 局限性

1. **阶段检测依赖特权状态**: 训练时依赖仿真器特权状态检测阶段完成，现实部署中无法直接使用；真实世界部署需要额外的状态估计或视觉阶段检测模块。
2. **线性阶段顺序假设**: Stage Compiler 假设阶段为有向线性链（无分支、无循环），对含条件分支、错误恢复或并行子任务的复杂操作场景适用性受限。
3. **RynnBrain-4B 依赖**: 阶段生成依赖特定模型，其泛化能力未经独立评估；若任务指令语义模糊则阶段质量下降。

### 潜在改进方向

1. **不确定性感知阶段检测**: 引入不确定性估计的阶段检测器，适应边界模糊的任务。
2. **动态阶段图**: 将线性阶段链扩展为有向无环图（DAG），支持分支、重试和并行阶段。
3. **视觉阶段检测**: 训练纯视觉的阶段检测器，消除对特权仿真器状态的依赖，打通 Sim2Real 路径。

### 可复现性评估

- [ ] 代码开源（论文未提供 GitHub 链接）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（超参数共享设计使对比实验可信）
- [x] 数据集可获取（RoboTwin 2.0、LIBERO 均为公开基准）

---

## 关联笔记

### 基于

- [[GRPO]]: 核心优化算法，Temporal GRPO 对其优势估计机制进行了阶段级重构
- [[OpenVLA-OFT]]: 所有实验的基础策略初始化
- [[RynnBrain-4B]]: 冻结模型，用于语义阶段生成

### 对比

- [[SimpleVLA-RL]]: 最强 baseline，被超越 7.0 个百分点（68.8% → 75.8%）
- [[Shared-Prefix GRPO]]: 同为 GRPO 变体，专注于共享前缀的高效采样
- [[DyGRO-VLA]]: 另一个解决多任务 VLA-RL 问题的同期工作（方向不同，DyGRO 关注跨任务遗忘）

### 方法相关

- [[轨迹级别信用混叠]]: 本文定义的核心问题
- [[阶段条件优势]]: 本文提出的核心解决机制
- [[时序信用分配]]: 广义技术类别，本文的具体实例化
- [[Action Chunking]]: VLA 策略的动作输出形式

### 硬件/数据相关

- [[RoboTwin 2.0]]: 主评测基准
- [[LIBERO]]: 受控消融实验基准

---

## 速查卡片

> [!summary] Temporal GRPO (2026)
> - **核心**: 把 GRPO 的信用比较单元从"整条轨迹"细化到"同一任务阶段内的轨迹子集"，解决轨迹级别信用混叠
> - **方法**: RynnBrain-4B 生成语义阶段 → Stage Compiler 编译 → 特权状态对齐 → 阶段条件优势分段分配
> - **结果**: RoboTwin 2.0 宏平均 75.8%（+7.0% vs SimpleVLA-RL），LIBERO-Long 99.1%
> - **代码**: 未公开

---

*笔记创建时间: 2026-08-15*
