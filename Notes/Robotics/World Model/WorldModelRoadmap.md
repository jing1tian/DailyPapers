---
title: "A Definition and Roadmap for World Models"
method_name: "WorldModelRoadmap"
authors: [Xinyuan Chen, Haoyu Guo, Shi Guo, Bingqi Jiang, Chunhua Shen, Xing Shen, Tianfan Xue, Yufei Xue, Mulin Yu, Weinan Zhang, Bin Zhao, Bowen Zhou, Ming Zhou]
year: 2026
venue: arXiv
tags: [world-model, world-action-model, model-based-rl, embodied-ai, survey]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.06401
created: 2026-07-09
---

# 论文笔记：A Definition and Roadmap for World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Shanghai AI Laboratory（Physical Intelligence Team） |
| 日期 | July 2026 |
| 项目主页 | 无 |
| 对比基线 | 无（综述/视角论文） |
| 链接 | [arXiv](https://arxiv.org/abs/2607.06401) |

---

## 一句话总结

> 首篇对世界模型给出科学定义的 perspective 论文，提出基于信息压缩的统一框架，并描绘从多模态统一到物理 AGI 的三阶段路线图。

---

## 核心贡献

1. **科学定义**: 将[[World Model|世界模型]]形式化为"在有限计算资源约束下，对物理世界状态转移过程的压缩建模"，明确三大核心属性
2. **二维分类体系**: 从功能（Renderer/Simulator/Planner）× 架构（观测级/潜空间/3D）两个维度构建系统分类法
3. **三阶段路线图**: 近期多模态统一 → 中期物理表示统一 → 远期基础级交互仿真器，指向 Physical AGI 的 Trinity Architecture

---

## 问题背景

### 要解决的问题

尽管"世界模型"在 AI 研究中被广泛讨论，各子领域对其定义、设计原则和评估标准仍缺乏共识，导致概念混乱、难以横向对比。

### 现有方法的局限

- 不同社区对世界模型的理解差异巨大：生成式视频模型强调视觉保真度，[[MBRL|model-based RL]] 侧重高效规划，具身 AI 侧重物理交互
- 缺乏统一的评估框架和 benchmark
- 数据不对称：网络视频丰富但缺乏细粒度操作数据，机器人专有数据规模不足

### 本文的动机

以信息论视角重新框架：世界模型的核心是**从高维感知数据中提炼隐含物理知识的压缩问题**，而非生成或仿真本身。数据多样性决定系统泛化的上限（Data Ceiling Principle）。

---

## 方法详解

### 世界模型核心定义

[[World Model|世界模型]]被定义为在有限计算资源约束下，对物理世界**状态转移过程**的压缩建模，具备三大属性：

- **全模态工作域（Omnimodal Workscope）**: 必须建模所有感知模态的数据
- **多维异步性（Multidimensional Asynchronicity）**: 处理多频率、异步的序列数据
- **局部性（Locality）**: 基于有界局部感知运作（[[POMDP]] 框架）

### 两种哲学视角

| 视角 | 重点 | 代表系统 |
|------|------|----------|
| **理解导向** | 学习潜在结构和因果关系 | [[JEPA]], [[Dreamer]] |
| **预测导向** | 优化前向仿真 | [[Sora]], [[Cosmos3\|Cosmos]] |

本文立场：**理解是主体，预测服务于理解**。

### 功能分类

在 [[POMDP]] 循环中，[[World Model|世界模型]]承担三种角色：

- **Renderer（渲染器）**: 条件化生成观测（给定场景描述 → 像素/感知）
- **Simulator（仿真器）**: 建模干预下的潜在状态演化
- **Planner（规划器）**: 评估反事实未来并选择动作

### 架构范式（四类）

| 范式 | 代表系统 | 优势 | 局限 |
|------|----------|------|------|
| **观测级生成模型** | [[Sora]], [[Cosmos3\|Cosmos]], Genie, Diffusion Forcing | 高视觉保真度；直观输出；可利用网络视频扩展 | 生成代价高；物理一致性差；长程漂移 |
| **潜空间模型** | [[DreamerV3]], [[JEPA]], PlaNet, [[MBRL\|TD-MPC2]], MuZero | 高效想象规划；紧凑表示；长程扩展 | 丢失关键细节；难以视觉检查；目标错配 |
| **3D/物体中心模型** | OccWorld, [[NeRF]], [[3D Gaussian Splatting\|3DGS]], SlotFormer | 空间一致性强；物体级推理；遮挡处理 | 需要结构化监督；动态场景困难 |
| **跨范式统一（全模态）** | LWM, HERMES, [[World Action Model\|DreamZero]], UWM, Cosmos3 | 融合生成/理解/交互/多模态 | 早期阶段；模态对齐问题 |

### World Action Model (WAM)

[[World Action Model]]（WAM）将动态预测与动作生成**耦合在单一生成过程内**，是从纯仿真迈向具身决策的关键桥接。区别于传统 [[MBRL]] 的"先学模型再学策略"范式，WAM 在世界模型内部学习策略。

### Chain-of-Imagination (CoI)

[[Chain-of-Imagination]]（CoI）将推理视为在**潜空间中结构化的动作-后果对序列**，而非外部规划树。通过多分支想象轨迹实现深思熟虑的推理深度与假设多样性的平衡。

---

## 关键公式

### 公式1: [[POMDP|POMDP 信念更新]]

$$
\mathbf{x}_{t}(\mathbf{s}_{t}) \propto O(\mathbf{o}_{t} \mid \mathbf{s}_{t}) \sum_{s_{t-1}} T(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}, \mathbf{a}_{t-1}) \mathbf{x}_{t-1}(\mathbf{s}_{t-1})
$$

**含义**: 给定新观测 $\mathbf{o}_t$，利用观测似然和状态转移更新后验信念分布。

**符号说明**:
- $\mathbf{x}_{t}(\mathbf{s}_{t})$: 时刻 $t$ 对状态 $\mathbf{s}_t$ 的后验信念
- $O(\mathbf{o}_{t} \mid \mathbf{s}_{t})$: 观测似然（观测模型）
- $T(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}, \mathbf{a}_{t-1})$: 状态转移模型
- $\mathbf{x}_{t-1}(\mathbf{s}_{t-1})$: 上一时刻信念

### 公式2: [[Scaling Law|神经扩展律]]

$$
\mathcal{L}(N, D) \approx \mathcal{L}_{\infty} + AN^{-\alpha} + BD^{-\beta}, \quad C \approx \kappa ND
$$

**含义**: 损失随模型参数量 $N$、数据量 $D$ 呈幂律下降；$C$ 为计算量。

**符号说明**:
- $N$: 模型参数量
- $D$: 训练数据量
- $\mathcal{L}_{\infty}$: 最优不可减少损失（数据分布熵下界）
- $\alpha, \beta$: 幂律指数
- $\kappa$: 计算效率系数
- $C$: 总计算量（FLOPs）

### 公式3: [[MBRL|模型基强化学习目标]]

$$
J(\pi_{\phi}) = \mathbb{E}_{\pi_{\phi}, P^{\star}} \left[ \sum_{t=0}^{H-1} \gamma^{t} r_{t} \right], \quad r_{t} = R(\mathbf{s}_{t}, \mathbf{a}_{t})
$$

**含义**: 策略 $\pi_\phi$ 在真实环境 $P^\star$ 下的期望折扣累积回报。

**符号说明**:
- $\pi_\phi$: 参数化策略
- $P^\star$: 真实环境转移分布
- $H$: 规划水平（horizon）
- $\gamma$: 折扣因子
- $R(\mathbf{s}_t, \mathbf{a}_t)$: 即时奖励函数

### 公式4: [[MBRL|学习转移模型]]

$$
\widehat{P}_{\theta}: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S}), \quad \widehat{\mathbf{s}}_{t+1} \sim \widehat{P}_{\theta}(\cdot \mid \mathbf{s}_{t}, \mathbf{a}_{t})
$$

**含义**: 学得的转移函数将当前状态-动作对映射为下一状态的概率分布。

**符号说明**:
- $\widehat{P}_\theta$: 参数化的学习转移模型
- $\Delta(\mathcal{S})$: 状态空间上的概率分布集合

### 公式5: [[MBRL|动力学模型损失]]

$$
\mathcal{L}_{dyn}(\theta) = -\mathbb{E}_{(\mathbf{s}, \mathbf{a}, \mathbf{s}') \sim \mathcal{D}} \left[ \log \widehat{P}_{\theta}(\mathbf{s}' \mid \mathbf{s}, \mathbf{a}) \right]
$$

**含义**: 在回放缓冲区 $\mathcal{D}$ 上最小化转移模型的负对数似然。

**符号说明**:
- $\mathcal{D}$: 经验回放缓冲区
- $(\mathbf{s}, \mathbf{a}, \mathbf{s}')$: 状态-动作-下一状态转移元组

### 公式6: [[Chain-of-Imagination|CoI 信念编码]]

$$
\mathbf{z}_{t} = E_{\eta}(\mathbf{o}_{\leq t}, \mathbf{a}_{< t})
$$

**含义**: 将历史观测和动作序列压缩为当前潜在信念状态 $\mathbf{z}_t$。

**符号说明**:
- $E_\eta$: 编码器网络
- $\mathbf{o}_{\leq t}$: 至今所有观测
- $\mathbf{a}_{< t}$: 至今所有动作
- $\mathbf{z}_t$: 潜在信念状态

### 公式7: [[Chain-of-Imagination|CoI 想象轨迹集合]]

$$
\mathcal{C}_{t} = \left\{ \left[ (\mathbf{a}_{b,1}, \widehat{\mathbf{z}}_{b,1|t}), \ldots, (\mathbf{a}_{b,K_b}, \widehat{\mathbf{z}}_{b,K_b|t}) \right] \right\}_{b=1}^{B_t}
$$

**含义**: $B_t$ 条想象分支的集合，每条分支包含 $K_b$ 步动作-状态对，构成推理的"想象链"。

**符号说明**:
- $B_t$: 并行想象分支数
- $K_b$: 第 $b$ 条分支的展开深度
- $\widehat{\mathbf{z}}_{b,k|t}$: 第 $b$ 条分支第 $k$ 步的预测潜在状态

### 公式8: [[Chain-of-Imagination|CoI 潜状态演化]]

$$
\widehat{\mathbf{z}}_{b,k|t} = W_{\theta}(\widehat{\mathbf{z}}_{b,k-1|t}, \mathbf{a}_{b,k}), \quad \widehat{\mathbf{z}}_{b,0|t} = \mathbf{z}_{t}
$$

**含义**: 世界模型 $W_\theta$ 在动作条件下递推演化潜在状态，以当前信念为起点。

### 公式9: [[Chain-of-Imagination|CoI 策略决策]]

$$
\widehat{\mathbf{a}}_{t} = \pi_{\phi}\left(\mathbf{z}_{t}, \operatorname{Agg}_{\psi}(\mathcal{C}_{t})\right)
$$

**含义**: 策略同时以当前信念状态和聚合后的想象轨迹为条件进行决策。

**符号说明**:
- $\operatorname{Agg}_\psi$: 轨迹聚合函数（可学习）
- $\mathcal{C}_t$: 当前时刻的完整想象轨迹集合

### 公式10: [[World Model|视频预测目标]]

$$
p_{\theta}(\mathbf{x}_{t+1:t+K} \mid \mathbf{x}_{\leq t}, \mathbf{a}_{t:t+K-1}, \mathbf{c})
$$

**含义**: 给定历史观测、动作序列和可选条件 $\mathbf{c}$，预测未来 $K$ 帧的条件概率。

**符号说明**:
- $\mathbf{x}_{t+1:t+K}$: 未来 $K$ 帧（视频帧/感知数据）
- $\mathbf{c}$: 可选条件（语言、任务描述等）

### 公式11: [[Next-Token Prediction|NTP 自回归损失]]

$$
\mathcal{L}_{NTP} = -\sum_{t=1}^{T} \log p_{\theta}(u_{t} \mid u_{<t}, \mathbf{a}_{\leq t}, \mathbf{c})
$$

**含义**: 在动作和条件的情境下，最大化序列中每个 token 的自回归预测概率。

**符号说明**:
- $u_t$: 时刻 $t$ 的 token（观测/动作/语言）
- $u_{<t}$: 所有历史 token

---

## 关键图表

### Figure 1: 数据金字塔倒置漏斗

![Figure 1](https://arxiv.org/html/2607.06401v1/figs/data4.png)

**说明**: 从海量非结构化公开媒体（网络视频、文本）经合成/筛选，最终压缩为任务优化的机器人训练数据的三级漏斗流水线，对应 Data Ceiling Principle 的数据来源层级。

### Figure 2: 世界模型的本质与属性

![Figure 2](https://arxiv.org/html/2607.06401v1/figs/WM-definition2.png)

**说明**: 可视化[[World Model|世界模型]]的三大核心属性（全模态工作域、多维异步性、局部性）及其相互关系。

### Figure 3: POMDP 智能体-环境循环

![Figure 3](https://arxiv.org/html/2607.06401v1/figs/pomdp.png)

**说明**: [[POMDP]] 框架下智能体与环境的交互示意，定义了信念状态 $\mathbf{x}_t$、观测 $\mathbf{o}_t$、动作 $\mathbf{a}_t$、转移 $T$ 和观测模型 $O$ 的符号系统。

### Figure 4: POMDP 循环中的功能角色

![Figure 4](https://arxiv.org/html/2607.06401v1/figs/functional_roles3.png)

**说明**: 展示 Renderer、Simulator、Planner 三类功能角色在 [[POMDP]] 循环中的位置和职责划分。

### Figure 5: 二维分类法（功能 × 架构）

![Figure 5](https://arxiv.org/html/2607.06401v1/figs/Taxonomy3.png)

**说明**: 以功能（Renderer/Simulator/Planner）为横轴、架构范式（观测级/潜空间/3D-中心/全模态统一）为纵轴的完整分类图谱，标注了 50+ 代表系统的位置。

### Figure 6: 架构统一趋势

![Figure 6](https://arxiv.org/html/2607.06401v1/figs/omnimodal.png)

**说明**: 展示生成-理解统一模型逐步引入交互和动作接地，演化为[[World Action Model]]（WAM）的历史趋势线。

### Figure 7: 训练与学习范式总览

![Figure 7](https://arxiv.org/html/2607.06401v1/x1.png)

**说明**: 覆盖自监督预训练（视频预测、MAE、NTP）、[[MBRL]]、策略内嵌学习、[[Chain-of-Imagination|CoI]]、物理约束学习、反事实推理、长程层级规划的主要训练范式概览。

### Table 1: 架构范式优劣对比

| **范式** | **优势** | **局限** | **代表系统** |
|----------|----------|----------|--------------|
| **观测级生成** | 高视觉保真度；直观输出；互联网视频可扩展 | 生成代价高；物理不一致；长程漂移 | Sora, Genie, Wan, Seedance, Cosmos, Diffusion Forcing |
| **潜空间** | 高效想象规划；紧凑潜状态；长程扩展 | 丢失关键细节；难以视觉检查；目标错配 | PlaNet, Dreamer, DreamerV2/V3, TD-MPC2, MuZero, I-JEPA, V-JEPA |
| **3D/物体中心** | 空间一致性强；物体级推理；遮挡处理好 | 需要结构化监督；动态场景困难 | OccWorld, NeRF, 3D Gaussians, Marble, SlotFormer |
| **跨范式统一** | 融合生成、理解、交互、多模态接地 | 早期阶段；模态对齐；动作保真度；评估开放 | LWM, WorldGPT, HERMES, DreamZero, UWM, Cosmos3 |

---

## 三阶段路线图

### 近期：统一多模态世界模型

在共享架构中整合语言、视觉、音频和动作，支持生成、理解和交互。核心挑战是跨模态时间对齐和全模态数据的高效获取。

### 中期：统一物理表示

开发跨多种具身形态通用的共享状态空间，桥接占据图、几何、物体和动力学。目标是 embodiment-agnostic 的物理先验。

### 远期：基础级交互仿真器

创建大规模、持续学习的仿真器，支持自主数据收集、自改善循环和跨领域迁移，实现闭环具身智能。

### Physical AGI 的 Trinity Architecture

通向物理 AGI 的框架由三部分组成：

1. **世界模型**（World Model）: 状态理解与预测
2. **价值/目标模型**（Value/Objective Model）: 目标表示
3. **动作/策略模型**（Action/Policy Model）: 控制执行

三者形成闭环，智能体通过世界建模自主改善，实现基于理解而非纯行为模仿的 Physical AGI。

---

## 开放挑战

1. **数据不对称**: 网络视频丰富但缺乏精细操作；机器人专有数据规模/多样性不足
2. **保真度 vs 精度 Trade-off**: 语义抽象与物理细节精度之间的权衡
3. **复合预测误差**: 自回归展开的长程不稳定性
4. **Sim-to-Real 迁移**: 模型偏差和部署时的分布偏移
5. **评估与 Benchmark 缺失**: 缺乏超越似然指标的统一评估标准
6. **安全、透明度与可持续性**: 安全探索、可验证控制、隐私保护治理

---

## 批判性思考

### 优点

1. **定义清晰**: 首次从信息论角度给出可操作的形式化定义，而非直觉描述
2. **分类体系完整**: 二维分类法覆盖 50+ 系统，横跨多个研究社区
3. **路线图实用**: 三阶段划分有明确技术里程碑，不流于空泛

### 局限性

1. **观点性强**: 作为 perspective 论文，对"理解优于预测"等核心论断缺乏实证验证
2. **社会世界模型明确排除**: 对大规模社会预测的伦理限制合理，但排除了大量 LLM 应用场景
3. **评估缺失**: 提出了评估 benchmark 的需求但未提供具体解决方案

### 可复现性评估

- [ ] 代码开源（理论/视角论文，不适用）
- [ ] 预训练模型（不适用）
- [x] 框架定义完整
- [x] 路线图可追踪

---

## 关联笔记

### 基于

- [[World Action Model]]: WAM 是本文提出的核心演化方向
- [[MBRL]]: 模型基强化学习是训练范式的核心章节
- [[DreamerV3]]: 潜空间模型的代表系统
- [[JEPA]]: 理解导向架构的代表

### 对比

- [[WAMSurvey]]: 同期 WAM 专项综述，覆盖范围更窄但更深
- [[WMRobotSurvey]]: 机器人方向世界模型综述

### 方法相关

- [[Chain-of-Imagination]]: CoI 推理框架
- [[POMDP]]: 形式化框架基础
- [[Scaling Law]]: 扩展律分析

### 硬件/数据相关

- [[Cosmos3]]: 代表性全模态世界模型系统

---

## 速查卡片

> [!summary] A Definition and Roadmap for World Models
> - **核心**: 世界模型 = 对状态转移过程的信息压缩，数据天花板决定泛化上限
> - **框架**: 三属性定义 + 功能三分类（Renderer/Simulator/Planner）+ 四类架构范式
> - **路线图**: 多模态统一 → 物理表示统一 → 基础仿真器 → Trinity Architecture (Physical AGI)
> - **代码**: 无（理论论文）

---

*笔记创建时间: 2026-07-09*
