---
title: "Learning Bilevel Policies over Symbolic World Models for Long-Horizon Planning"
method_name: "BISON"
authors: [Dillon Z. Chen, Till Hofmann, Toryn Q. Klassen, Sheila A. McIlraith]
year: 2026
venue: arXiv
tags: [bilevel-planning, symbolic-planning, imitation-learning, long-horizon-planning, graph-neural-network, task-and-motion-planning, policy-learning]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2605.15975v2
created: 2026-05-21
---

# 论文笔记：Learning Bilevel Policies over Symbolic World Models for Long-Horizon Planning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Toronto |
| 日期 | May 2026 |
| 项目主页 | [dillonzchen.github.io/bison](https://dillonzchen.github.io/bison) |
| 对比基线 | [[SmolVLA]], DetPlan, NdtPlan, DetReplan, NdtReplan, PureNN, PddlNN |
| 链接 | [arXiv](https://arxiv.org/abs/2605.15975) |

---

## 一句话总结

> BISON 将高层[[符号规划]]（first-order condition-action rules）与低层[[图神经网络]]策略结合，通过[[模仿学习]]从演示中学习双层策略，无需执行时搜索即可泛化到比训练时更多物体数量和更长规划地平线的任务。

---

## 核心贡献

1. **双层策略联合学习框架**: 提出从低层演示中同时学习高层符号策略（HL policy）和低层神经网络策略（LL policy），两者通过[[标注函数]]（labeling function $\mathcal{L}$）耦合，无需执行时规划搜索。
2. **符号策略的归纳泛化**: 利用[[目标回归]]（goal regression）从抽象轨迹中提取 condition-action 规则，再通过[[变量提升]]（lifting）替换具体对象为变量，实现对任意数量物体的零样本泛化。
3. **轻量级 GNN 低层策略**: 设计仅含 33,000 个参数的[[图神经网络]]低层策略，以对象为中心的表示支持开放世界场景，编码高层动作指导生成连续控制输出。
4. **非确定性下行精化性质（NDRP）理论保障**: 证明满足 NDRP 时双层策略的一致性，并在有限训练数据条件下证明 HL 策略可解决任意多物体问题的理论定理。

---

## 问题背景

### 要解决的问题

具身 AI 代理在执行长时域（long-horizon）任务时面临挑战：任务需要数十步乃至数百步操作，物体数量可变，且环境中存在外生不确定性（exogenous uncertainty）和内生不确定性（endogenous uncertainty）。

### 现有方法的局限

- **端到端神经网络**（如 [[VLA]]、PureNN）：不具备组合泛化能力，对更多物体或更长规划地平线失败，成功率为 0%。
- **经典符号规划**（DetPlan、NdtPlan）：需要精确的领域知识，在有不确定性的环境中失败，且每步都需重新规划（replanning），计算开销大。
- **重规划方法**（DetReplan、NdtReplan）：在每次执行失败时重新规划，推理时间开销高，且仍依赖精确模型假设。
- **现有双层规划（TAMP）**：需要执行时符号搜索，扩展性差；且对外生不确定性（如环境物体突然移动）缺乏鲁棒性。

### 本文的动机

通过从演示中**学习**双层策略，而非在执行时**搜索**，可以将符号规划的可解释性、可泛化性与神经网络的鲁棒性结合。[[模仿学习]]提供了学习信号，[[符号抽象]]提供了结构化归纳偏置，使策略既高效又可泛化。

---

## 方法详解

### 问题形式化

**低层规划问题**：$\mathbf{P}^{ll} = \langle \mathbf{S}, \mathbf{A}, \mathbf{s}_0, \mathbf{g} \rangle$

**双层规划问题**：$\mathbb{P} = \langle \mathbf{P}^{ll}, \mathbf{P}^{hl}, \mathcal{L} \rangle$，其中 $\mathcal{L}$ 为[[标注函数]]，将低层观测映射到高层抽象状态。

**低层状态表示**（以对象为中心）：

$$
\mathbf{s}^{ll} = \langle \mathbf{x}_e, \{x_o\}_{o \in \mathbf{O}} \rangle
$$

其中 $\mathbf{x}_e \in \mathbb{R}^n$ 为 agent 状态，$x_o \in \mathbb{R}^m$ 为各对象状态，$\mathbf{O}$ 为对象集合。

**演示数据集**：$\mathbb{T} = \{\langle g_i^{hl}, \mathbf{s}_0^{ll}, \mathbf{a}_0^{ll}, \ldots, \mathbf{s}_m^{ll}, \mathbf{a}_m^{ll} \rangle\}_{i=1}^n$

### 模型架构

BISON 采用**双层策略**架构：
- **输入**: 低层演示 + 高层目标 $g^{hl}$，以及领域理论 $\mathcal{D} = \langle \mathcal{P}, \mathcal{A} \rangle$ 和[[标注函数]] $\mathcal{L}$
- **高层策略（HL Policy）**: [[First-Order Logic|一阶]] condition-action 规则集，通过符号搜索（前向链）执行
- **低层策略（LL Policy）**: [[图神经网络]]（GNN），约 33,000 参数，以对象为中心
- **输出**: 低层连续动作 $\mathbf{a}^{ll}$
- **执行方式**: HL 策略提出高层动作 $a^{hl}$，LL 策略以此为条件生成低层动作

![BISON Pipeline](https://dillonzchen.github.io/bison/assets/images/pipeline.png)

**说明**: BISON 的整体框架。左侧为学习阶段：低层演示通过 $\mathcal{L}$ 诱导高层演示，分别训练 HL 和 LL 策略。右侧为执行阶段：通过 $\mathcal{L}(s^{ll})$ 计算符号状态，HL 策略提出动作，LL 策略执行。

### 核心模块

#### 模块 1: 高层策略学习（HL Policy Learning）

**设计动机**: 利用[[符号规划]]和[[目标回归]]从演示中自动归纳可泛化的规则，无需人工标注规则。

**三步流程**:

1. **HL 轨迹构建（Trace Construction）**: 将 $\mathcal{L}$ 应用于低层演示，提取高层动作序列：
   $$T^{hl} = \{\langle g_i^{hl}, a_0^{hl}, \ldots, a_m^{hl} \rangle\}_{i=1}^n$$

2. **规则提取（Rule Extraction）**: 用[[目标回归]]从 HL 轨迹和目标中反向推导 condition-action 规则。对于每个动作 $a^{hl}$，后继状态由下式计算：
   $$\text{succ}(s^{hl}, a^{hl}) = \{(s^{hl} \setminus \text{del}_i(a^{hl})) \cup \text{add}_i(a^{hl})\}_{i=1}^n$$

3. **归纳泛化（Inductive Generalization）**: 通过 $\text{lift}(a^{hl}, s^{hl}, g^{hl}) = \langle \text{var}, \text{sCond}, \text{gCond}, \text{action} \rangle$ 将具体对象替换为变量，生成一阶规则。

最终 HL 策略为规则集 $\{\langle \text{val}(r), \text{var}(r), \text{sCond}(r), \text{gCond}(r), \text{action}(r) \rangle\}$。

![HL Policy Learning](https://dillonzchen.github.io/bison/assets/images/hl_policy.png)

**说明**: HL 策略学习的三步流程。Step 1 从低层演示构建 HL 轨迹，Step 2 用目标回归提取 ground 规则（下划线），Step 3 用变量替换对象生成一阶符号策略。

#### 模块 2: 低层策略学习（LL Policy Learning）

**设计动机**: 使用[[图神经网络]]处理变长对象集合，以 HL 动作为条件，实现对任意物体数量的零样本泛化。

**GNN 架构**（3 类节点：global, action args, other objects）:

**初始嵌入**:
$$\mathbf{h}^{global} = \mathbf{x}_e \| \sum_{p() \in s^{hl}} \mathbf{e}_p^{|\mathcal{P}|} \| \sum_{p() \in g^{hl}} \mathbf{e}_p^{|\mathcal{P}|}$$

$$\mathbf{h}^{o_i} = \mathbf{x}_{o_i} \| \sum_{p(o_i) \in s^{hl}} \mathbf{e}_p^{|\mathcal{P}|} \| \sum_{p(o_i) \in g^{hl}} \mathbf{e}_p^{|\mathcal{P}|} \| \mathbf{e}_i^M$$

其中 $\mathbf{e}_p^{|\mathcal{P}|}$ 为谓词 one-hot 编码，$M$ 为最大 action arity，$\mathbf{e}_i^M$ 标记对象在动作参数中的位置。

**消息传递**（第 $l$ 层）:

对象聚合（element-wise max pooling）：
$$\mathbf{h}_o^{(l)} = \max_{i=1,\ldots,n} \mathbf{h}_{o_i}^{(l)}$$

Global 节点更新：
$$\mathbf{h}^{global(l+1)} = \sigma(\mathbf{W}_m^{(l)}(\mathbf{h}^{global(l)} + \mathbf{h}_a^{(l)} + \mathbf{h}_o^{(l)}))$$

对象节点更新：
$$\mathbf{h}_{o_i}^{(l+1)} = \sigma(\mathbf{W}_o^{(l)}(\mathbf{h}^{global(l+1)} + \mathbf{h}_a^{(l)} + \mathbf{h}_{o_i}^{(l)}))$$

![LL Policy GNN](https://dillonzchen.github.io/bison/assets/images/ll_policy.png)

**说明**: LL 策略的 GNN 结构。输入动作为 $a^{hl} = \text{pick}(\text{obj}, \text{loc})$，实线为图边，虚线为信息传递路径，粗体为欧式向量。

---

## 关键公式

### 公式 1: [[双层策略组合|双层策略边缘化]]

$$
\pi(\mathbf{a}^{ll} | \mathbf{s}^{ll}, g^{hl}) = \sum_{a^{hl}} \pi^{ll}(\mathbf{a}^{ll} | \mathbf{s}^{ll}, a^{hl}, g^{hl}) \cdot \pi^{hl}(a^{hl} | \mathcal{L}(\mathbf{s}^{ll}), g^{hl})
$$

**含义**: 完整的双层策略通过对高层动作边缘化，将 HL 策略和 LL 策略组合为一个统一策略。

**符号说明**:
- $\pi(\mathbf{a}^{ll} | \mathbf{s}^{ll}, g^{hl})$: 完整双层策略
- $\pi^{ll}$: 低层神经网络策略（GNN）
- $\pi^{hl}$: 高层符号规则策略
- $\mathcal{L}(\mathbf{s}^{ll})$: 将低层状态映射到高层符号状态
- $a^{hl}$: 高层动作（抽象操作）

### 公式 2: [[非确定性下行精化性质|NDRP（Nondeterministic Downward Refinement Property）]]

$$
\forall \mathbf{s} \in \text{exec}(\mathbf{s}^{ll}, \pi^{ll}),\ \mathcal{L}(\mathbf{s}) \in \text{exec}(\mathcal{L}(\mathbf{s}^{ll}), \pi^{hl}) \cup \{\mathcal{L}(\mathbf{s}^{ll})\}
$$

**含义**: 保证低层策略执行的每个状态，其高层抽象要么停留在当前 HL 状态，要么推进到合法的 HL 后继状态，确保双层策略的一致性。

**符号说明**:
- $\text{exec}(\mathbf{s}, \pi)$: 从状态 $\mathbf{s}$ 执行策略 $\pi$ 可达的状态集
- $\mathcal{L}(\mathbf{s}^{ll})$: 当前低层状态的高层抽象
- $\pi^{hl}$: 高层策略

### 公式 3: [[符号规划|高层动作后继状态]]

$$
\text{succ}(s^{hl}, a^{hl}) = \{(s^{hl} \setminus \text{del}_i(a^{hl})) \cup \text{add}_i(a^{hl})\}_{i=1}^n
$$

**含义**: 定义非确定性高层动作如何更新符号状态，通过删除旧事实、添加新事实实现状态转移。

**符号说明**:
- $s^{hl}$: 当前高层符号状态（命题集合）
- $\text{del}_i(a^{hl})$: 动作第 $i$ 个结果的删除效果
- $\text{add}_i(a^{hl})$: 动作第 $i$ 个结果的添加效果
- $n$: 非确定性结果数量

### 公式 4: [[目标回归]]

$$
\text{regr}(g^{hl}, a^{hl}) = \{(g^{hl} \setminus \text{add}_i(a^{hl})) \cup \text{pre}(a^{hl}) \mid i = 1, \ldots, n\}
$$

**含义**: 给定目标和动作，反向计算执行该动作之前需要满足的前提条件集合，用于从目标倒推出 condition-action 规则。

**符号说明**:
- $g^{hl}$: 高层目标（命题集合）
- $\text{add}_i(a^{hl})$: 动作第 $i$ 个结果的添加效果（已被目标满足，可以去除）
- $\text{pre}(a^{hl})$: 动作的前提条件

### 公式 5: [[变量提升|符号规则提升]]

$$
\text{lift}(a^{hl}, s^{hl}, g^{hl}) = \langle \text{var}, \text{sCond}, \text{gCond}, \text{action} \rangle
$$

**含义**: 将具体 ground 规则泛化为一阶规则，用变量替换具体对象，使规则可应用于训练时未见的对象组合。

**符号说明**:
- $\text{var}$: 规则中的变量集
- $\text{sCond}$: 状态条件（以变量表达）
- $\text{gCond}$: 目标条件（以变量表达）
- $\text{action}$: 动作模式（action schema）

---

## 关键图表

### Figure 1: 系统整体概览

![BISON Overview](https://dillonzchen.github.io/bison/assets/images/methods.png)

**说明**: BISON 方法概览。展示 [[标注函数]] $\mathcal{L}$、领域理论 $\mathcal{D}$、演示数据三大输入，以及双层策略学习（LL/HL 分别训练）和执行（HL 提议动作 → LL 执行）的完整流程。

### Figure 2: HL 策略学习流程

**说明**: 三步过程——(1) 通过 $\mathcal{L}$ 将低层演示转为 HL 轨迹；(2) 用[[目标回归]]提取 ground condition-action 规则；(3) [[变量提升]]生成一阶符号策略，泛化至任意对象。

### Figure 3: LL 策略 GNN 架构

**说明**: GNN 中 global 节点编码 agent 状态和 nullary 事实，action arg 节点编码高层动作参数，other 节点编码其他对象。通过消息传递汇聚信息，最终 global 节点输出低层动作 $\mathbf{a}^{ll}$。

### Figure 4: 各方法成功率对比（跨物体数量）

![Success Rate vs Objects](https://arxiv.org/html/2605.15975v2/x1.png)

**说明**: 不同方法在 8 个环境中随物体数量变化的成功率中位数（线）和范围（阴影）。BISON 在大多数环境中随物体数量增加保持高成功率，VLA 基线（SmolVLA）成功率为 0%（图中省略）。

### Figure 5: 成功率 vs. 训练/推理时间

![Success Rate vs Time](https://arxiv.org/html/2605.15975v2/x2.png)

**说明**: 以 10 个物体为例，BISON 的训练效率和推理效率对比。BISON 相比 replanning 方法（DetReplan、NdtReplan）在训练和推理阶段均显著更快，因为无需执行时搜索。

### Figure 6: HL 求解时间 vs. 物体数量

![HL Solving Time](https://arxiv.org/html/2605.15975v2/x3.png)

**说明**: BISON 的高层策略求解时间随物体数量的变化。由于一阶规则的归纳泛化，BISON HL 策略在 10,000 个物体的规划问题中仍可在 1 分钟内求解，而基于搜索的方法时间随物体数量指数增长。

### Table 1: 实验环境的不确定性属性

| 不确定性类型 | BlocksS | BlocksN | ColourS | ColourN | FactoryS | FactoryN | GachaS | GachaN |
|------------|---------|---------|---------|---------|----------|----------|--------|--------|
| 外生不确定性（Exogenous） | ✓ | ✓ | ✓ | ✓ | | | | |
| 内生不确定性（Endogenous） | | | | | ✓ | ✓ | ✓ | ✓ |
| 状态不确定性（State） | ✓ | ✓ | | | | | | |

**说明**: S/N 后缀分别表示确定性/非确定性变体。外生不确定性指环境物体随机移动，内生不确定性指动作结果不确定（如抓取是否成功），状态不确定性指部分可观测性。

### Table 2: 各方法平均成功率（测试物体数 10）

| 类型 | 方法 | BlocksS | BlocksN | FactoryS | FactoryN | ColourS | ColourN | GachaS | GachaN | 总计 |
|------|------|---------|---------|----------|----------|---------|---------|--------|--------|------|
| VLA | SmolVLA | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 |
| VLA | SmolVLA^MW | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 |
| GNN | PureNN | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 |
| GNN | PddlNN | 0.33±0.4 | 0.20±0.3 | 0.18±0.3 | 0.07±0.2 | 0.12±0.2 | 0.36±0.3 | 0.51±0.3 | 0.60±0.4 | 0.30±0.2 |
| Planning | DetPlan | 0.90±0.1 | 0.34±0.3 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.15±0.3 |
| Planning | NdtPlan | **0.99±0.0** | 0.39±0.4 | 0.00±0.0 | 0.00±0.0 | **0.61±0.2** | 0.01±0.0 | 0.00±0.0 | 0.00±0.0 | 0.25±0.4 |
| Planning | DetReplan | 0.98±0.0 | **0.97±0.1** | 0.94±0.1 | 0.59±0.3 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.00±0.0 | 0.44±0.4 |
| Planning | NdtReplan | **0.99±0.0** | **0.97±0.0** | 0.94±0.1 | 0.51±0.3 | 0.26±0.3 | 0.17±0.3 | 0.00±0.0 | 0.00±0.0 | 0.48±0.4 |
| **Ours** | **BISON** | **0.99±0.0** | 0.95±0.1 | **0.97±0.0** | **0.62±0.3** | 0.54±0.2 | **0.99±0.0** | **0.65±0.3** | **0.64±0.3** | **0.79±0.2** |

**关键发现**: BISON 在 8 个环境中平均成功率 **79%**，大幅超越所有基线。VLA 和端到端 GNN 方法均失败（0%）。BISON 在 2 个环境（ColourS、GachaN）的 HL 执行协方差漂移导致表现略低于 replanning 方法，但整体最优。

---

## 实验

### 数据集 / Benchmark

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| MetaWorld（扩展版） | 8 环境 × 21,600 episodes | 机械臂操作，支持变长物体数量 | 训练 + 测试 |
| BlocksS/N | 积木堆叠 | 外生/状态不确定性 | 测试 |
| FactoryS/N | 工厂流水线 | 内生不确定性 | 测试 |
| ColourS/N | 颜色分类 | 外生不确定性 | 测试 |
| GachaS/N | 随机抽取 | 内生不确定性 | 测试 |

- **训练**: 3 个物体的演示
- **测试**: 泛化到 4-10 个物体

### 实现细节

- **LL 策略参数量**: <33,000 参数（GNN）
- **训练方式**: [[行为克隆]]（behavioral cloning）
- **HL 策略**: 前向链推理（forward chaining），无需梯度
- **环境**: 扩展版 MetaWorld，支持非确定性和变长物体
- **对比基线**: SmolVLA（微调 MetaWorld）、SmolVLA^MW（MetaWorld 预训练）、PureNN、PddlNN、DetPlan、NdtPlan、DetReplan、NdtReplan

### 关键结果

- BISON 在 10 个物体的长时域任务中泛化成功，而训练仅用 3 个物体
- HL 策略可在 1 分钟内解决含 10,000 个相关物体的 PDDL 规划问题（经典符号求解器失败）
- 相比 replanning 方法，BISON 在训练时间和推理时间上均更高效（无执行时搜索）
- 2 个环境（ColourS、GachaN）中 BISON 低于 replanning，归因于 LL 策略执行 HL 动作时的协方差漂移

---

## 批判性思考

### 优点

1. **无执行时搜索**: 学习到的符号策略无需在推理时运行规划器，计算效率高于 replanning 方法
2. **可解释性强**: 一阶 condition-action 规则是人类可读的，支持调试和知识迁移
3. **开放世界泛化**: 从 3 个物体泛化到 10 个物体，理论上支持任意多物体
4. **理论保障**: NDRP 定理提供双层一致性保证，Theorem 1 保证在有限数据下的泛化能力

### 局限性

1. **依赖领域理论**: 需要预定义或预学习的符号抽象语言 $\mathcal{D}$ 和[[标注函数]] $\mathcal{L}$，在新领域需要人工设计
2. **LL 策略无最优性保证**: 缺乏对低层策略泛化和最优性的理论保证
3. **协方差漂移**: LL 策略执行 HL 动作时可能产生分布偏移，在某些环境中影响性能
4. **继承双层规划局限**: 有用的 HL 抽象必须存在，对高度连续或细粒度任务可能不适用

### 潜在改进方向

1. 自动学习领域理论 $\mathcal{D}$ 和[[标注函数]] $\mathcal{L}$，减少人工设计依赖
2. 引入在线 RL 微调 LL 策略以解决协方差漂移
3. 扩展到视觉输入（当前使用状态向量），接近真实机器人场景

### 可复现性评估

- [ ] 代码开源（项目主页有，但代码链接待确认）
- [x] 训练细节完整（论文附录中有详细超参数）
- [x] 数据集可获取（基于公开 MetaWorld）
- [x] 理论证明完整（附有完整定理证明）

---

## 关联笔记

### 基于

- [[符号规划]]: 提供高层抽象和规则语言
- [[模仿学习]]: 从演示中学习双层策略的训练范式
- [[图神经网络]]: 低层策略的神经网络架构
- [[目标回归]]: 用于从 HL 轨迹提取 condition-action 规则

### 对比

- [[SmolVLA]]: VLA baseline，成功率 0%，说明端到端方法无法处理长时域规划
- [[任务与运动规划|TAMP]]: 经典双层规划方法，BISON 避免了其执行时搜索的开销

### 方法相关

- [[双层策略]]: BISON 的核心框架
- [[标注函数]]: 连接低层观测与高层符号状态的关键组件
- [[非确定性下行精化性质]]: 双层一致性的理论保障
- [[行为克隆]]: LL 策略的训练方式

### 数据集相关

- [[MetaWorld]]: 主要 benchmark，BISON 在其扩展版上评估

---

## 速查卡片

> [!summary] BISON: Learning Bilevel Policies over Symbolic World Models
> - **核心**: 从演示中学习双层策略（一阶符号规则 + GNN），无需执行时搜索即可解决长时域规划
> - **方法**: HL 策略通过目标回归 + 变量提升学习一阶规则；LL 策略为 33K 参数 GNN
> - **结果**: 8 环境平均成功率 79%，训练 3 物体泛化 10 物体，1 分钟解决 10000 物体规划
> - **代码**: [dillonzchen.github.io/bison](https://dillonzchen.github.io/bison)

---

*笔记创建时间: 2026-05-21*
