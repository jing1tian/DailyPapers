---
title: "Closing the Loop in Humanoid VLA: Persistent 3D Object Tokens for Verifiable Loco-Manipulation"
method_name: "POT-VLA"
authors: [Peng Ren, Haoyang Ge, Jiang Zhao, Cong Huang, Yukun Shi, Pei Chi, Kai Chen]
year: 2026
venue: arXiv
tags: [vla, humanoid-robot, loco-manipulation, object-centric, 3d-perception, execution-verification]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.18016v1
created: 2026-07-22
---

# 论文笔记：Closing the Loop in Humanoid VLA: Persistent 3D Object Tokens for Verifiable Loco-Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开（arXiv 投稿） |
| 日期 | July 2026 |
| 项目主页 | N/A |
| 对比基线 | [[Being-H0.7]] (Being-0) |
| 链接 | [arXiv](https://arxiv.org/abs/2607.18016) |

---

## 一句话总结

> POT-VLA 通过维护角色索引的持久化 3D 物体记忆，同时驱动全身动作生成和执行验证，解决了人形机器人 [[Loco-Manipulation]] 中感知-动作-验证三者状态不一致的问题。

---

## 核心贡献

1. **Persistent Object Tokenization (POT)**: 为任务相关实体维护角色索引的 3D 物体记录（K=8 槽位，F=33 特征），在运动、接触、遮挡中持续追踪物体状态
2. **闭环 VLA 架构**: 将物体 token 插入 [[DiT]] 动作头的自注意力序列，使同一物体记忆驱动动作生成和后验证，消除感知分叉
3. **谓词监督器（Predicate Supervisor）**: 每个动作块执行后通过几何关系检查任务完成状态，支持定向恢复和重规划

---

## 问题背景

### 要解决的问题

现代 [[VLA]] 策略隐式编码视觉-语言特征，导致**物体状态分叉（object-state divergence）**：用于生成动作的物体状态与用于验证任务完成的状态来自不同表示，产生两类失败：
1. 机器人使用陈旧状态继续执行而任务已实际失败（过早完成判断）
2. 恢复逻辑用于错误的物体或使用过期的空间位置

### 现有方法的局限

- 通用 VLA（RT-2、π0、OpenVLA）：任务相关物体状态隐式编码，不可直接访问用于验证
- 目标中心 3D 方法（VIMA、VoxPoser、RVT）：适用于桌面操作，难以在行走、接触、遮挡、恢复中跨操作维持可寻址的任务实体
- 执行验证与重规划（Code as Policies、VerifyLLM）：使用独立于动作条件的感知，无法形成真正的闭环

### 本文的动机

通过维护**共享**的 3D 物体记忆，使感知、动作生成和任务验证三者共享同一物体状态表示，从根本上消除状态分叉。

---

## 方法详解

### 系统架构

POT-VLA 采用**持久化物体记忆驱动的全身 VLA** 架构：

- **输入**: 语言指令 $l$ + RGB-D 观测 $o_t$ + 机器人状态 $s_t$
- **物体记忆模块**: [[SAM3]] + RGB-D 反投影 → [[对象中心表示|角色索引 3D 物体记录]] $\mathcal{M}^{\tau_i}_t$
- **Backbone**: 视觉-语言主干（保持冻结），输出交叉注意力特征 $Z_t^{vl}$
- **核心模块**: POT → [[DiT]] 动作头（插入物体 token）→ [[Action Chunking|动作块]] $\hat{a}_{t:t+H}$
- **验证模块**: 谓词监督器（[[Predicate Supervisor]]）在每块执行后检查状态

整体循环：**维护物体记忆 → 生成动作 → 执行 → 观测更新 → 验证 → （需要时）恢复**

### 核心模块

#### 模块一：Persistent Object Tokenization (POT)

**设计动机**: 将任务物体建模为机器人基坐标系中的持久 3D 实体，跨运动/接触/遮挡保持可寻址性。

**槽位设计**（K=8 槽位，F=33 特征/槽位）：

每个槽位 $m_t^e$ 包含：
- **可见性与置信度**: 当前帧可见标志 $v_k$、追踪置信度 $q_k$
- **2D 感知**: 归一化边界框坐标 $b_k$
- **3D 几何**: 机器人基坐标系中心点 $c_k$、3D 范围 $s_k$、朝向字段 $o_k$
- **关系特征**: 物体-容器距离、支撑高度、末端执行器-物体偏移、目标-目的地位移、对齐度、包含性、交递线索等 8 维关系特征 $\rho_k$

**角色类型**（实体集 $\mathcal{E}_i$）：
- `TARGET`：待操作目标物体
- `DESTINATION`：放置目标位置
- `SUPPORT`：支撑面
- `HANDOVER_PARTNER`：交递协作对象
- `REFERENCE`：参考物体
- `PADDING`：填充槽位（屏蔽计算）

所有计算在**机器人基坐标系**中进行，通过 RGB-D 反投影获得 3D 信息。

#### 模块二：VLA 集成（DiT 动作头注入）

**设计思路**: 不修改视觉-语言主干，只在 [[DiT]] 动作头的自注意力序列中注入物体 token。

**投影架构**:

$$
Z_t^{obj} = f_\theta^{obj}(x_t^{obj}) \in \mathbb{R}^{K \times d}
$$

其中 $f_\theta^{obj}$ 为：`LayerNorm → Linear → GELU → Dropout → Linear`

**自注意力序列拼接**:

$$
S_t = [Z_t^{state}, Z_t^{obj}, Z_t^{action}]
$$

- 视觉-语言主干输出 $Z_t^{vl}$ 通过**交叉注意力**注入（不在序列中直接拼接）
- 角色嵌入（role embedding）和上下文嵌入（context embedding）区分任务语义
- PADDING 槽位通过掩码排除在注意力计算之外
- 仅微调动作头参数，主干保持冻结

**训练目标**: 与基础动作专家相同的动作块预测目标，无额外辅助损失。

#### 模块三：谓词监督器（Predicate Supervisor）

**谓词定义**:

$$
p = \langle \kappa, \alpha, \mathrm{op}, \nu, n \rangle
$$

- $\kappa$: 谓词类型（containment/support/proximity/displacement/bimanual/handover）
- $\alpha$: 绑定到物体记录的参数（来自 $\mathcal{M}^{\tau_i}_t$）
- $\mathrm{op}$: 比较算符
- $\nu$: 阈值或目标值
- $n$: 时序一致性稳定窗口

**证据函数**:

$$
\phi(\kappa, \alpha, \mathcal{M}^{\tau_i}_\xi)
$$

在最近 $n$ 帧记忆上评估以确保时序稳定。

**监督器返回状态**：`in_progress` / `done` / `blocked` / `failed` / `uncertain`

**任务结构**:
- 前置条件集 $\mathcal{P}_i$：检查就绪性
- 目标集 $\mathcal{G}_i$：检查完成性
- 低置信度记录触发重观测；诊断信息驱动定向恢复（重试/重定位/重规划）

---

## 关键公式

### 公式 1：[[对象中心表示|物体记忆集合]] $\mathcal{M}^{\tau_i}_t$

$$
\mathcal{M}^{\tau_i}_t = \{m_t^e \mid e \in \mathcal{E}_i\}
$$

**含义**: 任务 $\tau_i$ 在时间步 $t$ 的物体记忆，是各角色实体记录的集合。

**符号说明**:
- $\mathcal{M}^{\tau_i}_t$: 时刻 $t$ 任务 $\tau_i$ 的整体物体记忆
- $m_t^e$: 实体 $e$ 在时刻 $t$ 的单条物体记录（K=8 槽位之一）
- $\mathcal{E}_i$: 任务 $\tau_i$ 的角色实体集合

### 公式 2：[[对象中心表示|POT 序列化]] $x_t^{obj}$

$$
x_t^{obj} = T(\mathcal{M}^{\tau_i}_t)
$$

**含义**: 将物体记忆序列化为扁平化特征向量，用于后续投影。

**符号说明**:
- $T$: 序列化函数，将 K×F 记忆矩阵展平
- $x_t^{obj} \in \mathbb{R}^{K \times F}$: 原始槽位特征（K=8, F=33）

### 公式 3：[[DiT|物体 Token 投影]] $Z_t^{obj}$

$$
Z_t^{obj} = f_\theta^{obj}(x_t^{obj}) \in \mathbb{R}^{K \times d}
$$

**含义**: 通过 MLP 将原始槽位特征投影到动作头的隐空间维度 $d$。

**符号说明**:
- $f_\theta^{obj}$: LayerNorm → Linear → GELU → Dropout → Linear
- $K=8$: 槽位数量
- $d$: DiT 动作头隐空间维度

### 公式 4：[[Action Chunking|动作头自注意力序列]] $S_t$

$$
S_t = [Z_t^{state}, Z_t^{obj}, Z_t^{action}]
$$

**含义**: DiT 自注意力序列由机器人状态 token、物体 token 和动作 token 拼接而成。

**符号说明**:
- $Z_t^{state}$: 机器人状态 token
- $Z_t^{obj}$: 持久化 3D 物体 token（本文新增）
- $Z_t^{action}$: 动作 token（待预测）

### 公式 5：[[VLA|条件动作预测]]

$$
\hat{a}_{t:t+H} = \pi_\theta(S_t, Z_t^{vl})
$$

**含义**: 以 DiT 自注意力序列和视觉-语言交叉注意力特征为条件，预测未来 H 步动作块。

**符号说明**:
- $\hat{a}_{t:t+H}$: 预测的未来 H 步动作序列
- $Z_t^{vl}$: 视觉-语言主干输出的交叉注意力特征
- $\pi_\theta$: 参数化的 VLA 策略

### 公式 6：[[Predicate Supervisor|谓词结构]]

$$
p = \langle \kappa, \alpha, \mathrm{op}, \nu, n \rangle
$$

**含义**: 任务谓词的完整规范，定义了物理关系的验证逻辑。

**符号说明**:
- $\kappa$: 谓词类型（containment, support, proximity 等）
- $\alpha$: 从 $\mathcal{M}^{\tau_i}_t$ 绑定的物体参数
- $\mathrm{op}$: 比较算符（<, >, =, ≥ 等）
- $\nu$: 阈值或目标值
- $n$: 时序稳定窗口（需连续 n 帧满足条件）

### 公式 7：[[Predicate Supervisor|谓词证据评估]]

$$
\phi(\kappa, \alpha, \mathcal{M}^{\tau_i}_\xi)
$$

**含义**: 在最近 $\xi$ 帧的物体记忆窗口上评估谓词 $p$ 的证据，确保时序一致性。

**符号说明**:
- $\phi$: 证据评估函数
- $\mathcal{M}^{\tau_i}_\xi$: 最近 $\xi$ 帧的物体记忆历史

---

## 关键图表

### Figure 1: POT-VLA 系统概览

![Figure 1 - POT-VLA System Overview](https://arxiv.org/html/2607.18016v1/pot_vla_system.jpg)

**说明**: POT-VLA 的整体闭环架构。左侧：[[SAM3]] 将 RGB-D 输入转换为角色索引物体记录 $\mathcal{M}^{\tau_i}_t$；中间：序列化为持久化 3D 物体 Token 注入 [[DiT]] 动作头；右侧：执行后通过谓词监督器验证，同一物体记忆刷新用于下一轮。体现了 **维护→动作→观测→验证→恢复** 的完整闭环。

### Figure 2: 真实场景任务展示

| 任务编号 | 任务描述 | 代表帧 |
|----------|----------|--------|
| Task 1 | Cart transport/place（推车搬运） | ![T1](https://arxiv.org/html/2607.18016v1/tasks/task1_1.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task1_2.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task1_3.jpg) |
| Task 2 | Chip box to basket（薯片盒放篮） | ![T2](https://arxiv.org/html/2607.18016v1/tasks/task2_1.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task2_2.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task2_3.jpg) |
| Task 3 | Two balls to basket（双球放篮） | ![T3](https://arxiv.org/html/2607.18016v1/tasks/task3_1.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task3_2.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task3_3.jpg) |
| Task 4 | Stack three cups（三杯叠放） | ![T4](https://arxiv.org/html/2607.18016v1/tasks/task4_1.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task4_2.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task4_3.jpg) |
| Task 5 | Garments to basket（衣物放篮） | ![T5](https://arxiv.org/html/2607.18016v1/tasks/task5_1.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task5_2.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task5_3.jpg) |
| Task 6 | Drawer/tray place-close（抽屉托盘） | ![T6](https://arxiv.org/html/2607.18016v1/tasks/task6_1.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task6_2.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task6_3.jpg) |
| Task 7 | Tabletop sorting（桌面分类） | ![T7](https://arxiv.org/html/2607.18016v1/tasks/task7_1.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task7_2.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task7_3.jpg) |
| Task 8 | Close-range handover（近距离交递） | ![T8](https://arxiv.org/html/2607.18016v1/tasks/task8_1.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task8_2.jpg) ![](https://arxiv.org/html/2607.18016v1/tasks/task8_3.jpg) |

**说明**: 8 种任务族覆盖 [[Loco-Manipulation]] 的核心挑战：运动中的物体追踪（Task 1）、精确放置（Task 2/3）、需要持续 3D 关系的叠放（Task 4）、可变形物体（Task 5）、关节交互（Task 6）、多属性分类（Task 7）、协作交递（Task 8）。

### Table 1：主要对比实验（Unitree G1，每任务 10 次）

| 任务 | Direct Baseline | POT-VLA |
|------|:--------------:|:-------:|
| Cart transport/place | 3/10 | **8/10** |
| Chip box to basket | 9/10 | **10/10** |
| Two balls to basket | 5/10 | **9/10** |
| Stack three cups | 1/10 | **8/10** |
| Garments to basket | 3/10 | **9/10** |
| Drawer/tray place-close | 4/10 | **8/10** |
| Tabletop sorting | 5/10 | **9/10** |
| Close-range handover | 9/10 | **10/10** |
| **总计** | **39/80** | **71/80** |

**关键发现**: POT-VLA 总成功率 71/80（88.75%）vs Direct Baseline 39/80（48.75%）。提升最大的任务是"三杯叠放"（1→8）和"衣物放篮"（3→9），均为需要持续维护 3D 空间关系的场景。Direct Baseline 失败通常是度量误差（抓取不准、叠放偏移、放置失误），而非语义理解问题。

### Table 2：Being-0 对齐服务任务（50 次）

| 任务 | Being-0 | POT-VLA |
|------|:-------:|:-------:|
| Fetch-bottle | 9/10 | **9/10** |
| Deliver-basket | 8/10 | **8/10** |
| Grasp-bottle | 8/10 | **10/10** |
| Place-basket | 6/10 | **9/10** |
| Place-coffee | 6/10 | **8/10** |
| **总计** | **37/50** | **44/50** |

**说明**: 在 [[Being-H0.7|Being-0]] 基准任务上，POT-VLA 以 44/50 超过 Being-0 报告的 37/50，特别在需要精确放置的任务上提升显著。

### Table 3：消融实验（40 次，4 个代表任务）

| 变体 | 物体 Token | 谓词验证器 | 成功次数 |
|------|:----------:|:----------:|:--------:|
| Direct Baseline | — | — | 15/40 |
| Verifier only | — | ✓ | 22/40 |
| POT tokens only | ✓ | — | 31/40 |
| **POT-VLA（完整）** | **✓** | **✓** | **34/40** |

**关键发现**: 物体 Token 的贡献最大（15→31），将基线提升了 107%；验证器在 token 基础上进一步提升（31→34）。验证器单独使用也有效（15→22），说明即使无结构化 token，几何验证也能减少过早完成判断。

### Table 4：物体状态泛化测试（每组 10 次）

| 泛化类型 | Direct Baseline | POT-VLA |
|----------|:--------------:|:-------:|
| 新颖物体实例 | 6/10 | **9/10** |
| 位姿/布局偏移 | 5/10 | **9/10** |
| 干扰物体 | 8/10 | **9/10** |
| 执行中途扰动 | 4/10 | **8/10** |

**关键发现**: POT-VLA 在位姿偏移和执行中途扰动上提升最显著，证明持久化 3D 记忆对布局变化和意外中断的鲁棒性。

---

## 实验

### 机器人平台与任务设置

| 项目 | 内容 |
|------|------|
| 机器人 | [[Unitree G1]] 人形机器人 |
| 任务族 | 8 种，覆盖运输、放置、叠放、分类、关节交互、可变形、交递 |
| 评估规模 | 主实验 80 次（每任务 10 次）+ 消融 40 次 + 泛化 40 次 |
| 对比设置 | Being-0 对齐任务 50 次 |

### 实现细节

- **视觉-语言主干**: 保持冻结（与 Being-0 基线一致）
- **物体追踪**: [[SAM3]] + RGB-D 深度反投影，机器人基坐标系
- **槽位设计**: K=8 槽位，F=33 特征/槽位
- **投影架构**: LayerNorm → Linear → GELU → Dropout → Linear
- **训练目标**: 动作块预测（与基础动作专家相同，无辅助损失）
- **微调范围**: 仅动作头参数（主干冻结）
- 论文未公开具体训练超参（epoch、batch size、学习率、硬件）

### 定性观察

Direct Baseline 的失败模式：度量误差（抓取偏移、叠放不准、放置偏斜），而非语义误解——说明视觉-语言理解已较成熟，精确 3D 物理推理是瓶颈。POT-VLA 的恢复机制（重观测→重试/重定位/重规划）有效处理了夹爪滑落、物体位移等执行意外。

---

## 批判性思考

### 优点

1. **闭环设计的简洁性**: 通过在动作头注入 token（而非修改主干），以最小侵入性实现了感知-动作-验证的状态统一
2. **效果显著**: 71/80 vs 39/80，在空间精度要求高的任务上提升倍数级（杯叠 1→8）
3. **可分解性强**: 消融实验清晰展示了每个组件（token vs verifier）的独立贡献，方法论可信度高

### 局限性

1. **依赖 RGB-D 质量**: [[SAM3]] 追踪和深度反投影对相机标定、遮挡、反光面敏感，论文明确承认
2. **训练细节不透明**: 未公开训练数据规模、演示数量、硬件配置，难以复现
3. **固定槽位数**: K=8 槽位无法动态扩展，任务涉及超过 8 个实体时需要人工干预设计
4. **仅在单一机器人平台测试**: Unitree G1 实验，跨平台泛化性未验证

### 潜在改进方向

1. **多视角主动记忆**: 论文未来工作方向之一，通过多摄像头融合减少遮挡影响
2. **更丰富的接触建模**: 接触力/触觉信息与物体记忆融合
3. **动态槽位数**: 根据任务动态分配槽位

### 可复现性评估

- [ ] 代码开源（未开源）
- [ ] 预训练模型（未提供）
- [ ] 训练细节完整（不完整，缺少超参数）
- [ ] 数据集可获取（未公开演示数据）

---

## 关联笔记

### 基于

- [[VLA]]: 核心策略框架，POT-VLA 在 VLA 基础上增加 3D 物体记忆
- [[DiT]]: 用作动作头架构，接收物体 token 注入
- [[SAM3]]: 用于从 RGB-D 提取 3D 物体记录
- [[Action Chunking]]: 动作块预测范式

### 对比

- [[Being-H0.7]]: 直接在 Being-0 对齐任务上对比，39/80 vs 71/80

### 方法相关

- [[对象中心表示]]: POT 的核心表示思想
- [[Loco-Manipulation]]: 本文核心解决的任务类型
- [[Predicate Supervisor]]: 本文提出的执行验证机制

### 硬件相关

- [[Unitree G1]]: 实验所用人形机器人平台
- [[人形机器人]]: 研究背景

---

## 速查卡片

> [!summary] POT-VLA (2026)
> - **核心**: 持久化 3D 物体记忆同时驱动动作生成与验证，消除感知-动作状态分叉
> - **方法**: K=8 角色索引槽位 + DiT 动作头 token 注入 + 谓词监督器
> - **结果**: 71/80 vs 39/80（Direct Baseline），人形机器人 Loco-Manipulation
> - **代码**: 未开源

---

*笔记创建时间: 2026-07-22*
