---
title: "Physically Viable World Models: A Case for Query-Conditioned Embodied AI"
method_name: "PVWM"
authors: [Adam J. Thorpe, Stepan Tretiakov, Cheng-Hsi Hsiao, Su Ann Low, Xingjian Li, Hassan Iqbal, Neel P. Bhatt, Ufuk Topcu, Krishna Kumar]
year: 2026
venue: arXiv
tags: [world-model, embodied-ai, physical-reasoning, query-conditioned, physics-simulation, intervention-planning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.30542
created: 2026-06-02
---

# 论文笔记：Physically Viable World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Texas at Austin |
| 日期 | May 2026 |
| 项目主页 | [pvwm.github.io](https://pvwm.github.io/) |
| 对比基线 | [[V-JEPA 2]], [[LTX-2]], [[GPT-4o]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.30542) / [Code](https://github.com/pvwm-paper/physically-viable-world-models) |

---

## 一句话总结

> 当前 [[World Model|世界模型]] 只预测视觉观测，无法推断物理因果；本文提出以干预查询为中心，按需构建最简物理抽象的"物理可行世界模型"框架，解决具身 AI 规划中的根本性物理推理失效问题。

---

## 核心贡献

1. **系统性失效诊断**: 设计 5 类受控基准测试，证明 VLM、视频扩散模型和隐空间预测模型在潜在物理量变化下会产生视觉合理但物理错误的预测
2. **查询条件化框架**: 提出以干预查询（intervention query）驱动的模块化世界模型构建方法，强调"回答特定查询所需的最简物理抽象"
3. **三个具体演示**: 通过斜坡-杯子干预、粘度依赖倒液和涉水路面行驶验证框架可行性，揭示编排（orchestration）为核心开放问题

---

## 问题背景

### 要解决的问题

具身 AI 的规划、控制和验证需要预测智能体干预环境后的结果。当前主流方法是[[观测预测模型|观测预测世界模型]]（observation-predictive world model），即预测未来帧或隐状态序列。然而这类模型无法区分"视觉一致但物理不同"的情景——相同初始观测在不同潜在物理参数（质量、摩擦、粘度）下会有截然不同的干预结果。

### 现有方法的局限

- **[[视觉语言模型|VLM]]（如 GPT-5.5）**: 能识别相关物理效应，但面对需要阈值判断的结果（如物体是否倾倒）时预测不一致，无法推断不可见的物理参数
- **[[视频扩散模型]]（如 LTX-2）**: 通过关键帧锁定和文本引导生成视觉连贯的视频，但系统性违反接触动力学和流体力学规律
- **[[隐空间世界模型]]（如 [[V-JEPA 2]]）**: 在行动条件化的隐空间中进行 MPC 规划，生成的计划看似合理但在物理上不可行

核心问题：同一观测可以匹配多个物理系统，而这些系统在被干预时表现截然不同（the same observations can fit multiple physical systems that behave differently when acted on）。

### 本文的动机

物理可行性是相对于查询的，而非绝对的。回答"杯子会不会倾倒？"和"机器人在哪里？"所需的变量不同。因此，不应构建通用仿真器，而应针对每个干预查询确定"保留查询相关区分的最简模型"（the simplest model that preserves the distinctions relevant to the query）。

---

## 方法详解

### 框架概述

PVWM 提出以**干预查询**为出发点，模块化构建物理可行世界模型，而非端到端训练通用预测器。

**核心原则**: 物理结构不是大规模观测预测的涌现属性，而是模型构建过程的显式需求。

### 五大模块

#### 模块 1: 感知（Perception）

**设计动机**: 从原始传感器数据中恢复查询相关的场景变量

**具体实现**:
- 使用 [[3D Gaussian Splatting]] 重建场景几何
- 结合 [[深度估计]] 和物体检测获取初始状态
- 针对特定查询变量（如液位、接触点高度）进行专项感知

#### 模块 2: 表示（Representation）

**设计动机**: 将感知证据映射到物理量

**具体实现**:
- 选择适合查询的最小状态空间（如仅跟踪质心位置而非完整网格）
- 排除查询无关的视觉细节
- 处理不可观测的潜在变量（质量、摩擦系数、粘度）需要估计

#### 模块 3: 动作规范（Action Specification）

**设计动机**: 定义查询允许的干预空间

**具体实现**:
- 明确参数化可行动作集合（如倾倒角度轨迹、持续时间、停止条件）
- 区分连续控制参数和离散决策

#### 模块 4: 动力学与约束（Dynamics & Constraints）

**设计动机**: 在选定的物理抽象下演化状态

**具体实现**:
- 解析模型（刚体动力学）
- 数值求解器：[[NVIDIA Newton]]（刚体/可变形体/机器人接触，基于 XPBD）、[[Genesis]]（SPH 流固耦合）
- 学习代理（learned surrogate）针对特定参数范围
- 约束集合：不可穿透性、流体守恒、执行器极限

#### 模块 5: 查询响应（Query Response）

**设计动机**: 返回干预查询所需的决策形式

**具体实现**:
- 轨迹（规划任务）
- 可行性集合（安全验证）
- 安全证书（控制任务）
- 不是渲染视频，而是决策相关输出

### 编排器（Orchestrator）

**核心开放问题**: 如何自动选择兼容的组件库（解析模型、数值求解器、学习代理、估计器）来回答给定查询。

编排器的兼容性约束：所选的表示、动作接口、动力学、约束和输出必须支持同一物理抽象。

---

## 关键公式

### 公式 1: [[物理可行倒液约束|倒液目标体积约束]]

$$
V_{\text{recv}}(T) \in [V^{\star} - \epsilon,\ V^{\star} + \epsilon]
$$

**含义**: 机器人倒液任务中，时间 $T$ 时接收杯中液体体积必须在目标体积的容差范围内

**符号说明**:
- $V_{\text{recv}}(T)$: 时间 $T$ 时接收容器中的液体体积
- $V^{\star}$: 目标转移体积（目标量的一半）
- $\epsilon$: 体积容差

### 公式 2: [[物理可行倒液约束|倒液溢出约束]]

$$
V_{\text{spill}}(T) \leq \delta
$$

**含义**: 整个倒液过程中溢出量不得超过最大允许溢出量

**符号说明**:
- $V_{\text{spill}}(T)$: 时间 $T$ 时的总溢出体积
- $\delta$: 最大允许溢出量

**联合含义**: 上述两个约束共同定义了粘度相关倒液查询的规划目标——选择动作序列 $u_{0:T}$（包括倾斜轨迹、角速度、持续时间、停止规则），使接收体积在目标范围内且溢出可控。

---

## 关键图表

### Figure 1: 视觉世界模型的根本局限

![Figure 1](https://arxiv.org/html/2605.30542v1/x1.png)

**说明**: 视觉世界模型可生成视觉合理但物理上不可能的预测。相同外观的场景在干预下可能有截然不同的物理结果，因此需要能识别"回答干预查询的最简物理抽象"的世界模型。

### Figure 2: 受控评估场景套件

![Figure 2](https://arxiv.org/html/2605.30542v1/x2.png)

**说明**: 用于揭露视觉世界模型在潜在物理量变化下失效的 5 类受控评估场景：
1. 刚体碰撞（密度变化）
2. 可变形体交互（弹性墙）
3. 刚-流耦合（水杯交互）
4. 接触高度变化的机器人推物
5. 粘度相关的倒液任务

### Figure 3: 物理可行世界模型的构建流程

![Figure 3](https://arxiv.org/html/2605.30542v1/x3.png)

**说明**: 以用户指定的干预查询为起点，确定需要构建的物理抽象。感知和表示模块恢复查询相关场景变量，动作规范定义可行干预，动力学与约束在所选状态下演化，预测返回查询所需响应。兼容性关系强调：选定的表示、动作接口、动力学、约束和输出必须支持同一物理抽象。

### Figure 4: 斜坡-杯子干预查询的反事实评估

![Figure 4](https://arxiv.org/html/2605.30542v1/x4.png)

**说明**: 在不同释放高度、球材质和液位条件下评估斜坡-杯子干预查询。杯子是否倾倒取决于球的质量、恢复系数、释放高度、液位等视觉上不可见的变量——这是纯视觉模型无法处理的核心挑战。

### Figure 5: 粘度相关倒液的查询条件化世界模型

![Figure 5](https://arxiv.org/html/2605.30542v1/figures/genesis_pour_target_honey_2x5_timestep_grid.jpg)

**说明**: (a) 用固定的探测倒液动作，通过观察流动、填充、残留和溢出行为推断未知液体的粘度；(b) 利用估计的粘度选择倒液动作和停止条件，使转移体积恰好为源杯初始液量的一半。展示了[[主动感知|主动参数估计]]与物理规划的闭环。

### Figure 6: 从 Gaussian Splat 构建涉水驾驶的物理可行模型

![Figure 6](https://arxiv.org/html/2605.30542v1/x5.png)

**说明**: 上：从观测重建的原始 [[3D Gaussian Splatting|Gaussian Splat]] 场景。下：增加刚性支撑几何体和基于 [[MPM（物质点法）|MPM]] 的流体仿真，用于评估刚-流交互下的涉水路面通过性。橙色圆圈标注车辆位置。展示了将感知输出与查询专用物理仿真结合的能力。

### Figure 7: VLM 物理预测的静态图像基准

**场景图片**:
- 刚体碰撞: ![superball](https://arxiv.org/html/2605.30542v1/figures/superball.png)
- 单弹性墙: ![single jelly](https://arxiv.org/html/2605.30542v1/figures/single_jelly.png)
- 双弹性墙: ![double jelly](https://arxiv.org/html/2605.30542v1/figures/double_jelly.png)
- 水杯交互: ![cup water](https://arxiv.org/html/2605.30542v1/figures/cup_water.png)

**说明**: 用于测试 VLM（GPT-5.5）物理预测能力的静态图像基准。球从斜坡滚下进入不同交互场景：刚体碰撞、单弹性墙碰撞、双弹性墙碰撞、刚-流交互（水杯）。测试 VLM 预测是否能追踪干预下的底层物理响应，而非仅依赖视觉相似性。

### Figure 8: 隐藏材质变化下的刚体交互时序

![Figure 8](https://arxiv.org/html/2605.30542v1/figures/newton_sim_4x5_timestep_grid.jpg)

**说明**: 4 种条件下的仿真时序：(a) 密度变化组-木块塔；(b) 密度变化组-钢块塔；(c) 恢复系数组-高恢复系数弹丸；(d) 恢复系数组-高恢复系数弹丸+钢块。证明相同视觉外观下，材质差异导致截然不同的碰撞结果。

### Figure 9: 球-杯碰撞的物理仿真 vs 视频扩散模型

**(a) 物理仿真+渲染**:
![cup demo frame 1](https://arxiv.org/html/2605.30542v1/figures/cup_demo_frame_001.jpg)
![cup demo frame 2](https://arxiv.org/html/2605.30542v1/figures/cup_demo_frame_002.jpg)
![cup demo frame 3](https://arxiv.org/html/2605.30542v1/figures/cup_demo_frame_003.jpg)
![cup demo frame 4](https://arxiv.org/html/2605.30542v1/figures/cup_demo_frame_004.jpg)
![cup demo frame 5](https://arxiv.org/html/2605.30542v1/figures/cup_demo_frame_005.jpg)
![cup demo frame 6](https://arxiv.org/html/2605.30542v1/figures/cup_demo_frame_006.jpg)

**(b) 扩散模型（LTX-2）生成**:
![cup diff frame 1](https://arxiv.org/html/2605.30542v1/figures/cup_diff_frame_001.jpg)
![cup diff frame 2](https://arxiv.org/html/2605.30542v1/figures/cup_diff_frame_002.jpg)
![cup diff frame 3](https://arxiv.org/html/2605.30542v1/figures/cup_diff_frame_003.jpg)
![cup diff frame 4](https://arxiv.org/html/2605.30542v1/figures/cup_diff_frame_004.jpg)
![cup diff frame 5](https://arxiv.org/html/2605.30542v1/figures/cup_diff_frame_005.jpg)
![cup diff frame 6](https://arxiv.org/html/2605.30542v1/figures/cup_diff_frame_006.jpg)

**说明**: 物理仿真（NVIDIA Newton + Genesis SPH）正确捕捉刚体交互和流体粒子动力学；视频扩散模型（LTX-2）在固定初始帧和文本引导下生成视觉连贯但物理错误的续帧，无法正确处理接触和流体动力学。

### Figure 10: 球-双弹性墙碰撞的物理仿真 vs 视频扩散模型

**(a) 物理仿真+渲染**:
![jelly demo frame 1](https://arxiv.org/html/2605.30542v1/figures/jelly_demo_2_frame_001.jpg)
![jelly demo frame 2](https://arxiv.org/html/2605.30542v1/figures/jelly_demo_2_frame_002.jpg)
![jelly demo frame 3](https://arxiv.org/html/2605.30542v1/figures/jelly_demo_2_frame_003.jpg)
![jelly demo frame 4](https://arxiv.org/html/2605.30542v1/figures/jelly_demo_2_frame_004.jpg)
![jelly demo frame 5](https://arxiv.org/html/2605.30542v1/figures/jelly_demo_2_frame_005.jpg)
![jelly demo frame 6](https://arxiv.org/html/2605.30542v1/figures/jelly_demo_2_frame_006.jpg)

**(b) 扩散模型（LTX-2）生成**:
![jelly diff frame 1](https://arxiv.org/html/2605.30542v1/figures/jelly_diff_2_frame_001.jpg)
![jelly diff frame 2](https://arxiv.org/html/2605.30542v1/figures/jelly_diff_2_frame_002.jpg)
![jelly diff frame 3](https://arxiv.org/html/2605.30542v1/figures/jelly_diff_2_frame_003.jpg)
![jelly diff frame 4](https://arxiv.org/html/2605.30542v1/figures/jelly_diff_2_frame_004.jpg)
![jelly diff frame 5](https://arxiv.org/html/2605.30542v1/figures/jelly_diff_2_frame_005.jpg)
![jelly diff frame 6](https://arxiv.org/html/2605.30542v1/figures/jelly_diff_2_frame_006.jpg)

**说明**: 物理仿真正确处理弹塑性材料与弹性材料之间的交互；扩散模型无法正确模拟可变形体的弹性形变和恢复行为。

*(Figures 11-13 记录机器人推墙和粘度倒液变体，完整内容见论文附录 B)*

---

## 实验

### 受控基准设计（5 类场景）

| 场景类型 | 潜在变量 | 物理引擎 | 测试对象 |
|---------|---------|---------|---------|
| 刚体碰撞（斜坡-塔） | 密度、恢复系数 | NVIDIA Newton | VLM, 视频扩散, 隐模型 |
| 可变形墙交互 | 弹性模量、冲击能量 | NVIDIA Newton | VLM, 视频扩散 |
| 刚-流耦合（球-水杯） | 液位、填充量 | Genesis (SPH) | VLM, 视频扩散 |
| 机器人推物（接触高度） | 接触高度、摩擦 | NVIDIA Newton | 隐空间模型 |
| 粘度依赖倒液 | 液体粘度 | Genesis (SPH) | 查询条件化框架 |

### 基线模型

- **VLM**: GPT-5.5，在无/低/高上下文三种条件下测试静态图像预测
- **视频扩散**: LTX-2（2026年4月开源排名第一），固定初始帧+文本引导
- **隐空间世界模型**: V-JEPA 2，行动条件化 MPC

### 主要发现

1. **VLM**: 能识别相关物理效应，但阈值预测（是/否）不一致；无法从像素推断质量、摩擦、粘度
2. **视频扩散**: 视觉连贯性高，但系统性违反接触动力学和流体守恒；关键帧锁定和文本引导不足以强制物理一致性
3. **隐空间模型**: 规划轨迹在物理上不可行；潜在表示未保留干预相关的物理结构
4. **PVWM 框架**: 通过查询条件化物理抽象，在三个演示场景中成功完成规划和控制

### 物理引擎细节

- **[[NVIDIA Newton]]**: 基于 XPBD（位置约束动力学）的刚体、可变形体和机器人接触仿真
- **[[Genesis]]**: 基于 SPH（光滑粒子流体动力学）的流固耦合仿真

---

## 批判性思考

### 优点

1. **问题定位精准**: 清晰揭示了当前世界模型的根本局限——视觉预测与物理因果推理的分离，属于"认识论"层面的贡献
2. **受控实验严谨**: 5 类基准设计中，外观固定而物理参数变化，确保比较的公平性
3. **框架通用性**: 模块化设计允许混合使用解析模型、数值求解器和学习代理，适应不同查询需求

### 局限性

1. **编排问题未解决**: 如何自动为未知查询选择正确抽象层次，是论文承认的核心开放问题
2. **可辨识性限制**: 质量、摩擦、粘度等潜在量从被动观测中往往无法辨识，需要主动交互或条件响应
3. **更丰富抽象的代价**: 提高抽象层次可改善鲁棒性，但同时增加估计和验证负担
4. **当前演示规模有限**: 三个演示场景较为简单，实际机器人任务中的开放世界泛化尚未验证

### 潜在改进方向

1. **学习驱动的编排器**: 使用 LLM 或程序合成自动选择和组合物理模块
2. **主动感知策略**: 设计最优探测动作序列以最快速度辨识关键潜在变量
3. **不确定性下的鲁棒规划**: 当潜在量只能部分辨识时，利用集合成员资格或风险敏感规划

### 可复现性评估

- [x] 代码开源（[GitHub](https://github.com/pvwm-paper/physically-viable-world-models)）
- [ ] 预训练模型（依赖外部物理引擎 Newton 和 Genesis）
- [x] 训练细节完整（以受控仿真为主，无需神经网络训练）
- [x] 数据集可获取（仿真场景可自行生成）

---

## 关联笔记

### 基于

- [[3D Gaussian Splatting]]: 场景重建和感知基础
- [[XPBD（位置约束动力学）|XPBD]]: NVIDIA Newton 的底层物理求解算法
- [[SPH（光滑粒子流体动力学）|SPH]]: Genesis 流体仿真基础

### 对比

- [[V-JEPA 2]]: 行动条件化隐空间世界模型，作为主要对比基线
- [[LTX-2]]: 视频扩散世界模型基线
- [[GPT-4o]]: VLM 物理预测基线

### 方法相关

- [[查询条件化世界模型]]: 本文的核心方法概念
- [[干预规划]]: 世界模型的规划应用
- [[物理仿真]]: 底层物理引擎

### 领域相关

- [[World Model]]: 具身 AI 世界模型总览
- [[物理推理]]: CV/AI 领域的物理理解研究方向
- [[MPM（物质点法）]]: 涉水驾驶场景使用的流体仿真方法

---

## 速查卡片

> [!summary] Physically Viable World Models (PVWM)
> - **核心**: 视觉世界模型无法区分"看起来一样但物理不同"的场景，导致干预预测失效
> - **方法**: 以干预查询为中心，模块化构建"回答特定查询的最简物理抽象"
> - **结果**: 证明 VLM/扩散/隐模型的系统性失效；在斜坡-杯子、粘度倒液、涉水驾驶三场景验证框架
> - **代码**: [github.com/pvwm-paper/physically-viable-world-models](https://github.com/pvwm-paper/physically-viable-world-models)

---

*笔记创建时间: 2026-06-02*
