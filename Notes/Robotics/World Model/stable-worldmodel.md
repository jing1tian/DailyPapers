---
title: "stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation"
method_name: "stable-worldmodel"
authors: [Lucas Maes, Quentin Le Lidec, Dan Haramati, Nassim Massaudi, Damien Scieur, Yann LeCun, Randall Balestriero]
year: 2026
venue: arXiv
tags: [world-model, reproducibility, benchmark, evaluation, robustness, planning, simulation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2602.08968v2
created: 2026-05-31
---

# 论文笔记：stable-worldmodel-v1

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Mila / McGill University, NYU, Meta AI |
| 日期 | February 2026 |
| 项目主页 | — |
| 对比基线 | [[PLDM]], [[DINO-WM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2602.08968) / Code: — |

---

## 一句话总结

> 首个专为[[世界模型]]研究提供标准化基础设施的模块化生态系统 stable-worldmodel（SWM），通过统一环境、可控变异因子与基线实现，解决该领域可复现性危机。

---

## 核心贡献

1. **标准化世界模型生态系统**: 提供 16 个多样化环境、统一 API、4 个可复现基线实现，覆盖操控、导航、经典控制任务
2. **可控变异因子（FoV）框架**: 每个环境有 6-17 个可控变异维度（颜色/形状/物理参数等），支持系统性零样本鲁棒性研究
3. **DINO-WM 鲁棒性分析**: 通过 SWM 复现 [[DINO-WM]] 并发现其在全部扰动下成功率从 94% 骤降至 4-20%，揭示预训练视觉特征并不天然具备环境鲁棒性

---

## 问题背景

### 要解决的问题

[[世界模型]]（World Model, WM）研究缺乏共享基准和标准化基础设施。视觉领域有 ImageNet，强化学习有 OpenAI Gym，但 WM 领域每篇论文各自搭建环境。

### 现有方法的局限

两篇代表性工作 [[PLDM]] 和 [[DINO-WM]] 各自实现了相同的 Two-Room 环境，却产生了 81 处删除、86 处新增、18 处修改的代码差异——说明即使是同一环境，不同实现也无法直接比较。现有代码库缺乏文档、测试覆盖率低、维护不活跃（见 Table 1 对比）。

### 本文的动机

通过提供统一的模块化基础设施，降低新研究者的入门门槛，同时让比较实验在相同基础上进行，从根本上解决 WM 研究的可复现性问题。

---

## 方法详解

### 核心设计：World 接口

SWM 的核心抽象是 **World 类**，它封装了 [[Gymnasium]] 环境并提供统一的仿真接口：

- **与 Gymnasium 的区别**: `step()` 不返回 `(obs, reward, done, info)` 元组，而是将所有数据存入 `world.infos` 字典，由 [[Policy]] 对象主动查询
- **解耦设计**: 控制逻辑（策略）与环境执行完全解耦，策略只需实现 `get_action(info: dict) → np.ndarray` 接口
- **并行支持**: `num_envs` 参数支持向量化并行仿真

```python
world = swm.World('swm/PushT-v1', num_envs=8)
world.set_policy(YourExpertPolicy())
world.reset()   # 初始化环境
world.step()    # 用策略更新环境
```

### 环境套件（16 个）

| 类别 | 环境 | FoV 数量 |
|------|------|---------|
| 2D 操控 | PushT-v1 | 16 |
| 2D 导航 | TwoRoom-v1 | 17 |
| 3D 控制 | DMControl (Humanoid, Cheetah 等) | 6-10 |
| 机器人操控 | OGBench (多种变体) | 11-12 |

### 可控变异因子（Factors of Variation, FoV）

每个环境的 FoV 覆盖多个维度：

- **视觉属性**: 颜色（`agent.color`, `block.color`, `background.color`）、纹理、形状
- **几何属性**: 大小（`agent.size`, `block.size`, `anchor.size`）、角度、位置
- **物理参数**: 摩擦系数、阻尼、质量、重力

FoV 采用层级命名（如 `agent.color`），支持批量变异（`variation_values="all"`）。

### 评估协议

SWM 提供两种[[目标条件任务]]评估模式：

- **在线评估** (`evaluate`): 每个 episode 从环境中采样初始状态和目标，衡量真实泛化性
- **离线评估** (`evaluate_from_dataset`): 从专家轨迹中选取状态-目标对，保证可行性

### 规划支持

SWM 内置[[模型预测控制]]（MPC）框架，支持多种求解器：

- [[CEM]]（Cross-Entropy Method，交叉熵方法）
- [[MPPI]]（Model Predictive Path Integral）
- 梯度优化器

通过 `PlanConfig` 统一配置：规划时域（horizon）、滚动时域（receding horizon）、热启动（warm start）。

---

## 关键公式

本文为系统性论文，核心贡献不在新公式，而在框架设计。以下为评估相关的关键量化指标：

### 公式1: [[成功率|任务成功率]]

$$
\text{Success Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\text{episode}_i \text{ reaches goal}]
$$

**含义**: 在 $N$ 个 episode 中，成功到达目标的比例，是 SWM 评估的核心指标。

**符号说明**:
- $N$: 评估 episode 总数
- $\mathbf{1}[\cdot]$: 指示函数

### 公式2: [[零样本鲁棒性]]评估

$$
\text{Robustness}(f) = \frac{\text{SuccessRate}(f, \text{perturbed})}{\text{SuccessRate}(f, \text{default})}
$$

**含义**: 模型 $f$ 在扰动环境下的成功率与默认环境下成功率之比，衡量对分布偏移的鲁棒性。论文发现 [[DINO-WM]] 的此比值在所有扰动下均接近 0.1-0.2。

---

## 关键图表

### Figure 1: SWM 环境套件概览

![Figure 1 - PushT default](https://arxiv.org/html/2602.08968v2/x1.png)
![Figure 1 - PushT variation](https://arxiv.org/html/2602.08968v2/x2.png)
![Figure 1 - TwoRoom default](https://arxiv.org/html/2602.08968v2/x3.png)
![Figure 1 - TwoRoom variation](https://arxiv.org/html/2602.08968v2/x4.png)
![Figure 1 - DMC Humanoid default](https://arxiv.org/html/2602.08968v2/x5.png)
![Figure 1 - DMC Humanoid variation](https://arxiv.org/html/2602.08968v2/x6.png)
![Figure 1 - OGBench Scene default](https://arxiv.org/html/2602.08968v2/x7.png)
![Figure 1 - OGBench Scene variation](https://arxiv.org/html/2602.08968v2/x8.png)

**说明**: 展示 SWM 支持的多样化环境，包含 2D/3D 操控、导航、经典控制任务。每列为默认设置与变异设置对比，体现 FoV 框架的视觉效果。

### Figure 2: 全部 16 个环境可视化

（见 Appendix D，Figure 2 为 x9.png～x40.png 共 32 张图，展示所有环境的默认与变异配置）

### Table 1: 与现有代码库对比

| 特性 | PLDM | DINO-WM | **SWM（本文）** |
|------|------|---------|--------------|
| 文档质量 | 无 | 低 | **完整** |
| 基线实现数量 | 0 | 1 | **4** |
| 测试覆盖率 | 0% | 0% | **73%** |
| 维护活跃度（近6月PR） | < 5 | < 5 | **99** |
| 类型检查 | 无 | 无 | **有** |
| 环境数量 | 2 | 2 | **16** |

**说明**: SWM 在所有工程质量指标上显著优于现有代码库，体现其对可复现性的重视。

### Table 2: DINO-WM 零样本鲁棒性结果（PushT 环境）

| 扰动类别 | 扰动对象 | 成功率 |
|---------|---------|--------|
| 无扰动（基线） | — | **94.0%** |
| 颜色变化 | Anchor | 20% |
| 颜色变化 | Agent | 18% |
| 颜色变化 | Block | 18% |
| 颜色变化 | Background | 10% |
| 大小变化 | Anchor | 14% |
| 大小变化 | Agent | 4% |
| 大小变化 | Block | 16% |
| 角度变化 | Agent | ~12% |
| 角度变化 | Block | ~12% |
| 位置变化 | Anchor | 4% |
| 形状变化 | Agent | 18% |
| 形状变化 | Block | 8% |
| 速度变化 | Agent | 14% |

**关键发现**: [[DINO-WM]] 在默认设置下达到 94.0% 成功率，但在**所有类型的扰动下均骤降至 4-20%**，说明基于 [[DINOv2]] 的预训练视觉特征并不能天然提供对环境变化的鲁棒性。此外，当目标图像来自随机策略轨迹时（而非专家演示），成功率也从 94% 骤降至 12%，揭示评估数据来源对结果的巨大影响。

### Table 3: 全部环境 FoV 数量

| 环境 | FoV 数量 |
|------|---------|
| PushT-v1 | 16 |
| TwoRoom-v1 | 17 |
| OGBench-Scene-v1 | 12 |
| OGBench-Puzzle-v1 | 11 |
| OGBench-其他变体 | 11-12 |
| DMControl-Humanoid | 10 |
| DMControl-Cheetah | 8 |
| DMControl-其他任务 | 6-9 |

---

## 实验

### 数据集 / 环境

| 环境 | 类型 | 用途 |
|------|------|------|
| PushT-v1 | 2D 操控 | DINO-WM 鲁棒性主实验 |
| TwoRoom-v1 | 2D 导航 | 通用评估 |
| DMControl Suite | 3D 经典控制 | 多样性测试 |
| OGBench | 3D 机器人操控 | 复杂任务测试 |

### 实现细节（DINO-WM 复现）

- **训练框架**: PyTorch + stable-pretraining 库
- **训练轮数**: 20 epochs（与原始 DINO-WM 超参保持一致）
- **规划求解器**: [[CEM]]，50 步预算（为最小成功步数的 2 倍）
- **评估环境**: PushT-v1，在 16 个 FoV 维度上系统扰动

### 关键发现

1. **评估数据来源至关重要**: 用专家演示数据作为目标 → 94% 成功率；用随机策略数据作为目标 → 12% 成功率，差距悬殊
2. **预训练视觉特征不具备鲁棒性**: [[DINOv2]] 特征在训练时未见过的视觉变化下完全失效，打破了"预训练特征具有通用性"的直觉
3. **颜色扰动影响大**: 背景颜色变化导致最低成功率（10%），说明模型对视觉背景有强依赖

---

## 批判性思考

### 优点

1. **工程质量极高**: 73% 测试覆盖率、完整类型检查、活跃维护，在研究代码库中罕见
2. **发现了重要 insight**: DINO-WM 的鲁棒性分析揭示了现有 WM 评估的系统性偏差（数据来源问题）
3. **设计哲学清晰**: World 类的解耦设计（策略 vs 环境）既符合工程最佳实践，也便于研究者替换组件

### 局限性

1. **研究深度有限**: 作为基础设施论文，DINO-WM 实验是唯一的用例演示，WM 方法比较不够系统
2. **物理仿真环境缺失**: 16 个环境中缺乏真实物理感强的场景（作者自认是未来工作方向）
3. **无真实世界环境**: 所有环境均为仿真，Sim-to-Real 迁移性未探讨

### 潜在改进方向

1. 接入 Hugging Face 标准化 benchmark，支持结果自动上报与排行榜
2. 增加调试和可视化工具，便于诊断 WM 失败原因
3. 引入真实世界数据集接口，支持 Sim-to-Real 研究

### 可复现性评估

- [x] 代码开源（基础设施本身即开源）
- [ ] 预训练模型（未提供 DINO-WM 复现权重）
- [x] 训练细节完整（附录有完整超参）
- [x] 数据集可获取（通过 SWM API 生成）

---

## 关联笔记

### 基于

- [[DINO-WM]]: 被复现和分析的世界模型方法，本文鲁棒性实验的对象
- [[DINOv2]]: DINO-WM 使用的预训练视觉骨干
- [[PLDM]]: 另一个被对比的现有代码库
- [[Gymnasium]]: SWM 封装的底层环境接口

### 对比

- [[PLDM]]: 代码质量对比的参照
- [[DINO-WM]]: 复现验证和鲁棒性对比

### 方法相关

- [[世界模型]]: 核心研究对象
- [[MPC]]: SWM 内置的规划框架
- [[CEM]]: SWM 支持的 MPC 求解器之一
- [[MPPI]]: SWM 支持的另一种 MPC 求解器
- [[零样本鲁棒性]]: 主要评估维度
- [[OGBench]]: SWM 包含的机器人操控环境

### 硬件/数据相关

- [[OGBench]]: 机器人操控 benchmark 环境
- [[DMControl]]: DeepMind 经典控制套件

---

## 速查卡片

> [!summary] stable-worldmodel-v1
> - **核心**: 首个标准化世界模型研究基础设施，16 环境 + 可控 FoV + 4 基线
> - **方法**: 模块化 World 类 + 策略解耦 + MPC 规划 + 在线/离线评估
> - **结果**: DINO-WM 在所有视觉扰动下成功率从 94% 降至 4-20%，揭示预训练视觉特征缺乏鲁棒性
> - **代码**: [arXiv](https://arxiv.org/abs/2602.08968)

---

*笔记创建时间: 2026-05-31*
