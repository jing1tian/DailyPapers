---
type: concept
aliases: [精化需求场, RDF, Spatiotemporal Refinement Field]
---

# Refinement Demand Field（精化需求场）

## 定义
一种用于 [[WAM|World-Action Model]] 推理加速的统一时空分配框架，通过学习"还需要多少精化"来自适应决定何时提交（WHEN）、推进多远（HOW FAR）、刷新哪些区域（WHERE）的去噪资源。

## 数学形式

**局部需求曲线**（组 $g$ 关于归一化去噪进度 $z$ 的分布）：

$$
q_g(z \mid s) = a_g(s)\,\mathrm{Beta}(z;\,m_g(s),\,c_g(s))
$$

**共享时序基函数**（所有组共用同一形状）：

$$
q_g(z \mid s) = a_g(s)\,B_0(z \mid s), \quad B_0 = \mathrm{Beta}(z;\,m_0,c_0)
$$

**全局需求场**：

$$
Q(z \mid s) = D(s)\,B_0(z \mid s), \quad D(s) = \sum_g \frac{n_g}{N_F}\,a_g(s)
$$

**组幅度（[[Stop-Gradient]] 解耦）**：

$$
a_g(s) = \mathrm{sigmoid}(b_0(s) + \mathrm{stopgrad}[\Delta u_g(s)])
$$

## 核心要点
1. **三维统一决策**：WHEN（继续概率 $D(s)$ 阈值触发提交）、HOW FAR（$B_0$ 众数映射到去噪步长）、WHERE（贪心 top-K 选择 $K_0=96$ token）
2. **参数解耦**：共享偏移 $b_0$ 控制总需求，相对优先级 $\Delta u_g$ 控制空间分布，通过 [[Stop-Gradient]] 独立学习
3. **三阶段动作接地训练**：依次学习时序分布（冻结骨干）→ 空间优先级（冻结时序）→ 总体需求（整合），每步以动作误差为监督

## 代表工作
- [[ElasticWAM]]: 提出 Refinement Demand Field，在 LIBERO-Plus、RoboCasa-GR1 等 benchmark 上实现 SOTA 性能，延迟降低 32.2%

## 相关概念
- [[WAM]]: 所适用的基础框架（World-Action Model）
- [[Fast-WAM]]: 固定预算的部分去噪方法（Refinement Demand Field 的改进对象）
- [[Beta Distribution]]: 需求曲线的概率分布形状
- [[Stop-Gradient]]: 解耦参数化的关键技巧
- [[Straight-Through Estimator]]: WHERE 阶段可微分 top-K 选择
