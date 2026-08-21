---
type: concept
aliases: [可演化封套, Evolvable Harness, 运行时封套, 封套]
---

# Evolvable Harness

## 定义

包裹在冻结基础策略外部的可演化运行时组件集合，通过累积 Critic-Recovery 对实现具身智能的无参数化扩展，无需修改基础模型权重。

## 数学形式

$$
\mathcal{H} = \{\mathcal{C},\; \mathcal{R},\; \mathcal{T}\}
$$

演化更新规则：

$$
\mathcal{H}^{(k+1)} \leftarrow \mathcal{A}_{evo}\!\left(\mathcal{D}_{fail}^{(k)},\; \mathcal{H}^{(k)}\right)
$$

## 核心要点

1. **策略解耦**: 基础策略 $\pi$ 参数完全冻结 ($\nabla\theta=0$)，所有改进通过 $\mathcal{H}$ 积累
2. **三组件结构**:
   - $\mathcal{C}$（Critics）：[[Runtime Critics|运行时批评器]]，高频监控物理状态
   - $\mathcal{R}$（Recovery Playbook）：结构化恢复方案库，按失败类型索引
   - $\mathcal{T}$（Tools）：异构工具集（视觉、物理信号、外部 API）
3. **累积扩展**: 物理智能通过 harness 版本的迭代积累而非模型重训实现扩展
4. **验证门控**: 新 Critic-Recovery 对必须通过历史回归测试和 held-out 泛化验证才能纳入

## 与 HarnessWAM 的关系

[[HarnessWAM]] 是早期探索运行时封套思想的工作；Evolvable Harness 在此基础上引入自动化演化循环和严格的验证门控机制。

## 代表工作

- [[Zetta]]: 提出 Evolvable Harness 概念，通过三时间尺度闭环实现自演化，LIBERO-Pro 提升 +56.3%

## 相关概念

- [[Runtime Critics]]
- [[Hierarchical Causal Diagnosis]]
- [[Closed-Loop Control]]
- [[HarnessWAM]]
