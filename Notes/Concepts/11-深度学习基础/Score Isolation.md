---
type: concept
aliases: [分数隔离, 评估隔离, score-isolation]
---

# Score Isolation（分数隔离）

## 定义

在推理时自适应系统中，**分数隔离**指持出（held-out）评估结果对系统的控制状态、路由决策和更新接受标准保持**只读**（read-only）的形式化约束，防止测试集分数回流影响已部署系统。

## 数学形式

$$
\Omega_R = f(D_0, D_1, \ldots, D_{R-1}), \quad \Omega_R \perp \mathcal{E}_{\text{held-out}}
$$

其中 $D_r$ 为开发集，$\mathcal{E}_{\text{held-out}}$ 为持出评估集，两者严格分离。

## 核心要点

1. **三分离原则**: 区分"提议的更新"、"已部署的控制状态"和"持出评估分数"三个对象
2. **预承诺路由**: 评估前锁定完整路由函数 $\Phi_{\Omega_R}$，评估期间不接受任何状态更新
3. **可审计性**: 通过溯源绑定链（[[Provenance Binding]]）记录每次状态转移，任何违反分数隔离的操作均可被检测

## 代表工作

- [[SCOPE]]: 提出分数隔离原则并形式化实现于视频世界模型推理时自适应

## 相关概念

- [[Inference-Time Adaptation]]
- [[Provenance Binding]]
- [[Bootstrap Confidence Interval]]
