---
type: concept
aliases: [表征推断, RD, Representational Deduction Framework]
---

# Representational Deduction

## 定义

Representational Deduction 是 PILOT 提出的一种方法论框架，通过将机器人策略分解为"先推断运动意图（Motion-CoT）、再以意图为条件生成动作轨迹"的两阶段结构，消除 [[World Action Model]] 中高层语义与低层轨迹的表征纠缠，并以 [[Causal Dynamics Engine]] 在表征空间（而非像素空间）监督意图 token 的物理正确性。

## 数学形式

策略分解为边缘化运动语义潜变量 $\mathbf{m}$：

$$
p_\Theta(\mathbf{a}_t | o_t, \ell, s_t) = \int p_\Theta(\mathbf{a}_t | \mathbf{m}, s_t) \cdot p_\Theta(\mathbf{m} | o_t, \ell, s_t) \, d\mathbf{m}
$$

总训练目标：

$$
\mathcal{L} = \mathcal{L}_{act} + \lambda_{wm}\,\mathcal{L}_{wm} + \lambda_{RD}\,\mathcal{L}_{RD}
$$

## 核心要点

1. **因果分解**: 先建模"状态应如何演变"（物理层），再建模"如何生成精细运动指令"（控制层）
2. **推理效率**: 推理时 Motion-CoT 直接作为中间表征可用，无需生成像素级未来帧
3. **泛化优势**: 意图解耦降低动作对视觉背景的依赖，从而提升鲁棒性和少样本迁移能力

## 代表工作

- [[PILOT]]: 首次系统提出 Representational Deduction 框架，包含 [[Motion-CoT]]、[[Causally-Decoupled Attention]] 和 [[Causal Dynamics Engine]] 三个核心组件

## 相关概念

- [[Motion-CoT]]
- [[Causal Dynamics Engine]]
- [[Causally-Decoupled Attention]]
- [[World Action Model]]
