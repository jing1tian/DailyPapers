---
type: concept
aliases: [GEM, Gradient Episodic Memory]
---

# GEM

## 定义

GEM（Gradient Episodic Memory）是一种持续学习（Continual Learning）算法，通过存储旧任务的少量样本作为 episodic memory，在学习新任务时约束梯度方向，防止遗忘已有技能。

## 数学形式

GEM 的约束条件：

$$\langle g, g_k \rangle \geq 0, \quad \forall k \in \mathcal{M}$$

即新任务的梯度 $g$ 不应与记忆中旧任务的梯度 $g_k$ 相冲突（内积非负）。当冲突发生时，用 QP 投影修正梯度：

$$\min_{\tilde{g}} \|\tilde{g} - g\|^2 \quad \text{s.t. } \langle \tilde{g}, g_k \rangle \geq 0$$

## 核心要点

1. **梯度投影**: 当新任务梯度与旧任务梯度冲突时，投影到满足约束的最近点
2. **Episodic Memory**: 为每个已学任务保留少量样本（通常每任务几百条）
3. **比较基线**: A-GEM（近似版本）降低计算成本；ER（经验回放）是更简单的替代
4. **VLA 应用**: 在 VLA 持续学习中作为标准 baseline（见 [[VLA-CL]]）

## 代表工作

- Lopez-Paz & Ranzato 2017: GEM 原始论文

## 相关概念

- [[经验回放]]
- [[课程学习]]
- [[Reinforcement Learning]]
