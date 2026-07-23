---
type: concept
aliases: [Decomposed World Model, DWM]
---

# DWM（Decomposed World Model）

## 定义

**DWM** 是一个基于 [[LeWM]] 的潜在世界模型扩展框架，通过在训练阶段引入辅助 World Head，将预测状态转换分解为**动作驱动分量** $\hat{z}^a$ 与**动作不变的[[世界效应]]分量** $\hat{z}^w$，提升在含持续性环境动力学任务上的 [[Model Predictive Control|MPC]] 规划能力。

## 核心设计

$$
\hat{z}^w := h_w(r_t), \quad \hat{z}^a := \hat{z} - \hat{z}^w
$$

$$
\mathcal{L} = \mathcal{L}_{WM} + \lambda_{wc} \cdot \mathcal{L}_{wc} + \lambda_{orth} \cdot \mathcal{L}_{orth}
$$

- $\mathcal{L}_{wc}$：[[InfoNCE]] 世界对比损失，强制 $\hat{z}^w$ 动作不变
- $\mathcal{L}_{orth}$：[[正交约束]]，使 $\hat{z}^w \perp \hat{z}^a$
- 推理时丢弃 World Head，退化为标准 [[LeWM]]

## 核心要点

1. 训练专用分支，推理零开销
2. 两项损失缺一不可（见消融实验）
3. 最优超参：$\lambda_{wc}=0.3$，$\lambda_{orth}=0.5$
4. 在 PushT-W / Reacher-W / TwoRoom-W 上平均改善 +13.1 pp

## 代表工作

- [[DWM]]（论文笔记）：arXiv 2607.18715，Zhang et al., 2026

## 相关概念

- [[LeWM]]：基础骨干架构
- [[世界效应]]：分解目标
- [[动作不变表示]]：World Head 学到的表示类型
- [[InfoNCE]]：对比损失形式
- [[正交约束]]：解耦约束
