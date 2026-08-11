---
type: concept
aliases: [CDE, 因果动态引擎, Causal Dynamics Predictor]
---

# Causal Dynamics Engine（CDE）

## 定义

Causal Dynamics Engine（CDE）是一种在 JEPA 表征空间（而非像素空间）预测物理状态演变的模块，以冻结的 [[V-JEPA2|VJEPA2-AC]] 编码器提取 patch 级表征为基础，以 [[Motion-CoT]] 和本体状态为条件，预测未来帧的表征向量，通过 [[Representational Deduction]] 损失进行监督。

## 数学形式

冻结编码器提取当前与未来表征（P 个 patch，$d_r$ 维）：

$$
\mathbf{r}_t = E_\psi(o_t), \quad \mathbf{r}_{t+\Delta} = E_\psi(o_{t+\Delta})
$$

线性投影映射到 VJEPA 潜空间：

$$
\tilde{\mathbf{m}} = \mathbf{W}_m \mathbf{m} \in \mathbb{R}^{K \times d_r}, \quad \tilde{\mathbf{s}} = \mathbf{W}_s s_t \in \mathbb{R}^{1 \times d_r}
$$

CDE 预测未来表征（256×1408 维）：

$$
\hat{\mathbf{r}}_{t+\Delta} = G_\xi(\mathbf{r}_t;\, \tilde{\mathbf{m}},\, \tilde{\mathbf{s}}) \in \mathbb{R}^{P \times d_r}
$$

[[Representational Deduction]] 损失（SmoothL1）：

$$
\mathcal{L}_{RD} = \text{SmoothL1}(\hat{\mathbf{r}}_{t+\Delta},\, \mathbf{r}_{t+\Delta})
$$

## 核心要点

1. **表征空间监督**: 在 JEPA 特征空间而非像素空间预测，天然忽略与任务无关的背景细节
2. **物理局部性**: PCA 可视化表明 CDE 激活集中于物体交互区域（抓取点、接触面），具备空间局部性
3. **训练时专用**: 推理时不需要未来帧 GT，CDE 的作用体现在强化 Motion-CoT 表征的物理感知

## 代表工作

- [[PILOT]]: CDE 是其 [[Representational Deduction]] 框架的核心组件之一

## 相关概念

- [[Motion-CoT]]
- [[V-JEPA2]]
- [[Representational Deduction]]
- [[World Action Model]]
