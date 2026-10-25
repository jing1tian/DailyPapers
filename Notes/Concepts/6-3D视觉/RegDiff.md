---
type: concept
aliases: [Registration Diffusion, 扩散位姿估计]
---

# RegDiff

## 定义
RegDiff：基于扩散模型的 6D 物体位姿估计方法，通过迭代去噪过程生成位姿分布，适合处理多假设（对称物体等）情形。

## 数学形式
$$p_\theta(\mathbf{R}_0 | \mathbf{x}) = \int p_\theta(\mathbf{R}_0 | \mathbf{R}_T, \mathbf{x}) \prod_{t=1}^{T} p_\theta(\mathbf{R}_{t-1} | \mathbf{R}_t, \mathbf{x}) d\mathbf{R}_{1:T}$$

在 SO(3) 上做扩散，生成旋转矩阵的后验分布。

## 核心要点
1. 将位姿估计转化为条件去噪问题，天然支持多模态位姿分布
2. 需要多步推理（候选生成 + 选择），推理延迟高于单步方法
3. 在 F2G-Pose 中被作为对比 baseline，后者以单次前向 28 FPS 优于 RegDiff

## 代表工作
- [[F2G-Pose]]: 对比 baseline，单步方法在速度上超越 RegDiff

## 相关概念
- [[FoundationPose]]: 同领域方法
- [[F2G-Pose]]: 对比方法
