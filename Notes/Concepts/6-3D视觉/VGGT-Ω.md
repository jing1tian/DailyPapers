---
type: concept
aliases: [VGGT-Omega, VGGT Omega, Geometry-Aware Multi-View Encoder]
---

# VGGT-Ω

## 定义

[[VGGT]] 的扩展版本，支持在单次前向传播中联合聚合多个摄像头视角的观测，并显式建模跨视角几何关系（视觉对应、相机相对位置、场景几何）。

## 核心要点

1. **联合多视角聚合**: 在每个时间步将所有视角图像一起输入，而非独立编码后拼接，使 attention 机制在 forward pass 中跨视角传递几何信息。
2. **双类型输出 tokens**: 输出两类 token——**register tokens** $R$（全局几何上下文，跨视角信息汇总）和**patch tokens** $P$（各视角局部特征）。
3. **冻结预训练使用**: 在下游任务（如 VLA 训练）中通常保持冻结（❄️），利用其在大规模多视角数据上习得的几何先验。

## 数学形式

$$
\{R_t, P_t\} = E_\Omega\left(\{I_t^v\}_{v=1}^V\right)
$$

其中 $I_t^v$ 为时间步 $t$、视角 $v$ 的图像，$V$ 为总视角数，$R_t$ 为 register tokens，$P_t$ 为 patch tokens。

## 代表工作

- [[GWM-VLA]]: 将冻结 VGGT-Ω 作为几何感知编码器，引入潜在世界模型 VLA 框架

## 相关概念

- [[VGGT]]: 基础模型（VGGT-Ω 的前身）
- [[Latent Predictive World Model]]: VGGT-Ω 输出 tokens 作为世界模型的输入状态
