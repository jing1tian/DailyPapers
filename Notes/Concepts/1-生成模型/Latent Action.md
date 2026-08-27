---
type: concept
category: 1-生成模型
aliases: [Latent Action, Implicit Action, 隐式动作嵌入]
tags: [action-conditioned, world-model, representation-learning]
created: 2026-06-05
---

# Latent Action（隐式动作嵌入）

## 定义

Latent Action 是一类动作条件视频世界模型中的条件表示方法，将机器人动作（关节角、末端执行器轨迹等）压缩为低维连续嵌入向量，隐式编码运动信息，而非直接使用可解释的几何表示。

## 核心要点

1. **优点**: 无需精确相机标定，可从纯视频数据中自监督学习动作表示；
2. **缺点**: 隐式嵌入在不同机器人本体间存在分布偏移，跨本体泛化能力弱；
3. 与骨架渲染（[[Skeleton Rendering|2D 骨架投影]]）相比：动作精度更低（PSNR 19.22 vs 23.48），但数据依赖更少。

## 数学形式

$$
a_\text{latent} = \text{Enc}_\phi(a_\text{raw}), \quad a_\text{latent} \in \mathbb{R}^d
$$

其中 $\phi$ 通过最小化视频预测误差端到端优化。

## 代表工作

- [[IRASim]]: 使用隐式动作嵌入的机器人视频世界模型；
- [[OSCAR]]: 对比实验证明骨架显式条件优于隐式动作（策略评估 Spearman ρ 从 0.643 提升至 0.750）；
- [[LAWA]]: 将离散隐动作序列作为"意图"代理，绕过未来视频生成实现高效 WAM 推理。

## 相关概念

- [[Forward Kinematics]]
- [[Classifier-Free Guidance]]
- [[Diffusion Transformer]]
