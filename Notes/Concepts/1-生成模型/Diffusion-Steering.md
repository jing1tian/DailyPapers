---
type: concept
aliases: [Noise-Space MPC, Diffusion Steering MPC]
---

# Diffusion Steering

## 定义
在 diffusion policy 的噪声空间（而非动作空间）中进行轨迹优化/MPC 规划的技术。通过在采样过程的中间噪声表示上施加 guidance，让规划器在能量景观中搜索最优噪声向量，再通过 denoising 过程解码为动作。

## 数学形式
$$\hat{z}^* = \arg\min_{z} \mathcal{L}(D_\theta(z), \text{goal}) \quad z \sim \mathcal{N}(0, I)$$

其中 $D_\theta$ 是 diffusion decoder，$\mathcal{L}$ 是在解码后动作空间定义的代价函数（如 MPC cost）。

## 核心要点
1. 相比在动作空间直接 MPC：噪声空间搜索受 diffusion prior 约束，不会产生不合理的 out-of-distribution 动作
2. 防止 MPC 利用 world model 的动力学误差（动作空间 MPC 的典型问题）
3. 与 Classifier Guidance 类似，但应用于机器人规划而非图像生成

## 代表工作
- [[LeWAM]]：首次将 diffusion steering 用于 WAM + MPC 组合中的动作规划
- 相关：Score Distillation Sampling（SDS）在图像域的类似思路

## 相关概念
- [[LeWAM]] — 应用实例
- [[LeWM]] — LeWAM 的前驱世界模型
- [[Diffusion Policy]] — 底层 diffusion action 生成框架
- [[MPC]] — 结合使用的规划框架
