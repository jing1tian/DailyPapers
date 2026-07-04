---
type: concept
aliases: [Free-body Gaussian Velocity, Free Gaussian Velocity, 自由体高斯速度]
---

# FreeGave

## 定义
在 3D Gaussian Splatting 世界模型中，为每个 Gaussian 粒子附加速度和加速度状态，并施加刚体动力学约束，使粒子群体遵循物理运动规律（动量守恒、刚体不变形），从而实现动态目标的物理一致性预测。

## 数学形式
每个 Gaussian 粒子的状态扩展为：
$$
\mathcal{G}_i = \{\mu_i, \Sigma_i, \alpha_i, c_i, \mathbf{v}_i, \mathbf{a}_i\}
$$

状态传播遵循刚体动力学：
$$
\mu_i(t+\Delta t) = \mu_i(t) + \mathbf{v}_i \Delta t + \frac{1}{2}\mathbf{a}_i \Delta t^2
$$

刚体约束（物体内各粒子相对位置保持不变）：
$$
\|\mu_i(t) - \mu_j(t)\| = \|\mu_i(0) - \mu_j(0)\|, \quad \forall i,j \in \text{same object}
$$

## 核心要点
1. 标准 3DGS 只有静态几何表示，没有时序动力学；FreeGave 通过速度/加速度字段把物理动力学引入 Gaussian 表示
2. 与纯神经网络预测（如 video diffusion）相比，刚体约束大幅减少了物理不一致性（穿透、形变等）
3. 适用于快速运动目标的 manipulation（接球、拦截），不适用于柔性/流体目标
4. 可与 [[LoRA]] 结合将物理 world model 轻量级接入 VLA backbone

## 代表工作
- [[PhysMani]]: 首次提出 FreeGave 并用于动态目标操纵，在 FreeGave benchmark 和真实机械臂上验证

## 相关概念
- [[ManiGaussian]]
- [[3DGS]]
- [[World Action Model]]
- [[Flow Matching]]
