---
type: concept
aliases: [Conditional Flow Matching, Continuous Flow Matching, 条件流匹配]
---

# CFM

## 定义
一种基于 ODE 的生成式训练框架，通过学习从噪声分布到数据分布的连续速度场（velocity field），用比扩散模型更少的推理步数生成样本。

## 数学形式
$$\frac{dx}{dt} = v_\theta(x, t), \quad t \in [0, 1]$$

条件流匹配目标：
$$\mathcal{L}_{CFM} = \mathbb{E}_{t, x_0, x_1} \| v_\theta(x_t, t) - (x_1 - x_0) \|^2$$

## 核心要点
1. 基于 ODE 的生成，推理时比 DDPM 快（通常 1-10 步）
2. 条件版本 (CFM) 通过边缘化条件路径来训练无条件速度场
3. 与 [[Diffusion Policy]] 相比，Action 生成更平滑，推理延迟更低
4. ForesightFlow 将值函数作为势能场引导 CFM ODE 轨迹，实现 offline policy improvement

## 代表工作
- [[ForesightFlow]]: 势能引导的 CFM 用于 VLA 策略改进
- Lipman et al. (2022): Flow Matching for Generative Modeling（原始论文）

## 相关概念
- [[Diffusion Policy]]
- [[action chunking]]
- [[ODE]]
