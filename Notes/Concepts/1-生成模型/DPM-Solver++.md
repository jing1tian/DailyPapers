---
type: concept
aliases: [DPM-Solver Plus Plus, DPM++]
---

# DPM-Solver++

## 定义

DPM-Solver++ 是一种针对扩散概率模型（Diffusion Probabilistic Models）的**高阶快速采样器**，通过分析求解扩散 ODE 实现在极少采样步数（通常 10-25 步）内生成高质量样本。

## 数学形式

基于指数积分器（Exponential Integrator）的高阶离散化：

$$
x_{t_{i-1}} = \frac{\alpha_{t_{i-1}}}{\alpha_{t_i}} x_{t_i} - \alpha_{t_{i-1}} \sum_{n=0}^{k-1} \phi_{n+1}(\lambda_{t_{i-1}}, \lambda_{t_i}) \hat{\epsilon}^{(n)}_\theta
$$

其中 $\phi_n$ 为指数积分系数，$\hat{\epsilon}^{(n)}_\theta$ 为高阶导数估计。

## 核心要点

1. **极少步数高质量生成**：10-25 步即可达到 DDPM 1000 步的质量
2. **无训练**：无需对预训练模型做任何修改，直接替换采样器
3. **高阶精度**：2 阶（DPM-Solver++(2M)）或 3 阶方法，比 DDIM 精度更高

## 代表工作

- [[RISE]]: Planner 模块使用 DPM-Solver++ 20 步生成候选轨迹

## 相关概念

- [[Diffusion Transformer]]
- [[1-生成模型]]
