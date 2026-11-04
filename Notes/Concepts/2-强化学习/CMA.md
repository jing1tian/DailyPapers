---
type: concept
aliases: [CMA-ES, Covariance Matrix Adaptation Evolution Strategy, 协方差矩阵自适应进化策略]
---

# CMA-ES（Covariance Matrix Adaptation Evolution Strategy）

## 定义
一种无梯度的黑盒优化算法，通过自适应更新参数空间的协方差矩阵来引导搜索方向，常用于 RL 策略参数搜索和行为优化。

## 数学形式
$$\theta_{t+1} = \theta_t + \alpha \cdot \sum_{i=1}^{\lambda} w_i (\theta_i^{(t)} - \theta_t)$$

协方差矩阵 $C$ 随迭代更新，捕捉参数间的相关性。

## 核心要点
1. 不需要梯度，适用于不可微目标函数
2. 自适应步长（step-size control）避免过早收敛
3. 在低维参数空间（< 1000 维）效率较高，高维时计算协方差矩阵开销大
4. 在 VLA behavioral steering 中用于少参数行为调节

## 代表工作
- [[BehaviorSteering-VLA]]: 用 CMA-ES 在低参数 regime 下调节 VLA 行为风格

## 相关概念
- [[PPO]]
- [[GRPO]]
- [[ES]]
