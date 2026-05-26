---
type: concept
aliases: [Model Predictive Path Integral, 模型预测路径积分]
---

# MPPI

## 定义
Model Predictive Path Integral，一种基于随机采样的 MPC 方法，通过采样大量控制序列在世界模型中 rollout，用 cost 加权平均更新控制输入，不依赖梯度。

## 数学形式
重要性加权更新：
$$u_t^* = \frac{\sum_k w_k \cdot \delta u_t^{(k)}}{\sum_k w_k}$$
$$w_k = \exp\left(-\frac{1}{\lambda} S(\tau^{(k)})\right)$$

其中 $S(\tau)$ 为轨迹代价，$\lambda$ 为温度参数。

## 核心要点
1. 在每个时间步采样 $K$ 条扰动控制序列 $\{u + \delta u^{(k)}\}$
2. 用世界模型 rollout 每条序列，计算代价 $S$
3. 指数加权平均所有扰动，更新基准控制序列
4. 不需要代价函数可微，适用于非光滑代价
5. 并行采样可在 GPU 上高效实现

## 代表工作
- [[Dream-MPC]]: 对比 MPPI，改用梯度式 MPC 在潜空间优化
- [[Model Predictive Control]]: MPPI 是 MPC 的一个变体

## 相关概念
- [[Model Predictive Control]]
- [[RSSM]]
- [[强化学习]]
