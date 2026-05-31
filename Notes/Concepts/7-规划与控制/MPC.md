---
type: concept
aliases: [Model Predictive Control, 模型预测控制]
---

# MPC

## 定义
模型预测控制（MPC）是一种基于系统动力学模型在有限时域内滚动优化控制输入序列的闭环控制方法，每步执行最优序列的第一个动作后重新规划。

## 数学形式
$$
\min_{u_0, \dots, u_{T-1}} \sum_{t=0}^{T-1} \ell(x_t, u_t) + \ell_T(x_T)
$$
$$
\text{s.t.} \quad x_{t+1} = f(x_t, u_t), \quad x_0 = x_{\text{current}}, \quad u_t \in \mathcal{U}
$$

其中 $f$ 为预测模型，$\ell$ 为阶段代价，$T$ 为预测时域。

## 核心要点
1. **滚动时域**：每步只执行优化序列的第一个动作，然后以新观测重新规划，形成闭环反馈
2. **显式约束处理**：控制量、状态约束可直接编码在优化问题中
3. **模型依赖**：性能高度依赖预测模型 $f$ 的精度；WM + MPC 是 model-based RL 的重要范式
4. **计算开销**：每步需在线求解优化，对实时系统是瓶颈；常用近似（MPPI、iLQR）替代精确求解

## 代表工作
- [[MPPI]]: 基于采样的随机 MPC，适合非线性系统
- [[stable-worldmodel-v1]]: 用 WM 作为 $f$，在 PushT / MuJoCo 上做 MPC 规划
- [[DreamerV3]]: 在 latent space 中做 MPC planning

## 相关概念
- [[MPPI]]
- [[MuJoCo]]
- [[MCTS]]
