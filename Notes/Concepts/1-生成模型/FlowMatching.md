---
type: concept
aliases: [Flow Matching, 流匹配, Rectified Flow]
---

# Flow Matching

## 定义
Flow Matching 是一种训练连续归一化流（CNF）的方法，通过回归一个简单的目标向量场（而非 score function）来学习从噪声到数据的路径，避免了模拟 ODE/SDE 轨迹的高计算代价。

## 数学形式
$$\mathcal{L}_{FM} = \mathbb{E}_{t, p_t(x)} \| v_\theta(x, t) - u_t(x) \|^2$$

其中 $u_t(x)$ 是条件向量场，$v_\theta$ 是网络预测的向量场。推理时沿 $\frac{dx}{dt} = v_\theta(x,t)$ 积分。

## 核心要点
1. 比 DDPM 推理步骤少（直线路径），更快生成
2. Conditional Flow Matching (CFM) 通过引入条件 $x_1$ 使目标可计算
3. Rectified Flow 是一种特例：插值路径 $x_t = (1-t)x_0 + tx_1$
4. 广泛用于机器人 action head（如 FabriVLA、$\pi_0$）

## 代表工作
- [[Lipman2022]]: Flow Matching for Generative Modeling（原始论文）
- [[FabriVLA]]: flow-matching action head for manipulation
- [[FlowDAgger]]: 利用流匹配 ODE 可逆性进行潜噪声空间 DAgger 适应

## 相关概念
- [[DiT]]
- [[DDPM]]
- [[FlowUniPC]]
