---
type: concept
aliases: [Action Dynamic Manifold, 动作动力学流形]
---

# ADM（Action Dynamic Manifold）

## 定义

ADM 是 [[PRIME W0]] 提出的一种流匹配动作解码器，核心创新在于用从演示数据统计的协方差结构化噪声替代标准各向同性高斯噪声，使初始采样点反映机器人运动学耦合的内在流形。

## 数学形式

协方差估计与 Cholesky 分解：

$$
\hat{\Sigma} = \frac{1}{N} \sum_{i=1}^{N} (a^{(i)} - \bar{a})(a^{(i)} - \bar{a})^T, \quad \hat{\Sigma} = LL^T
$$

推理时结构化噪声采样：

$$
a_0 = L\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

流匹配目标：

$$
\mathcal{L}_{FM} = \mathbb{E}_{\tau, a_0, a_1}\Big[\| v_\theta(a_\tau, \tau, c) - (a_1 - a_0) \|^2\Big]
$$

## 核心要点

1. **协方差结构化噪声**: 离线从 100 万动作块统计协方差，捕捉双臂机器人关节耦合约束
2. **Cholesky 采样**: $a_0 = L\epsilon$ 使噪声起点落在真实运动学流形附近，缩短流场估计路径
3. **6 层 Transformer 解码器**: 512 维隐状态，8 注意力头，8 步 Euler ODE 推理
4. **动作块输出**: H=16，执行窗口 $h_{exec}=8$，20 维双臂笛卡尔命令

## 代表工作

- [[PRIMEW0]]: ADM 的提出论文，在 LIBERO（99.1%）和真机（78.1%）达到 SOTA

## 相关概念

- [[Flow Matching]]: ADM 的基础算法框架
- [[Action Chunking]]: ADM 输出的动作预测范式
- [[MoE]]: ADM 与 Adaptive MoE 配合使用
