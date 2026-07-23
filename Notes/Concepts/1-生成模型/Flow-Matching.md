---
type: concept
aliases: [Flow Matching, Conditional Flow Matching, CFM, 流匹配]
---

# Flow-Matching

## 定义
一种生成模型训练范式，通过回归向量场而非 score function 来学习从噪声到数据的确定性流（ODE），无需 SDE 采样，推理更快。

## 数学形式
目标：学习向量场 $v_\theta(x_t, t)$ 使得：
$$\frac{dx_t}{dt} = v_\theta(x_t, t)$$
从 $x_0 \sim p_0$（噪声）积分到 $x_1 \sim p_1$（数据）。

训练 loss（Conditional Flow Matching）：
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

其中 $x_t = (1-t)x_0 + t x_1$ 为线性插值路径。

## 核心要点
1. 比 DDPM 推理步数少（ODE 比 SDE 更稳定），典型只需 10-20 步
2. 训练目标比 score matching 更简单：直接回归方向向量
3. 可以使用 Optimal Transport 路径（OT-CFM），减少轨迹弯曲
4. 在 robot action prediction 中替代 DDPM（[[Diffusion Policy]]），更快推理

## 代表工作
- [[GTA-VLA]]: 用异步 Flow-Matching 解耦推理和执行
- [[π0]]: 用 Flow-Matching 做机器人策略
- Lipman et al. 2022: 原始 Flow Matching 论文

## 相关概念
- [[DiT]]（常作 Flow-Matching 的 backbone 架构）
- [[Diffusion Policy]]（前身，基于 DDPM）
- [[CEM]]（规划时的搜索方法，不同范式）
