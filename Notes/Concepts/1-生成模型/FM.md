---
type: concept
aliases: [Flow Matching, 流匹配]
---

# FM（Flow Matching）

## 定义
一种基于连续归一化流的生成模型训练方法，通过回归从噪声到数据的概率流 ODE 向量场，避免了 DDPM 的多步采样开销。

## 数学形式
$$\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, x_1} \| v_\theta(x_t, t) - (x_1 - x_0) \|^2$$

其中 $x_t = (1-t)x_0 + t x_1$，$v_\theta$ 为参数化向量场。

## 核心要点
1. 相比 DDPM 的 score matching，FM 直接回归条件向量场，训练更稳定
2. 推理可用更少的 NFE（函数评估次数）完成生成
3. Conditional FM（CFM）扩展到条件生成，是 policy/action 生成的常用框架

## 代表工作
- [[π0]]: 用 FM 训练机器人扩散策略
- [[ODE-Policy]]: 把 action dynamics 建模为 FM

## 相关概念
- [[DiT]]
- [[Diffusion Policy]]
- [[DDPM]]
