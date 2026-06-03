---
type: concept
category: Training
tags: [generative, flow-matching]
created: 2026-05-09
---

# Conditional Flow Matching

[[Flow Matching]] 的条件版本：学一个条件向量场 $\mathbf{u}_\theta(\mathbf{x}_\tau, \tau, \mathbf{c})$ 把噪声 $\epsilon$ 沿直线映射到数据 $\mathbf{x}$，目标 $\mathcal{L} = \|\mathbf{u}_\theta - (\mathbf{x} - \epsilon)\|^2$。

相比 score matching，回归目标更稳定、ODE 推理无随机性。

## 代表工作

- [[Pi0]] / [[Pi05]]: 动作头使用 CFM
- [[RLDX-1]]: 动作流与物理流均使用 CFM
- [[AHEAD]]: 用 CFM 建模潜空间中视觉 token 的时序动力学，实现动态物体操作预测
