---
type: concept
aliases: [LDAD, Latent Difference Action Decoder, 潜在差分动作解码器]
---

# LDAD（潜在差分动作解码器）

## 定义

一种逆动力学解码器，仅从相邻潜变量之差 $\Delta z_t = z_{t+1} - z_t$ 重建执行动作 $\hat{a}_t$，而非从拼接的潜变量 $[z_t, z_{t+1}]$ 解码，是 [[Delta-JEPA]] 的核心模块。

## 数学形式

$$
\Delta z_t = z_{t+1} - z_t
$$

$$
\hat{a}_t = D_\Theta(\Delta z_t)
$$

$$
\mathcal{L}_\text{action} = \|\hat{a}_t - a_t\|_2^2
$$

多步扩展：

$$
\{\hat{a}_\tau\}_{\tau=t}^{t+N-1} = D_\Theta(z_{t+N} - z_t)
$$

## 核心要点

1. **防坍缩**：若编码器坍缩则 $\Delta z_t \to \mathbf{0}$，动作无法恢复，训练信号自然驱动编码器保持可区分性
2. **消除快捷依赖**：仅用位移向量，消除解码器利用 $z_t$ 绝对位置信息作弊的捷径
3. **动作敏感几何**：不同动作必须产生不同的 $\Delta z_t$，潜空间的转移方向本身承载动作语义
4. **实现**：3 层非因果 Transformer + [[AdaLN]] 注入位移，含 $N$ 个可学习 Action Query 支持多步解码

## 代表工作

- [[Delta-JEPA]]: LDAD 的提出论文，在四个视觉连续控制任务上全面超越 JEPA 基线

## 相关概念

- [[JEPA]]
- [[LeWM]]
- [[PLDM]]
- [[Representation Collapse]]
- [[AdaLN]]
