---
type: concept
aliases: [Hierarchical Multi-Expert Fusion, 层次化多专家融合规划器, HMEF Planner]
---

# HMEF

**Hierarchical Multi-Expert Fusion Diffusion Planner（层次化多专家融合扩散规划器）**

## 定义

一种双流扩散 Transformer 规划器，以多路专家 Token 为条件联合去噪，通过可学习融合权重 $\alpha_e$ 聚合各专家轨迹预测，在推理时将多源世界知识转化为一致的自车轨迹。

## 数学形式

**双流去噪**:

$$
X_0 = D_\theta(C, R, e_\tau)
$$

**专家融合**:

$$
\bar{A} = \sum_{e=1}^{N_e} \alpha_e \hat{A}_e, \quad \alpha_e = \operatorname{softmax}(w_e)
$$

**联合损失**:

$$
\mathcal{L}_{act} = \mathcal{L}_{diff} + \lambda_{fusion} \|\bar{A} - A^{norm}\|_2^2
$$

## 核心要点

1. **双流架构**: 干净场景 Token 流 $C$ 与带噪专家动作 Token 流 $R$ 通过联合 Self-Attention 交互
2. **每路专家独立预测**: 各专家轨迹 $\hat{A}_e$ 独立监督，再通过 softmax 权重 $\alpha_e$ 加权融合
3. **10 步去噪即可**: 相比生成模型通常需要更多步，规划任务 10 步达到最优，多步反而略降
4. **推理时融合**: 专家权重在训练中优化，推理时固定聚合，不增加推理延迟

## 代表工作

- [[CoWorld-VLA]]: 将 HMEF 用于自动驾驶轨迹规划，NAVSIM v1 PDMS 提升 +1.3（88.7→90.0）

## 相关概念

- [[Diffusion Model]]: 底层去噪机制
- [[Latent Chain-of-Thought]]: 提供多专家条件 Token 的推理框架
- [[Flow Matching]]: HMEF 训练使用 logit-normal 噪声调度，思路类似
- [[Multi-Expert Training]]: Stage 2 为 HMEF 提供训练好的专家 Token
