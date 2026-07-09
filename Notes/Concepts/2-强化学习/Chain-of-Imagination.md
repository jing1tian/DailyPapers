---
type: concept
aliases: [CoI, 想象链, 链式想象推理]
---

# Chain-of-Imagination

## 定义

**Chain-of-Imagination**（CoI）是一种在世界模型潜空间中进行结构化推理的范式：将推理过程视为有序的动作-后果对序列（想象轨迹），实现深思熟虑的规划与多假设探索的统一。

## 数学形式

**信念编码**:

$$
\mathbf{z}_{t} = E_{\eta}(\mathbf{o}_{\leq t}, \mathbf{a}_{< t})
$$

**想象轨迹集合**:

$$
\mathcal{C}_{t} = \left\{ \left[ (\mathbf{a}_{b,1}, \widehat{\mathbf{z}}_{b,1|t}), \ldots, (\mathbf{a}_{b,K_b}, \widehat{\mathbf{z}}_{b,K_b|t}) \right] \right\}_{b=1}^{B_t}
$$

**潜状态演化**:

$$
\widehat{\mathbf{z}}_{b,k|t} = W_{\theta}(\widehat{\mathbf{z}}_{b,k-1|t}, \mathbf{a}_{b,k}), \quad \widehat{\mathbf{z}}_{b,0|t} = \mathbf{z}_{t}
$$

**策略决策**:

$$
\widehat{\mathbf{a}}_{t} = \pi_{\phi}\left(\mathbf{z}_{t}, \operatorname{Agg}_{\psi}(\mathcal{C}_{t})\right)
$$

## 核心要点

1. 在**潜空间**而非观测空间展开想象，计算代价远低于像素级展开
2. 支持**多分支并行想象**（$B_t$ 条分支），覆盖多种假设
3. 聚合函数 $\operatorname{Agg}_\psi$ 可学习，将多条想象轨迹压缩为策略输入
4. 与 Chain-of-Thought 的区别：CoI 在结构化潜状态空间中操作，而非文本 token 序列
5. 对应具身 AI 中的"在脑中预演行动"的认知机制

## 代表工作

- [[WorldModelRoadmap]]: 首次系统定义 CoI 框架

## 相关概念

- [[POMDP]]
- [[World Model]]
- [[World Action Model]]
- [[MBRL]]
