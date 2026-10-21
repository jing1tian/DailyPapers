---
type: concept
aliases: [非对称注意力, Asymmetric Cross-Attention]
---

# Asymmetric Attention

## 定义

Asymmetric Attention（非对称注意力）是一种多流注意力设计：不同模态/流的 token 具有不对称的注意力范围——部分流只与同类 token 自注意力，另一部分流则对所有 token 做交叉注意力，从而在保持某些流独立性的同时实现信息融合。

## 数学形式

以视频流 $V$ 和动作流 $A$ 为例：

$$
\begin{aligned}
V' &= \operatorname{Attn}(V, V) \quad \text{（视频流：仅自注意力）} \\
A' &= \operatorname{Attn}(A, [V; A]) \quad \text{（动作流：交叉注意力于全部 token）}
\end{aligned}
$$

## 核心要点

1. **单向信息流**: 视频预测不受动作 token 影响，保证视频生成质量独立于动作分支
2. **动作读取视频**: 动作 token 可充分利用受轨迹控制的视频表示来预测动作
3. **计算效率**: 避免视频-动作全双向注意力，降低计算量
4. **避免干扰**: 防止动作分支反向影响视频生成过程，维持视频表示稳定性

## 代表工作

- [[TrajWAM]]: 视频 token 做自注意力，动作 token 对视频+动作全部 token 做交叉注意力，联合 Flow Matching 训练

## 相关概念

- [[Flow Matching]]: 常与 Asymmetric Attention 联合使用的训练目标
- [[Action Expert]]: 通过 Asymmetric Attention 读取视频表示的动作预测模块
- [[World Action Model]]: 使用非对称注意力进行视频-动作联合建模的框架
