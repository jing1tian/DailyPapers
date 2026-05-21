---
type: concept
aliases: [stop-gradient, sg, 梯度截断, detach]
---

# Stop-Gradient

## 定义
在反向传播时阻断某一分支的梯度流动，使该分支作为固定目标（不更新参数），常用于自监督学习和一致性约束训练。

## 数学形式
$$
\mathcal{L} = \| f(x) - \text{sg}(g(x)) \|^2
$$
梯度只通过 $f(x)$ 分支反传，$g(x)$ 分支被固定，防止"捷径"解（两侧同时坍缩为常数）。

## 核心要点
1. 等价于 PyTorch 中的 `.detach()` 操作
2. 避免表示坍缩（representation collapse）：自监督学习中不能让两侧同时更新
3. 与 EMA（指数移动平均）目标网络配合使用效果更好

## 代表工作
- [[RoVLA]]: OC 损失中对干净预测施加 stop-gradient，防止一致性约束导致的平凡解
- SimSiam: 纯依赖 stop-gradient 实现自监督，无需负样本
- BYOL: stop-gradient + EMA 目标网络

## 相关概念
- [[对比学习]]
- [[EMA]]
- [[一致性正则化]]
