---
type: concept
aliases: [Predictive Coding, 预测编码, PC]
---

# Predictive Coding

## 定义
一种神经计算框架，将感知和学习建模为预测误差最小化过程：高层神经元预测低层输入，低层神经元传递预测误差，学习通过最小化层级预测误差进行。

## 数学形式
$$\mathcal{L}_{PC} = \sum_l \|\hat{z}_l - z_l\|^2$$

其中 $\hat{z}_l$ 是第 $l$ 层的预测，$z_l$ 是实际激活值。

## 核心要点
1. 自顶向下预测 + 自底向上误差传播的双向信息流
2. 在机器人策略中：预测未来状态潜在表示 → 从预测状态解码动作
3. 计算效率高：预测器和策略解耦，可以用极轻量结构（<1M 参数）
4. 受神经科学启发，与 JEPA 类方法有深层联系

## 代表工作
- [[PredVLA]]: 将 Predictive Coding 用于 sub-million 参数 VLA 策略

## 相关概念
- [[JEPA]]
- [[LeWM]]
- [[Action-Conditioned World Model]]
