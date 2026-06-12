---
type: concept
aliases: [MSE损失, Mean Squared Error, 均方误差损失, MSE Loss]
---

# MSE 损失（均方误差）

## 定义
均方误差（Mean Squared Error, MSE）是预测值与真实值之差的平方均值，是回归任务和连续特征重建的标准损失函数。

## 数学形式

$$
\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} \|\hat{y}_i - y_i\|^2
$$

## 核心要点
1. 对大误差惩罚更重（平方放大效应），适合需要精确对齐的任务
2. 在视觉潜码重建中用于衡量特征空间的重建质量
3. 相比 L1 损失，MSE 对异常值更敏感但梯度更平滑

## 代表工作
- [[AffordanceVLA]]: Which2Act 使用 MSE 损失监督目标物体视觉潜码重建

## 相关概念
- [[均方误差]]
- [[Smooth L1损失]]
- [[二值交叉熵]]
