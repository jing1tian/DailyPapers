---
type: concept
aliases: [Normalized Mean Square Error, 归一化均方误差]
---

# NMSE

## 定义
归一化均方误差（Normalized Mean Square Error）：均方误差除以参考值的方差，消除量纲影响，便于跨场景比较。

## 数学形式
$$\text{NMSE} = \frac{\text{MSE}}{\text{Var}(y)} = \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

## 核心要点
1. NMSE = 0 表示完美预测；NMSE = 1 表示预测等同于预测均值
2. 常用于评估动作/轨迹预测质量
3. 在 VLA 潜动作学习中用于衡量动作表示的保真度

## 代表工作
- [[MVP-LAM]]：用 NMSE 衡量跨视角潜动作一致性

## 相关概念
- [[均方误差]]
- [[Action Chunking]]
