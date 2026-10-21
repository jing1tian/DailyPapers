---
type: concept
aliases: [空间 Softmax, SpatialSoftmax, 空间软最大值]
---

# Spatial Softmax

## 定义

Spatial Softmax 是一种将卷积特征图转化为 2D 关键点坐标的可微操作，通过对每个通道的特征图做 softmax 后计算加权期望坐标，保留空间位置信息并保持梯度流动。

## 数学形式

对通道 $c$ 的特征图 $F^c \in \mathbb{R}^{H \times W}$：

$$
p_{ij}^c = \frac{\exp(F_{ij}^c)}{\sum_{i'j'} \exp(F_{i'j'}^c)}
$$

$$
(x_c, y_c) = \sum_{ij} p_{ij}^c (i, j)
$$

其中 $(x_c, y_c)$ 是第 $c$ 个通道对应的"软"关键点坐标。

## 核心要点

1. **可微性**: 全过程可微，梯度可回传到 CNN 特征提取层
2. **位置编码**: 将高维特征图压缩为低维关键点坐标序列，信息损失小
3. **平移不变性破坏**: 有意保留空间位置信息，适合姿态/关节估计任务
4. **降维效果**: 将 $C \times H \times W$ 特征压缩为 $C \times 2$ 关键点坐标

## 代表工作

- [[EVA]]: 在逆动力学模型（IDM）中用 Spatial Softmax 提取帧的空间关键点，再经 MLP 预测关节动作；消融实验表明其带来 +0.21 精度提升

## 相关概念

- [[逆动力学模型]]
- [[卷积神经网络]]
- [[MLP]]
