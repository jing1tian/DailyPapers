---
type: concept
aliases: [Gram Matrix Alignment, Gram Matrix Loss, 格拉姆矩阵对齐, Gram矩阵]
---

# Gram 矩阵对齐（Gram Matrix Alignment）

## 定义

通过约束特征矩阵的 Gram 矩阵（特征内积矩阵）与目标网络对齐，隐式匹配特征间的相关结构（风格/关系一致性），而无需逐像素或逐 token 对齐。

## 数学形式

设学生特征矩阵 $Q \in \mathbb{R}^{M \times d}$，教师特征矩阵 $S \in \mathbb{R}^{M \times d}$，Gram 矩阵对齐损失为：

$$
\mathcal{L}_{\text{Gram}} = \frac{1}{M^2} \|S S^\top - Q Q^\top\|_1
$$

其中 $S S^\top \in \mathbb{R}^{M \times M}$ 编码 token 间的特征相关结构，$\|\cdot\|_1$ 为元素级 L1 范数。

## 核心要点

1. **结构对齐而非点对齐**: Gram 矩阵捕获特征间的相对关系（哪些 token 语义相似），比逐 token MSE 更鲁棒于绝对位置偏移
2. **起源于风格迁移**: 经典神经风格迁移中，Gram 矩阵用于匹配图像风格纹理统计量
3. **时序扩展**: 用于约束视频帧间对象特征的关系一致性，防止物体状态随时间漂移
4. **计算高效**: 无需对齐具体 token 位置，只需对整个特征矩阵做内积

## 代表工作

- [[DreamX-Phi]]: 用冻结 [[V-JEPA]] 教师的 Gram 矩阵约束操作视频中的对象关系一致性，配合门控机制过滤不可靠 batch

## 相关概念

- [[V-JEPA]]: DreamX-Phi 中提供 Gram 矩阵对齐目标的教师网络
- [[知识蒸馏]]: Gram 矩阵对齐是一种结构层面的知识蒸馏形式
