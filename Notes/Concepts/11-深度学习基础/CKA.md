---
type: concept
aliases: [Centered Kernel Alignment, 中心化核对齐]
---

# CKA (Centered Kernel Alignment)

## 定义
CKA 是一种用于比较两个神经网络层（或不同模型）表示相似度的方法，通过核矩阵的对齐程度来衡量表示空间的一致性，对模型宽度和旋转变换不变。

## 数学形式
$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

其中 HSIC 是 Hilbert-Schmidt Independence Criterion，$K, L$ 是由样本特征计算的核矩阵。

## 核心要点
1. 对表示的线性变换（旋转、缩放）不变，比直接比较权重更鲁棒
2. Linear CKA（使用线性核）计算高效，RBF CKA 更通用
3. 常用于分析 task-vector arithmetic 操作对模型各层的影响
4. 值域 [0, 1]，1 表示表示完全一致

## 代表工作
- [[Suppression-Sticks]]: 用 CKA 分析 VLA 中 task-vector subtraction 对不同技能表示的影响

## 相关概念
- [[Task Vector]]
