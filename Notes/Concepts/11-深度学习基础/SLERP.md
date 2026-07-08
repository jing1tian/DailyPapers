---
type: concept
aliases: [球面线性插值, Spherical Linear Interpolation]
---

# SLERP（球面线性插值）

## 定义

在高维球面上对两个向量进行线性插值的方法，保留插值路径在球面上的等速性，适用于方向向量、四元数和语言 embedding 的平滑过渡。

## 数学形式

给定两个向量 $e_1, e_2 \in \mathbb{R}^d$，先计算夹角：

$$
\theta = \arccos\left(\frac{e_1^T e_2}{\|e_1\| \|e_2\|}\right)
$$

对插值系数 $t \in [0,1]$，SLERP 插值为：

$$
\text{SLERP}(e_1, e_2, t) = \frac{\sin((1-t)\theta)}{\sin\theta} e_1 + \frac{\sin(t\theta)}{\sin\theta} e_2
$$

## 核心要点

1. **角速度恒定**：插值路径在球面上等速运动，不像线性插值（LERP）会减速穿过球面
2. **保持模长**：若 $\|e_1\| = \|e_2\|$，则插值结果模长不变
3. **优于 LERP**：在嵌入空间中语义方向的插值中，SLERP 更好地维持语义一致性

## 代表工作

- [[GigaWorld1]]: 用于长视频生成中文本 embedding 的渐进语义过渡，避免提示切换导致的外观突变

## 相关概念

- [[流匹配]]: 也在 embedding 空间中操作
- [[扩散模型]]: SLERP 常用于扩散模型的噪声或条件插值
