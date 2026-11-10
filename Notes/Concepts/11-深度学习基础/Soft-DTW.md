---
type: concept
aliases: [Soft Dynamic Time Warping, 软动态时间规整]
---

# Soft-DTW

## 定义
Dynamic Time Warping (DTW) 的可微分变体，将 DTW 的 min 操作替换为 soft-min（通过温度参数 γ 控制平滑程度），使其可以作为可微分损失函数用于序列对齐。

## 数学形式
$$\text{DTW}_\gamma(A, B) = -\gamma \log \sum_{\mathbf{A} \in \mathcal{A}_{n,m}} \exp\left(-\frac{1}{\gamma} \langle \mathbf{A}, \Delta(A,B) \rangle_F\right)$$

其中 $\Delta(A,B)_{ij} = \delta(a_i, b_j)$ 是逐对距离矩阵，$\gamma \to 0$ 时退化为标准 DTW。

## 核心要点
1. 允许两个序列之间存在局部时间偏移而不惩罚，比 L2 对时序不对齐更鲁棒
2. 通过 soft-min 保持梯度可传播，可直接作为训练损失
3. 温度参数 γ 控制平滑度：γ 越大越平滑但越偏离原始 DTW

## 代表工作
- Cuturi & Blondel, NeurIPS 2017（原始论文）
- [[SURE]]：用于 VLA 训练中的时序不确定性建模，解决示教者执行节奏差异

## 相关概念
- [[DTW]] — 原始硬版本
- [[SURE]] — 应用 Soft-DTW 进行 VLA 策略学习
