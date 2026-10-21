---
type: concept
aliases: [BCE, 二元交叉熵, Binary Cross-Entropy Loss, BCE Loss]
---

# Binary Cross-Entropy

## 定义

Binary Cross-Entropy（BCE）是用于二分类和像素级二值分割任务的损失函数，衡量预测概率分布与真实二值标签之间的交叉熵，是分割模型中最基础的逐像素监督信号。

## 数学形式

$$
\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\bigl[g_i \log p_i + (1 - g_i)\log(1 - p_i)\bigr]
$$

其中 $p_i \in (0,1)$ 为第 $i$ 个元素的预测概率（通常为 Sigmoid 激活输出），$g_i \in \{0, 1\}$ 为真实标签，$N$ 为总元素数。

## 核心要点

1. **像素级独立**: 每个像素独立计算损失，适合密集预测任务（分割、掩码预测）
2. **类别不平衡敏感**: 背景像素远多于目标像素时，BCE 容易被背景主导，需配合加权 BCE 或 Dice Loss
3. **常见组合**: $\mathcal{L} = 0.5 \cdot \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Dice}$，兼顾像素精度和区域覆盖
4. **数值稳定**: 实践中使用 BCEWithLogitsLoss（先 Sigmoid 后 BCE 融合），避免梯度消失

## 代表工作

- [[TrajWAM]]: 以 0.5 权重与 Soft Dice Loss 组合，训练物体遮罩序列预测器

## 相关概念

- [[Soft Dice Loss]]: 与 BCE 互补的区域级损失，两者组合效果优于单独使用
- [[SAM]]: 语义分割模型，其掩码解码器通常使用 BCE + Dice 损失
