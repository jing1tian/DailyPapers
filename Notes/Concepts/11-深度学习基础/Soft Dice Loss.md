---
type: concept
aliases: [软 Dice 损失, Dice Loss, Sørensen–Dice Loss]
---

# Soft Dice Loss

## 定义

Soft Dice Loss 是基于 Dice 系数（Sørensen–Dice similarity）的分割损失函数，通过直接最大化预测与真实掩码的重叠区域，解决类别不平衡问题，广泛用于语义分割和目标掩码预测。

## 数学形式

$$
\mathcal{L}_{Dice} = 1 - \frac{2 \sum_i p_i g_i}{\sum_i p_i + \sum_i g_i + \epsilon}
$$

其中 $p_i$ 为第 $i$ 个像素的预测概率，$g_i$ 为真实标签，$\epsilon$ 为防止除零的平滑项。

常与 BCE 组合使用：

$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{BCE} + (1-\alpha) \cdot \mathcal{L}_{Dice}
$$

## 核心要点

1. **类别不平衡鲁棒**: 不受背景像素数量主导，更关注目标区域的形状覆盖
2. **区域级优化**: BCE 优化像素级概率，Dice 优化区域重叠度，两者互补
3. **"Soft"含义**: 使用预测概率（非离散化二值标签）直接计算，支持端到端梯度回传
4. **常见组合权重**: $\alpha = 0.5$（如 TrajWAM 的遮罩预测器）

## 代表工作

- [[TrajWAM]]: 以 0.5 权重组合 BCE + Soft Dice 训练物体遮罩序列预测器

## 相关概念

- [[Binary Cross-Entropy]]: 常与 Soft Dice Loss 组合使用的像素级损失
- [[SAM]]: 语义分割模型，输出掩码可作为 Soft Dice Loss 的训练目标
