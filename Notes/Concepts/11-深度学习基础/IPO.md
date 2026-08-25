---
type: concept
aliases: [Identity Preference Optimization, Identity PO]
---

# IPO

## 定义
Identity Preference Optimization，一种偏好优化训练方法，作为 DPO 的变体，通过 identity 映射简化损失函数，避免 DPO 对参考模型分布的过度依赖。

## 数学形式
$$\mathcal{L}_\text{IPO}(\pi_\theta) = \mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[\left(h_\theta(x, y_w, y_l) - \frac{1}{2\beta}\right)^2\right]$$

其中 $h_\theta = \log\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}$

## 核心要点
1. 不依赖参考模型，简化训练流程
2. 相比 DPO 更稳定，对 over-fitting 更鲁棒
3. 常与 SFT 组合用于约束对齐

## 代表工作
- [[Logic-VLA]]: 用 SFT + IPO 对齐 STL 形式化约束

## 相关概念
- [[GRPO]]
- [[RLHF]]
