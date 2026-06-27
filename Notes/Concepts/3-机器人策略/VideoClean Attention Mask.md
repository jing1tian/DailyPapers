---
type: concept
aliases: [VideoClean, VideoClean 掩码, VideoClean 注意力掩码]
---

# VideoClean Attention Mask

## 定义
[[Tactile Asymmetric Attention Mechanism (TAAM)|TAAM]] 的第一个组件，一种无需学习参数的注意力掩码规则：阻断视频 query 对触觉 key/value 的访问，同时保留动作 query 对触觉 token 的访问，用于防止"[[Tactile Pollution|触觉污染]]"。

## 数学形式

$$
M^{\mathrm{vc}}_{q,k}=
\begin{cases}
-\infty, & G(q)=V,\; G(k)=\tau, \\
0, & \text{otherwise},
\end{cases}
$$

叠加到基础注意力偏置：$\bar{B}_{q,k}=B^{0}_{q,k}+M^{\mathrm{vc}}_{q,k}$

## 核心要点
1. **方向性**：只阻断 "video→tactile"，不阻断 "action→tactile" 或 "tactile→tactile"
2. 无学习参数，纯粹是结构化掩码规则，可直接叠加在骨干网络原有的因果/分块/局部注意力规则之上
3. 实验验证（Tactile-WAM Table I）：在 5.5K checkpoint 上，VideoClean 使 RGB MSE 降低 4.42%，PSNR 提升 0.14 dB，SSIM 提升 0.0017
4. 单独使用 VideoClean（无触觉感知偏置）只能让成功率从 1.3%（Naive VT-WAM）恢复到 4.9%，仍接近 RGB-only 基线（5.8%）——说明"保护视频"必要但不充分

## 代表工作
- [[Tactile-WAM]]: 首次提出该掩码规则，并系统验证其对触觉污染的缓解效果

## 相关概念
- [[Tactile Asymmetric Attention Mechanism (TAAM)]]
- [[Touch-Aware Bias]]
- [[World Action Model]]
