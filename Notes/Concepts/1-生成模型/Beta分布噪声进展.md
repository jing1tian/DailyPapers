---
type: concept
aliases: [Beta Distribution Noise Progression, Adaptive Noise Jumping, 自适应噪声跳进]
---

# Beta 分布噪声进展

## 定义

一种用于扩散/流匹配去噪过程的**自适应步长控制机制**，通过 [[Beta分布]] 对噪声保留比 $r_k \in (0,1)$ 建模，允许在不同去噪阶段动态决定下一步的噪声水平，而非使用固定等间距步长。

## 数学形式

$$
r_k \sim \text{Beta}(\alpha_k, \beta_k), \quad \sigma_{k+1} = r_k \sigma_k
$$

$$
\alpha_k = m_k(c_k - 2) + 1, \quad \beta_k = (1 - m_k)(c_k - 2) + 1
$$

$$
x_{k+1} = x_k + \hat{v}_\phi(x_k, \sigma_k, c)(\sigma_{k+1} - \sigma_k)
$$

## 核心要点

1. **相对而非绝对步长**：$\sigma_{k+1} = r_k \sigma_k$ 保证单调递减，$r_k$ 越小步长越大（跳过越多噪声层）
2. **众数-浓度参数化**：MLP 输出均值 $m_k$ 和浓度 $c_k$，训练时采样、部署时用众数确定性推理
3. **任意噪声水平间的更新**：配合[[流匹配]]一阶更新，可在非均匀噪声步上准确推进潜变量

## 代表工作

- [[SANTS]]: 首次将 Beta 分布步长控制用于 WAM 视频去噪的自适应加速

## 相关概念

- [[Beta分布]]
- [[流匹配]]
- [[累积风险函数]]
- [[扩散模型]]
