---
type: concept
aliases: [DMD, 分布匹配蒸馏, DMD蒸馏]
---

# Distribution Matching Distillation

## 定义

通过最小化真实数据分布与生成数据分布之间的 Score 函数差异，将多步扩散模型蒸馏为少步（如 4 步）快速推理学生模型的训练方法。

## 数学形式

$$
\nabla \mathcal{L}_{DMD} = -\mathbb{E}_t \left( \int (s_{real}(x_t, t) - s_{fake}(x_t, t)) \frac{dx_t}{d\theta} \, dz \right)
$$

其中 $s_{real}$ 为冻结教师模型的 Score 函数，$s_{fake}$ 为可训练学生模型的 Score 函数，$\theta$ 为学生模型参数。

## 核心要点

1. 同时训练真实分布 Score 网络（冻结）和生成分布 Score 网络（可训练）
2. 通过分布差异梯度直接优化生成器，无需逐步去噪
3. 可将 50+ 步扩散过程压缩至 4 步，推理速度大幅提升
4. 与 Consistency Distillation、DDIM 等方法不同，从分布匹配角度建模

## 代表工作

- [[HYWorld2]]: WorldStereo 2.0 使用 DMD 蒸馏为 4 步推理模型

## 相关概念

- [[Video Diffusion Transformer]]
- [[1-生成模型/Score Distillation Sampling|Score Distillation Sampling]]
