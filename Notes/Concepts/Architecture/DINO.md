---
type: concept
aliases: [DINO, DINOv2, Self-supervised Vision Transformer, 自监督视觉表示]
---

# DINO / DINOv2

## 定义

DINO（Self-DIstillation with NO labels）是一种基于自蒸馏的自监督视觉表示学习方法，通过无标签数据训练 ViT 编码器，学习到语义丰富的特征空间。DINOv2 是其改进版本。

## 数学形式

DINO 特征相似度（用于 WAM 世界建模评估）：

$$
\text{DINO}(g_t, r_t) = \frac{\langle f(g_t), f(r_t) \rangle}{\| f(g_t) \|_2 \| f(r_t) \|_2}
$$

其中 $f(\cdot)$ 是 DINOv2 编码器，$g_t$ 为生成帧，$r_t$ 为参考帧。

## 核心要点

1. **自监督预训练**：无需标签，通过教师-学生网络的知识蒸馏训练
2. **丰富语义特征**：保留对象身份和场景语义，比像素距离更鲁棒
3. **评估用途**：在 WAM 世界建模评估中作为语义/实例级对齐信号
4. **LDA-1B 应用**：在 DINO 潜在空间预测未来状态，而非视频 VAE 潜在

## 代表工作

- [[WAMSurvey]]: 用于世界模型生成质量的语义对齐评估
- LDA-1B: 在 DINOv3 潜在空间预测未来状态
- [[SKIP]]: 冻结 DINOv2 ViT-L/16 作为 SKIP-Selector 的语义特征流（2048D CLS + 均值块特征）

## 相关概念

- [[Diffusion Model]]
- [[World Action Model]]
- [[LPIPS]]
- [[FVD]]
