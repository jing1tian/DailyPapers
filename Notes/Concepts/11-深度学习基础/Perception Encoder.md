---
type: concept
aliases: [感知编码器, PE, Perception Encoder]
---

# Perception Encoder

## 定义
Meta 提出的大规模视觉基础模型，通过在海量图像和视频数据上进行自监督预训练，学习泛化性强的视觉表示。常被用作视觉特征提取的冻结教师模型，为下游任务提供语义丰富的特征对齐监督。

## 核心要点
1. **冻结教师角色**：在下游任务训练中保持冻结，仅提供语义监督信号（对齐损失）
2. **强语义表示**：捕获图像/视频的高层语义特征，有别于像素级重建目标
3. **视觉对齐监督**：被 [[RepViTok]] 用于将视频潜变量对齐至语义空间

## 数学形式

在 [[RepWAM]] 中的使用方式：

$$
\mathcal{L}_\text{align} = \left\| \text{avg}(W_\text{align} z) - \text{avg}(G(o)) \right\|_2^2
$$

其中 $G$ 即为 Perception Encoder（冻结），$z$ 为待对齐的视频潜变量。

## 代表工作
- [[RepWAM]]: 以 Perception Encoder 为冻结教师模型，对齐视觉 token 的语义空间

## 相关概念
- [[RepViTok]]
- [[视觉编码器]]
- [[视觉基础模型]]
- [[知识蒸馏]]
