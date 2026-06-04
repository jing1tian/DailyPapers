---
type: concept
aliases: [语义分割, 像素级分类]
---

# Semantic Segmentation（语义分割）

## 定义

语义分割是计算机视觉中的密集预测任务，对图像中每个像素分配一个语义类别标签（如背景、物体类别），输出与输入同分辨率的类别图。

## 数学形式

$$
\hat{y} = \arg\max_c \, f_\theta(x)_c, \quad \hat{y} \in \mathbb{R}^{H \times W}
$$

训练时通常使用像素级交叉熵损失：

$$
\mathcal{L}_{\text{sem}} = -\frac{1}{HW} \sum_{h,w} \log p(y_{h,w} | x)
$$

## 核心要点

1. 与实例分割（Instance Segmentation）的区别：语义分割不区分同类物体的不同实例
2. 常用架构：FCN、DeepLab、SegFormer、DPT
3. 在机器人操作中用于识别物体类别和任务状态

## 代表工作

- [[GeoSem-WAM]]: 将语义分割作为辅助监督信号，改善世界模型的语义表示

## 相关概念

- [[Dense Prediction Transformer]]
- [[Dense Prediction]]
