---
type: concept
aliases: [残差网络, Residual Network, Deep Residual Learning]
---

# ResNet

## 定义

ResNet（Residual Network）是何凯明等人提出的深度卷积网络架构，通过跨层残差连接（shortcut connection）解决深层网络梯度消失问题，是计算机视觉领域影响最广的骨干网络之一。

## 数学形式

残差块：

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

其中 $\mathcal{F}$ 为待学习的残差映射，$\mathbf{x}$ 为恒等捷径连接。

## 核心要点

1. **残差学习**: 学习目标从完整映射转为残差映射，大幅降低优化难度
2. **深度可扩展**: ResNet-18/34/50/101/152 等变体，最深可达数千层
3. **预训练广泛**: ImageNet 预训练的 ResNet-50 是众多视觉任务的标准特征提取器

## 代表工作

- [[SKIP]]: 使用冻结 ResNet-50 作为间距预测器的图像编码骨干网络

## 相关概念

- [[DINO]]
- [[视频扩散模型]]
