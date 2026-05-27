---
type: concept
aliases: [ResNet50, Residual Network 50, ResNet]
---

# ResNet-50

## 定义

ResNet-50 是由 He 等人提出的 50 层深度残差网络，通过引入跳跃连接（skip connection）解决深层网络训练中的梯度消失问题，是计算机视觉中最广泛使用的骨干网络之一。

## 数学形式

残差块的基本形式：

$$
y = \mathcal{F}(x, \{W_i\}) + x
$$

其中 $\mathcal{F}(x, \{W_i\})$ 是残差映射，$x$ 是跳跃连接的恒等映射。

## 核心要点

1. **残差学习**: 让网络学习残差 $\mathcal{F}(x) = H(x) - x$ 而非直接拟合目标映射 $H(x)$
2. **瓶颈结构**: 使用 1×1 → 3×3 → 1×1 卷积的瓶颈块，降低计算量
3. **全局平均池化**: 将特征图压缩为固定维度向量（ResNet-50 输出 2048-D）
4. **轻量级应用**: 相比 ViT 等大型 Transformer 编码器，ResNet-50 计算量小，适合作为辅助网络（如 Critic）

## 代表工作

- [[EXPO-FT]]: 使用 ResNet-50 作为 Critic 网络的轻量级视觉骨干，避免使用大型 VLA 编码器带来的计算开销

## 相关概念

- [[Actor-Critic]]
- [[VLA]]
