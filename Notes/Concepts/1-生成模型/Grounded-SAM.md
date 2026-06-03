---
type: concept
aliases: [Grounded SAM, GroundedSAM, Grounded Segment Anything]
---

# Grounded-SAM

## 定义

将 Grounding DINO（开放集目标检测）与 SAM（Segment Anything Model）结合的工具链，输入文本 prompt 即可对图像中的任意物体进行零样本检测与分割。

## 核心要点

1. **两阶段流程**: Grounding DINO 输出目标边界框 → SAM 以边界框为 prompt 生成精细分割掩码
2. **开放词汇**: 支持任意文本描述的物体分割，无需训练集中的类别限制
3. **高精度掩码**: SAM 的分割质量远优于传统检测框 crop，适合精细图像处理

## 代表工作

- [[RoboDream]]: 用于从机器人场景图像中分割任务相关物体，构建物体先验

## 相关概念

- [[SAM]]
- [[OmniPaint]]
- [[视频扩散模型]]
