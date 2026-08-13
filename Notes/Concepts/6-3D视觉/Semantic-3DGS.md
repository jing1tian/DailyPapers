---
type: concept
aliases: []
---

# Semantic 3D Gaussian Splatting

## 定义
语义高斯溅射，在 3DGS 的每个高斯点上附加语义特征（CLIP/DINO/SAM），支持开放词汇的场景语义查询和分割。

## 核心要点
1. 每个高斯点除几何/颜色属性外，额外存储 CLIP 对齐特征和 DINOv2 特征，通过渲染-特征对比损失（余弦距离）蒸馏语义
2. 支持开放词汇查询：对正负提示词做 softmax 相关性评分，实现语言驱动的目标定位
3. 局部版本（本地 Semantic-3DGS）：从少量视角（如 4 张腕部相机图像）构建，面向机器人操作的实时局部感知

## 代表工作
- [[EmbodiedMG]]: 主动多视角局部 Semantic-3DGS，支持四足机器人开放词汇移动操作（MM '26）
- [[LangSplat]]: 经典大场景语义高斯溅射方法

## 相关概念
- [[3DGS]]
- [[LangSplat]]
- [[CLIP]]
- [[DINOv2]]
- [[VGGT]]
- [[主动视角选择]]
