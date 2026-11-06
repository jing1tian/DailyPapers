---
type: concept
aliases: [Cross-View Fusion for 3DGS]
---

# CrossPressor

## 定义
Feed-forward 3DGS 的跨视角融合方法：用 cross-view attention 和 [[ToMe]] token merging 在多 anchor feature map 间做信息聚合，解决长多视角序列下 anchor 固定导致的表示瓶颈。

## 核心要点
1. 针对 anchor-based 3DGS 方法（如 ZPressor）在长序列下 anchor 固定的问题
2. QK attention 跨 anchor 聚合信息，ToMe merging 控制 token 数量
3. 对比 [[PixelSplat]], [[DepthSplat]]
4. 纯视觉重建，无机器人应用

## 相关概念
- [[3DGS]]
- [[PixelSplat]]
- [[DepthSplat]]
- [[ToMe]]
