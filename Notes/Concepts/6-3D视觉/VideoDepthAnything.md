---
type: concept
aliases: [Video Depth Anything]
---

# VideoDepthAnything

## 定义
VideoDepthAnything 是 Depth Anything 系列的视频扩展版本，利用视频帧间一致性约束生成时序上连贯的深度估计，解决单帧深度估计在视频中存在的闪烁问题。

## 核心要点
1. 在 Depth Anything V2 基础上加入时序一致性约束（视频帧间深度一致）
2. 输出的深度序列具有全局尺度一致性，适合作为几何初始化
3. 常用于 novel view synthesis 中提供几何先验
4. 零样本泛化能力强，不需要针对特定场景微调

## 代表工作
- [[UniWorld-View]]: 用 VideoDepthAnything 提供深度先验辅助大基线 NVS

## 相关概念
- [[VGGT]]
- [[NeRF]]
- [[Triple-Reprojection]]
