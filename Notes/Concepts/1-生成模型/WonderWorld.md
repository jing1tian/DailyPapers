---
type: concept
aliases: [WonderWorld]
---

# WonderWorld

## 定义
基于 layered Gaussian Splatting 的交互式驾驶场景视频世界模型，支持用户用鼠标控制相机并实时生成新视角。

## 核心要点
1. 用 layered 3DGS 表示场景（前景物体 + 背景），支持解耦编辑
2. 控制信号：鼠标点击指定运动方向，Real-Time（~10fps）生成
3. 主要评测场景：驾驶/户外视频

## 代表工作
- [[WonderWorld]] (arXiv 2024): "WonderWorld: Interactive 3D Scene Generation with a Single Image"

## 相关概念
- [[FlashWorld]]
- [[LucidDreamer]]
- [[Latent Spatial Memory]]
