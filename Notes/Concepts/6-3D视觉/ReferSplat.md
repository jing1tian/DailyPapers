---
type: concept
aliases: [ReferSplat, Referring 3D Gaussian Splatting]
---

# ReferSplat

## 定义

ReferSplat 是一种在 3D Gaussian Splatting 场景中通过自然语言指称表达（referring expression）定位和分割目标物体的方法，作为 TrackRef3D 的对比基线之一。

## 核心要点

1. **语言指称分割**: 用"the red cup on the left"这类自然语言定位 3DGS 场景中的特定物体
2. **多视角一致性挑战**: 传统方法在不同视角的分割结果可能不一致
3. **与 TrackRef3D 对比**: TrackRef3D 用 Track-then-Label 流程解决多视角不一致问题

## 相关概念

- [[3D Gaussian Splatting]]
- [[LangSplat]]
- [[LERF]]
- [[OpenGaussian]]
- [[SAM-2]]
