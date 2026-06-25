---
type: concept
aliases: []
---

# SplArt

## 定义
基于 3D Gaussian Splatting 的可动关节物体（articulated object）重建方法，从多视角/RGB-D 观测中同时估计物体的几何外观和关节运动参数。

## 核心要点
1. 在 3DGS 表征基础上引入关节运动建模，区别于静态场景的标准 3DGS 重建
2. 与 [[PARIS]] 同属"关节物体重建"路线，但表征基底从隐式/NeRF 换成显式高斯
3. 常作为可动数字孪生重建工作的直接对比基线

## 相关概念
- [[3D Gaussian Splatting]]
- [[PARIS]]
