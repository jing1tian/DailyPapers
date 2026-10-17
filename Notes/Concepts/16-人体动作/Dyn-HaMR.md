---
type: concept
aliases: [Dyn-HaMR, Dynamic HaMeR]
---

# Dyn-HaMR

## 定义

Dyn-HaMR 是基于 [[HaMeR]] 的扩展方法，引入动态场景建模以处理相机运动和手部时序动态，用于从自我中心视频中重建 3D 手部运动轨迹。

## 核心要点

1. **时序动态建模**: 在 [[HaMeR]] 的单帧重建基础上，引入跨帧时序约束，提升手部轨迹的时序一致性
2. **相机运动处理**: 试图分离手部运动与相机自我运动（ego-motion），但单目相机尺度歧义仍是难点
3. **局限性**: 在复杂自我中心场景下相机空间 [[MPJPE]] 仍较大（宏平均 231.54 mm），不如联合建模方法

## 数学形式

基于 [[MANO]] 参数化手部模型，在 [[HaMeR]] 回归框架上添加时序平滑约束。

## 代表工作

- [[P3WAM]]: 将 Dyn-HaMR 作为 Perceive 模式的 baseline，P3-WAM 相机空间 MPJPE（63.05 mm）相比 Dyn-HaMR（231.54 mm）低 72.8%

## 相关概念

- [[HaMeR]]
- [[WiLoR]]
- [[HaWoR]]
- [[MANO]]
- [[MPJPE]]
