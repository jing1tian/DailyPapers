---
type: concept
aliases: [NaviTrace, Navigation Trace Representation]
---

# NaviTrace

## 定义
将机器人导航轨迹投影到像素空间形成 trace image 的表示方法，作为跨平台（cross-embodiment）导航策略的统一接口。

## 核心要点
1. 把 3D 轨迹点投影到当前帧图像坐标系，得到 2D trace 叠加图
2. trace image 与 RGB 观测拼接后送入 policy head
3. 不同 embodiment 共享同一个 trace 表示，通过 residual adapter 适配各自运动特性
4. 相比直接输出 waypoints，trace 是连续可微的视觉信号，易于 VLA 解析

## 代表工作
- [[CrossTracer]]: 提出 NaviTrace + residual adapter 的跨平台导航框架

## 相关概念
- [[NavDP]]
- [[FlowNav]]
- [[VAMOS]]
