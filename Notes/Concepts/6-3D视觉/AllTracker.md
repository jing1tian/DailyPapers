---
type: concept
aliases: [All-Tracker, AllTracker]
---

# AllTracker

## 定义
一种统一的密集视觉追踪框架，能够在视频序列中同时追踪任意点、像素或区域，无需针对具体对象类别做专门设计。

## 核心要点
1. 将追踪问题统一为"任意点追踪"（any-point tracking）
2. 支持密集追踪（dense tracking），覆盖整帧所有像素
3. 常用于世界模型中作为视觉感知前端，提供时序对应关系

## 代表工作
- [[Hydra-0]]: 将 AllTracker 作为空间感知辅助组件，辅助世界模型的状态估计

## 相关概念
- [[World Model]]
- [[DiT]]
