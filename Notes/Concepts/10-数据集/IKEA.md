---
type: concept
aliases: [IKEA家具, IKEABench, IKEA Assembly]
---

# IKEA

## 定义

瑞典家具品牌，其标准化、模块化的家具产品（如 LACK 桌、KALLAX 书架、IVAR 椅子）被广泛用作机器人装配研究的 benchmark 对象。

## 核心要点

1. **标准化设计**: IKEA 家具有明确的部件接口规范，适合定义量化的装配成功标准
2. **难度梯度**: 不同产品复杂度差异显著，LACK（12 子任务 / ~650 步）到 IVAR（25 子任务 / ~1550 步），自然构成难度递进的 benchmark
3. **磁性连接替代**: 机器人研究中常用磁性连接件替代真实螺丝，降低精度要求同时保持装配挑战性

## 代表工作

- [[FurnitureBench]]: 首个系统化的 IKEA 家具机器人装配 benchmark
- [[FurnitureVLA]]: 在 LACK / KALLAX / IVAR 三类家具上验证 VLA 双臂装配

## 相关概念

- [[FurnitureBench]]: IKEA 场景的专用仿真 benchmark
- [[装配成功判据]]: 用于评估 IKEA 部件对准的量化指标
- [[双臂机器人]]: IKEA 装配通常需要双臂协调
