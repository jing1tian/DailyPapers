---
type: concept
aliases: [DAVIS, Dynamic and Active-pixel Vision Sensor, DAVIS346, DAVIS240]
---

# DAVIS 事件相机

## 定义

DAVIS（Dynamic and Active-pixel Vision Sensor）是一种同时输出异步事件流和标准灰度帧的复合视觉传感器，结合了事件相机的高动态范围特性与传统帧式相机的纹理信息。

## 核心要点

1. **双模输出**：同时提供事件流（高时间分辨率）和灰度帧（高空间分辨率），两者共享同一像素阵列
2. **高动态范围**：>120 dB，远优于普通 CMOS 相机
3. **微秒级时间分辨率**：事件延迟 <1ms，适合高速场景
4. **常用型号**：DAVIS346（346×260 分辨率）、DAVIS240
5. **真实部署**：在弱光、高速运动等 RGB 退化场景下维持可用视觉输入

## 代表工作

- [[EventVLA]]: 在 Franka Research 3 机器人真实部署中使用 DAVIS 事件相机，验证弱光下机器人操控鲁棒性

## 相关概念

- [[事件相机]]: 事件相机的通用概念
- [[PREI]]: 基于事件相机输出的物理残差表征
- [[Franka Research 3]]: 与 DAVIS 配合使用的机器人平台
