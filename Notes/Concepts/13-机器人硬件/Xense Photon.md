---
type: concept
aliases: [Xense Photon, Xense光子触觉传感器]
---

# Xense Photon

## 定义

Xense 公司生产的视觉触觉传感器，通过内置摄像头拍摄弹性凝胶表面的形变来感知接触力和纹理，与 [[GelSight]] 同属视觉触觉传感器（Visuotactile Sensor）类别。

## 核心要点

1. **工作原理**: 利用弹性凝胶在接触时的光学形变（颜色/位移变化）推断接触力分布
2. **与 GelSight 的关系**: 功能类似但来自不同厂商，传感器外观和标定参数不同
3. **跨传感器迁移**: VTAM 验证了同一视频 VAE 编码器可在 GelSight Mini 和 Xense Photon 之间零样本迁移，线性探测精度达 90%+

## 代表工作

- [[VTAM]]: 在 Flexiv Rizon 4 机械臂上使用 Xense Photon 进行 Hanoi 插盘任务，成功率 75%

## 相关概念

- [[GelSight]]
- [[Visuo-Tactile World Models]]
- [[Flow-Force Proxy]]
