---
type: concept
aliases: [mmWave Radar, 毫米波雷达, millimeter-wave radar]
---

# mmWave Radar

## 定义

工作在毫米波频段（30 GHz–300 GHz）的雷达传感器，利用电磁波反射感知目标距离、速度和方位，可穿透遮挡物（如布料、烟雾），常用于自动驾驶和机器人感知。

## 核心要点

1. **穿透性**: 可探测布料、泡沫等轻度遮挡物后的物体，弥补 RGB 相机视线局限
2. **多维感知**: 同时提供距离、速度（多普勒）、方位角等信息
3. **与 Digital Beamforming 结合**: 通过[[Digital Beamforming|数字波束成形]]将天线阵列信号转化为 2D 空间热力图
4. **代表产品**: Calterah 4T4R 60GHz（MuseVLA 实验平台使用）

## 代表工作

- [[MuseVLA]]: 将 mmWave 雷达热力图通过 Grounded Sensor Image 机制融入 VLA 模型，实现遮挡物体的感知与抓取

## 相关概念

- [[Digital Beamforming]]
- [[Grounded Sensor Image]]
- [[Thermal Camera]]
