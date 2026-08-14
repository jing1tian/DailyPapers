---
type: concept
aliases: [Thermal Camera, 热成像相机, 红外热像仪, infrared camera]
---

# Thermal Camera

## 定义

通过探测物体发射的红外辐射生成温度分布图（热力图）的相机，可在无可见光条件下感知物体温度，用于机器人辨别材质温度属性。

## 核心要点

1. **温度感知**: 直接测量物体表面温度分布，区分热饮/冷饮、高温/低温物体
2. **被动感知**: 不需要主动照明，物体自发辐射红外线即可成像
3. **分辨率限制**: 热成像相机分辨率通常低于 RGB 相机，定位精度受限
4. **与 RGB 融合**: 通过[[Homography|单应变换]]对齐到 RGB 视角，叠加为[[Grounded Sensor Image]]
5. **代表产品**: infiRay T2S（MuseVLA 实验平台使用）

## 代表工作

- [[MuseVLA]]: 将热成像数据融入 VLA 模型，实现温度引导的机器人操作（如拿取最热/最冷的饮料）

## 相关概念

- [[mmWave Radar]]
- [[Grounded Sensor Image]]
- [[Sensor Token]]
