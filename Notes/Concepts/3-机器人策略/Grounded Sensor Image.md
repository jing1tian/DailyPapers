---
type: concept
aliases: [Grounded Sensor Image, 接地传感器图像]
---

# Grounded Sensor Image

## 定义

将异构传感器（热成像、声学、毫米波雷达等）的热力图叠加到语义分割目标区域的 RGB 图像上，形成统一的多模态视觉表示，使标准视觉骨干网络无需修改即可处理多模态感知信息。

## 数学形式

$$
M = f_{\text{seg}}(o_{\text{RGB}},\; l_d)
$$

$$
m = M \odot s + (1 - M) \odot o_{\text{RGB}}
$$

其中 $M$ 为语义分割掩码，$s$ 为对齐后的传感器热力图，$\odot$ 为逐像素乘法。

## 核心要点

1. **模态无关**: 热成像、声强图、雷达反射图均可转化为 2D 热力图，统一叠加
2. **目标区域定位**: 通过语义分割（如 SAM3）精确定位目标物体，避免无关区域噪声
3. **视角对齐**: 传感器热力图需先通过 [[Homography|单应变换]] 对齐到 RGB 相机坐标系
4. **无架构改动**: 整个 VLA 模型无需增加新的编码器或融合层

## 代表工作

- [[MuseVLA]]: 提出 Grounded Sensor Image 概念，将热成像/声学/毫米波雷达统一为该表示

## 相关概念

- [[Sensor Token]]
- [[语义分割]]
- [[Homography]]
- [[mmWave Radar]]
- [[Thermal Camera]]
