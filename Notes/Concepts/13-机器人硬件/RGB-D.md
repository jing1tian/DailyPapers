---
type: concept
aliases: [RGB-D, RGB-D Camera, 深度相机, RGBD]
---

# RGB-D

## 定义
RGB-D（Red-Green-Blue Depth）传感器同时输出彩色图像与逐像素深度信息，为机器人提供三维感知能力，常见形式包括结构光深度相机（如 RealSense、Kinect）和飞时测距（ToF）相机。

## 核心要点
1. **深度来源**: 结构光、ToF 或主动立体视觉，室内范围通常 0.1–10 m
2. **应用场景**: 对象识别与定位、遮挡场景重建、三维导航与 SLAM
3. **限制**: 室外强光（眩光）、透明/反光表面会导致深度噪声或缺失
4. **与 RGB 结合**: 深度通道可直接融合点云，或与彩色特征结合用于 3D 对象分割

## 代表工作
- [[SCOUT]]: 以腕部 RGB-D 相机采集拥挤货架的稀疏多视角观测

## 相关概念
- [[Boston Dynamics Spot]]
- [[SAM]]
- [[GHOST]]
