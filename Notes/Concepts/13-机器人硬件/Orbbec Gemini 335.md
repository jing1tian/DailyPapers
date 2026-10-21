---
type: concept
aliases: [Orbbec Gemini335, 奥比中光Gemini335, RGB-D Camera]
---

# Orbbec Gemini 335

## 定义
Orbbec Gemini 335 是奥比中光（Orbbec）推出的一款结构光 RGB-D 深度相机，提供同步 RGB 和深度图像输出，广泛用于机器人操作实验。

## 核心要点

1. **输出类型**: 同步 RGB + 深度图（结构光测距）
2. **典型噪声**: 边缘缺失（Edge Missing）、公制抖动（Metric Jitter）、面片空洞（Patch Hole）
3. **机器人操作应用**: 与 Piper-X 机械臂配合，提供操作空间的几何感知
4. **与 RealSense D435 比较**: PrismWAM 同时参考两款传感器的规格设计深度噪声压力测试

## 代表工作
- [[PrismWAM]]: 真实世界实验使用 Orbbec Gemini 335 + Piper-X 机械臂，并据此传感器规格设计深度腐蚀测试

## 相关概念
- [[Depth Prediction]]
- [[Decoupled Representation]]
