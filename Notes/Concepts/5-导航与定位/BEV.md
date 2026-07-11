---
type: concept
aliases: [Bird's Eye View, 鸟瞰视角, BEV感知]
---

# BEV（Bird's Eye View）

## 定义
BEV 感知是将多视角摄像头或 LiDAR 数据投影到统一的俯视坐标系（俯视图）进行表示和推理，消除摄像头视角差异，便于目标检测、路径规划和地图构建。

## 数学形式
$$F_{BEV}(x, y) = \text{ViewTransformer}(\{F_{cam}^i\}_{i=1}^{N})$$

典型实现通过 IPM（Inverse Perspective Mapping）或 Lift-Splat-Shoot 将图像特征投影到 BEV 平面。

## 核心要点
1. 自动驾驶标准感知范式（BEVFormer、BEVDet 等）
2. 消除各摄像头视角差异，统一空间坐标
3. 与 VLA 结合时可作为 world state 的空间先验（如 WCog-VLA）
4. 适合 multi-camera fusion，不适合单目设置

## 代表工作
- [[BEVFormer]]: 经典 BEV 感知 Transformer
- [[WCog-VLA]]: 用 BEV 做 autonomous driving VLA 的世界预测

## 相关概念
- [[NAVSIM]]
- [[UniAD]]
