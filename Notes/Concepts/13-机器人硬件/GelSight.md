---
type: concept
aliases: [GelSight Sensor, 光学触觉传感器]
---

# GelSight

## 定义
MIT 开发的光学触觉传感器，通过弹性凝胶接触面在内部相机下的形变图像，高分辨率重建接触表面的 3D 几何和法向力分布。

## 数学形式
法向量估计（光度立体法）：
$$\mathbf{n}(x,y) = \arg\min_n \sum_i \| I_i(x,y) - \rho \cdot (\mathbf{l}_i \cdot \mathbf{n}) \|^2$$

## 核心要点
1. 弹性凝胶涂有反光涂层，接触时形变被内置摄像头记录
2. 通过彩色 LED 多角度照明实现光度立体，重建接触面法向量
3. 空间分辨率可达 ~0.05mm，远超压力传感器阵列
4. 无需逐像素标定，但依赖初始几何校准
5. 商业化版本：DIGIT（Meta）、GelSlim（CMU）

## 代表工作
- [[TactileReflex]]: 利用 GelSight 噪声统计实现力敏感操控
- [[触觉传感]]: GelSight 是主流光学触觉传感器代表

## 相关概念
- [[GelSlim]]
- [[接触力估计]]
