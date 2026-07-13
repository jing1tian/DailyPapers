---
type: concept
aliases: [Plucker Embeddings, Plücker 嵌入, Plücker 坐标, 普吕克嵌入]
---

# Plücker Embeddings

## 定义

将三维空间中的视线（viewing ray）编码为 6 维 Plücker 坐标的技术，广泛用于相机位姿表示和条件视频/图像生成中的相机控制。Plücker 坐标同时编码了射线的方向与位置，具有旋转/平移不变性优势。

## 数学形式

对于从相机中心 $\mathbf{o} \in \mathbb{R}^3$ 出发、方向为 $\mathbf{d} \in \mathbb{R}^3$（单位向量）的视线，其 Plücker 坐标为：

$$\mathbf{p} = (\mathbf{d},\, \mathbf{m}) \in \mathbb{R}^6, \quad \mathbf{m} = \mathbf{o} \times \mathbf{d}$$

其中 $\mathbf{m}$ 为矩向量（moment vector），编码射线相对原点的位置信息。

**约束**: $\mathbf{d} \cdot \mathbf{m} = 0$（有效 Plücker 坐标满足此正交性条件）

## 核心要点

1. **6D 表示**: 相比 4×4 外参矩阵（16 维），6D Plücker 坐标更紧凑，且对神经网络更友好
2. **逐像素编码**: 为图像每个像素分配其对应视线的 Plücker 坐标，形成与图像同分辨率的位姿特征图
3. **注入方式**: 在扩散/生成模型中常通过[[AdaLN]]（自适应层归一化）注入，实现相机控制
4. **坐标系无关**: 相同射线在不同坐标系下的 Plücker 表示可通过线性变换转换，利于多视角一致性

## 代表工作

- [[CameraCtrl]]: 利用 Plücker 嵌入实现视频生成中的精确相机轨迹控制
- [[LingBot-World-Infinity]]: 用 Plücker 嵌入编码相机位姿，经 [[AdaLN]] 注入 DiT 块

## 相关概念

- [[AdaLN]]: Plücker 嵌入常用的注入机制
- [[DiT]]: 常与 Plücker 嵌入结合的扩散 Transformer 骨干
- [[Causal VAE]]: 视频生成中的 VAE 编码器
