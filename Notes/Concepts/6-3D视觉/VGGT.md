---
type: concept
aliases: [Visual Geometry Grounded Transformer]
---

# VGGT

## 定义
一种 feed-forward 的 3D 重建模型，从多视角图像中直接预测 3D 几何结构（相机姿态、深度图、3D 点云），无需迭代优化。

## 数学形式
输入多视角图像 $\{I_1,\ldots,I_N\}$，输出几何表示 $G$（点图、深度、相机参数）：
$$G = f_{\text{VGGT}}(I_1, \ldots, I_N)$$

## 核心要点
1. Feed-forward 推理：无需 NeRF 的迭代优化，速度快
2. 直接预测 3D 结构，支持多种下游任务（重建、新视角合成）
3. 可作为机器人 3D 感知的高效前端
4. 在 GaussianDream 中用于从观测直接生成 3DGS 场景

## 代表工作
- [[GaussianDream]]：利用 VGGT-like 架构做 feed-forward 3DGS 世界状态预测

## 相关概念
- [[SAM]]
- [[3DGS]]
