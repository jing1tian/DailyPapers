---
type: concept
aliases: [深度预测, 深度估计, Depth Estimation, Depth Rollout]
---

# Depth Prediction

## 定义
深度预测是指从 RGB 图像或其他传感器输入估计场景中物体到相机距离的任务，在机器人操作中也特指预测未来帧的深度演变序列（深度展开）。

## 数学形式

在世界模型中，深度展开预测：

$$
p_\theta(d_{1:F} \mid d_0, z_R, c)
$$

其中 $d_0$ 为当前深度帧，$z_R$ 为 RGB 语义引导，输出未来 $F$ 帧深度序列。

## 核心要点

1. **物理表示优势**: 深度直接编码空间占据和位姿信息，比 RGB 更适合表示物体运动
2. **外观不变性**: 深度不受光照、颜色、纹理变化影响，对外观 OOD 场景更鲁棒
3. **传感器噪声**: 实际深度传感器存在 Metric Jitter、Edge Missing、Patch Hole 等噪声
4. **与 RGB 互补**: 深度提供几何信息，RGB 提供语义信息，两者互补

## 代表工作
- [[PrismWAM]]: 在深度空间进行密集物理展开，结合稀疏 RGB 语义引导，实现解耦 WAM
- [[4D-WAM]]: 将 RGB 与几何信息联合处理（但未完全解耦）

## 相关概念
- [[Decoupled Representation]]
- [[World Action Model]]
- [[RGB VAE]]
- [[Orbbec Gemini 335]]
