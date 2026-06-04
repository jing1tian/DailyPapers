---
type: concept
aliases: [DPT, Dense Prediction Transformer]
---

# Dense Prediction Transformer (DPT)

## 定义

DPT 是一种将 Transformer 骨干（如 ViT）与密集预测任务（深度估计、语义分割）结合的架构，通过多尺度特征重组（Reassemble）和渐进式融合（Fusion）模块，将 token 序列转换为像素级密集预测图。

## 数学形式

Reassemble 操作从不同层提取 token 并重整为 2D/3D 特征图，Fusion 模块逐级上采样融合：

$$
F_{\text{dense}} = \text{Fusion}\left(\text{Reassemble}(t_1, t_2, t_3, t_4)\right)
$$

## 核心要点

1. 原始 DPT（Ranftl et al., 2021）针对 ViT 设计，用于单目深度估计和语义分割
2. 多尺度 Reassemble 模块：从不同 Transformer 层提取 token，重整为不同分辨率特征图
3. Fusion 模块通过残差卷积+双线性上采样逐级恢复空间分辨率
4. GeoSem-WAM 将其扩展为 3D DPT，用于视频 token 的密集监督

## 代表工作

- [[GeoSem-WAM]]: 将 DPT 辅助头应用于视频 DiT 的几何和语义监督

## 相关概念

- [[Dense Prediction]]
- [[Semantic Segmentation]]
- [[Video Diffusion Transformer]]
- [[Transformer]]
