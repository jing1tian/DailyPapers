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
- [[3DThinkVLA]]：使用 VGGT 输出作为几何对齐目标，通过几何适配器将 3D 几何先验蒸馏到 VLA 视觉特征（仅训练时使用）
- [[G3VLA]]：以 VGGT 系列模型 π³ 作为几何教师，通过置信度门控蒸馏稠密点图（光线坐标 + 对数深度）来预训练 VLA 的几何感知视觉模块；论文同时发现 π³ 在干净合成场景下存在尺度不一致问题，是蒸馏失败的根源之一
- [[Mind-VLA]]：冻结 VGGT 提取目标物体三视图的 4 层中间特征，作为指令感知几何对齐的监督目标（$\mathbf{g}_{m(\ell),j}$）

## 相关概念
- [[SAM]]
- [[3DGS]]
