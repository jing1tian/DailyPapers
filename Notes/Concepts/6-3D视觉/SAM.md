---
type: concept
aliases: [Segment Anything Model, 任意分割模型]
---

# SAM (Segment Anything Model)

## 定义
Meta AI 提出的通用图像分割基础模型，通过 prompt（点、框、文字）对任意物体进行零样本分割，是视觉感知的基础工具。

## 数学形式
输入图像 $I$ 和 prompt $P$，输出分割掩码 $M$：
$$M = f_{\text{SAM}}(I, P)$$
采用 MAE 预训练的 ViT 编码图像，轻量 mask decoder 解码分割。

## 核心要点
1. 零样本泛化：在 1B 张图片上训练，覆盖绝大多数物体类别
2. 交互式 prompt：支持点、框、文字、掩码等多种 prompt 形式
3. SAM2 进一步支持视频分割和实时追踪
4. 在机器人感知中常用于初始化 3D 重建（与 NeRF/3DGS 结合）

## 代表工作
- [[DeformMaster]]：用 SAM 分割视频帧中的可形变物体，初始化粒子场

## 相关概念
- [[CoTracker3]]
- [[PointNet]]
