---
type: concept
aliases: [潜空间空间记忆, 潜空间记忆, Latent Spatial Memory]
---

# Latent Spatial Memory

## 定义

将三维场景信息以 `{(世界坐标, VAE潜特征)}` 对的形式持久缓存，在生成新视角时直接从潜空间读出条件特征，无需 RGB 点云的"光栅化→重编码"往返。

## 数学形式

$$
\mathcal{M} = \{(\mathbf{p}_i, \mathbf{f}_i)\}, \quad \mathbf{p}_i \in \mathbb{R}^3, \; \mathbf{f}_i \in \mathbb{R}^C
$$

与 RGB 点云的区别：颜色 $\mathbf{c}_i \in [0,1]^3$ 替换为 VAE 潜特征 $\mathbf{f}_i \in \mathbb{R}^C$（$C=48$）。

## 核心要点

1. **消除编码往返**: 无需 Rasterise→Encode，读出直接得到潜变量条件
2. **低分辨率投影**: 在潜分辨率（$W/s \times H/s$，$s=16$）下完成 [[z-Buffering]]，计算量是像素分辨率的 $1/s^2$
3. **自回归更新**: 每帧生成后将新区域特征反投影追加，支持无限长程视频
4. **动态目标过滤**: 仅缓存刚性几何，避免运动物体污染持久记忆

## 代表工作

- [[Mirage]]: 首次提出 Latent Spatial Memory 框架，应用于相机控制视频生成

## 相关概念

- [[VAE]]
- [[z-Buffering]]
- [[ControlNet]]
- [[单目深度估计]]
