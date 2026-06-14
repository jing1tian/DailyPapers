---
type: concept
aliases: [RGB Point Cloud Memory, 彩色点云记忆, RGB point cloud, colored point cloud]
---

# RGB 点云记忆

## 定义

一类视频世界模型中的三维场景表示方法，将场景存储为带 RGB 颜色的三维点集 $\mathcal{M}_\text{rgb} = \{(\mathbf{p}_i, \mathbf{c}_i)\}$，每步生成时需将点云光栅化为图像再重编码为 VAE 潜变量。

## 数学形式

**内存表示**：

$$
\mathcal{M}_\text{rgb} = \{(\mathbf{p}_i, \mathbf{c}_i)\},\quad \mathbf{p}_i \in \mathbb{R}^3,\; \mathbf{c}_i \in [0,1]^3
$$

**条件化流水线**（每步均需执行）：

$$
\hat{\mathbf{z}}^t = \mathcal{E}\bigl(\mathrm{Rasterise}(\mathcal{M}_\text{rgb};\, \mathbf{E}^t, K^t)\bigr)
$$

即：光栅化 → 重编码，每步引入两次冗余操作。

## 核心要点

1. 光栅化（$\mathrm{Rasterise}$）：将三维点云渲染到目标视角，是 GPU 密集型操作
2. 重编码（$\mathcal{E}$）：将渲染 RGB 图像重新过 VAE 编码器，引入信息损失且计算昂贵
3. 缓存大小与场景覆盖面积和像素分辨率成正比，随视频长度快速增长
4. 代表方法：[[Spatia]]、[[Voyager]]；相比 [[Latent Spatial Memory]]，慢 10.57× 且内存多 55×

## 代表工作

- [[Spatia]]: RGB 点云基线，WorldScore 69.73（Mirage 以 70.36 超越）
- [[Voyager]]: 另一 RGB 点云方法，RealEstate10K SSIM 0.636 vs Mirage 0.779
- [[Mirage]]: 提出以[[Latent Spatial Memory]]替代 RGB 点云记忆，消除光栅化-重编码瓶颈

## 相关概念

- [[Latent Spatial Memory]]
- [[z-Buffering]]
- [[VAE]]
- [[透视投影]]
