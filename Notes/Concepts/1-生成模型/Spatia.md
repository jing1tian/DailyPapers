---
type: concept
aliases: [Spatia]
---

# Spatia

## 定义

Spatia 是一种基于 **RGB 点云** 的相机可控视频世界模型，通过维护显式三维彩色点云并在每步条件化时光栅化渲染 + 重编码为潜变量，实现跨帧的空间一致性。

## 核心要点

1. **RGB 点云记忆**: 维护 $\mathcal{M}_\text{rgb} = \{(\mathbf{p}_i, \mathbf{c}_i)\}$，存储世界坐标与 RGB 颜色
2. **两步条件化**: 每帧先光栅化点云到目标视角，再用 VAE 编码器将 RGB 图重编码为潜变量
3. **性能瓶颈**: 光栅化（GPU 密集）+ 编码（encoder 推理）随视频长度线性增长
4. **WorldScore 得分**: 69.73（vs Mirage 70.36）

## 与 Mirage 的对比

| 指标 | Spatia | Mirage |
|------|--------|--------|
| WorldScore Avg | 69.73 | **70.36** |
| 每帧耗时 | 2.73s | **0.25s** |
| 缓存内存/chunk | 62.5 MiB | **0.48 MiB** |
| RealEstate10K SSIM | 0.646 | **0.779** |

## 代表工作

- [[Mirage]]: 使用潜空间记忆替代 RGB 点云，实现 10.57× 加速，性能同时超越

## 相关概念

- [[RGB 点云记忆]]
- [[Latent Spatial Memory]]
- [[VAE]]
- [[z-Buffering]]
