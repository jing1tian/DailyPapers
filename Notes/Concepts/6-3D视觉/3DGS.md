---
type: concept
aliases: [3D Gaussian Splatting, 三维高斯散射]
---

# 3DGS

## 定义
3D Gaussian Splatting（Kerbl et al., SIGGRAPH 2023）：用各向异性 3D 高斯基元代替隐式神经场，通过可微分光栅化实现实时高质量新视角合成。

## 数学形式
每个高斯基元由位置 $\mu \in \mathbb{R}^3$、协方差矩阵 $\Sigma$、不透明度 $\alpha$ 和球谐函数颜色表示 $c$ 定义：

$$\mathcal{G}(x) = e^{-\frac{1}{2}(x-\mu)^T \Sigma^{-1}(x-\mu)}$$

协方差通过旋转矩阵 $R$ 和缩放矩阵 $S$ 分解为 $\Sigma = RSS^TR^T$，保证半正定。

## 核心要点
1. 从 SfM 稀疏点云初始化，通过梯度密度控制（剪枝 + 克隆）自适应调整基元分布
2. Tile-based 光栅化 + $\alpha$-blending 排序，实现 30+ fps @1080p 渲染
3. 显式表示支持场景编辑（增删、变形单个高斯）
4. 相比 NeRF：训练快（30min vs. 数小时）、渲染快（实时 vs. 秒级）、可编辑

## 代表工作
- [[3D Gaussian Splatting for Real-Time Radiance Field Rendering]]：原始论文
- [[4DGS]]：时序动态高斯
- [[GaussianDream]]：3DGS + world model
- [[GaussianWAM]]：3DGS 蒸馏到 WAM

## 相关概念
- [[NeRF]]
- [[4DGS]]
- [[VGGT]]
