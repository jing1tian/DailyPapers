---
type: concept
aliases: [z-Buffer, 深度缓冲, z缓冲, 深度缓冲算法]
---

# z-Buffering

## 定义

计算机图形学中的可见性判断算法：当多个几何图元投影到同一像素（或格子）时，保留深度值（z 值）最小（即距相机最近）的图元，实现正确的遮挡关系。

## 数学形式

对目标视角格子 $(u,v)$，候选点集为：

$$
\Omega_t(u,v) = \{i : \pi^\ell(\mathbf{q}_i) = (u,v), [\mathbf{q}_i]_z > 0\}
$$

选择最近点：

$$
i^*(u,v) = \operatorname*{arg\,min}_{i \in \Omega_t(u,v)} [\mathbf{E}^t \mathbf{p}_i]_z
$$

## 核心要点

1. **逐像素深度比较**: 维护深度缓冲区，只更新深度更小的像素
2. **处理遮挡**: 自然处理前景遮挡后景的情形
3. **潜空间应用**: 在 [[Latent Spatial Memory]] 中，z-buffering 在潜分辨率格子上执行，而非像素级别
4. **无候选时**: 投影格子无命中时输出零向量，由扩散模型自行补全

## 代表工作

- [[Mirage]]: 将 z-buffering 应用于潜空间记忆读出，实现高效可见性判断

## 相关概念

- [[Latent Spatial Memory]]
- [[针孔相机模型]]
- [[透视投影]]
