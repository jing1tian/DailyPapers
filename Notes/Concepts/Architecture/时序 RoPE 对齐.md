---
type: concept
aliases: [Temporal RoPE Alignment, Temporal RoPE, 时序旋转位置编码对齐]
---

# 时序 RoPE 对齐

## 定义
在多模态联合生成模型中，将异构序列（如力矩 token）通过旋转位置编码（[[RoPE]]）映射到视频 latent 的时间轴，使两种模态的时间位置一致，从而支持跨模态的同步去噪。

## 数学形式

$$
\rho(i) = \text{round}\!\left(\frac{i}{T-1} \times (f-1)\right), \quad i = 0, \ldots, T-1
$$

其中 $T$ 为力矩 token 序列长度，$f$ 为视频 latent 时间维度，$\rho(i)$ 为第 $i$ 个力矩 token 分配的时间位置。

## 核心要点
1. 解决视频 latent 和力矩序列帧率/长度不同导致的时间对齐问题
2. 线性插值保证力矩信号在视频时间轴上均匀分布
3. 对齐后两种模态共享同一 Transformer 做联合去噪，无需额外融合模块

## 代表工作
- [[TACO]]: 将 Xense 力矩序列（$T$ 步）与视频 latent（$f$ 步）通过 Temporal RoPE 对齐，实现视触觉联合 [[Flow Matching]] 去噪

## 相关概念
- [[RoPE]]
- [[Flow Matching]]
- [[视触觉生成]]
- [[Wan2.2]]
