---
type: concept
aliases: [Roofline, 屋脊线模型, roofline analysis]
---

# Roofline Model

## 定义

Roofline 模型是一种硬件性能分析框架，通过计算"算术强度"（FLOPs/Byte）将计算任务分类为计算密集型或内存密集型，指导硬件利用率优化。

## 数学形式

$$
\text{可达性能} = \min\left(\text{峰值算力},\; I \times \text{内存带宽}\right)
$$

其中算术强度：

$$
I = \frac{\text{FLOPs}}{\text{Bytes transferred}}
$$

屋脊点：$I_{\text{ridge}} = \frac{\text{峰值算力}}{\text{内存带宽}}$

## 核心要点

1. **计算密集区**（$I > I_{\text{ridge}}$）: 受峰值算力限制，提升算法效率有效
2. **内存密集区**（$I < I_{\text{ridge}}$）: 受内存带宽限制，降低内存访问量更有效
3. **VLA 分析**: VLM prefix 阶段（数百 token，计算密集）vs 动作专家阶段（少量 token，内存密集）

## 代表工作

- [[vla.cpp]]: 用 Roofline 分析区分 VLA 推理的两个阶段，指出量化降低容量但不改善延迟（对内存密集阶段更有效）

## 相关概念

- [[vla.cpp]]: 应用 Roofline 分析 VLA 推理瓶颈
- [[VLM Prefix]]: Roofline 框架下的计算密集阶段
