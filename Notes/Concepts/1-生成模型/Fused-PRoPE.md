---
type: concept
aliases: [Fused PRoPE, 融合投影式旋转位置编码, 多视角PRoPE]
---

# Fused-PRoPE（融合投影式旋转位置编码）

## 定义

[[PRoPE]]（投影旋转位置编码）的多视角扩展，通过统一焦距近似不同相机内参，构造融合内外参的投影矩阵注入 [[RoPE|旋转位置编码]]，使视角层注意力感知多路相机的几何关系。

## 数学形式

**投影矩阵**:

$$
P_i = K_f \cdot E_i
$$

其中：
- $K_f$：统一焦距的相机内参矩阵（以共享焦距近似多路异构相机）
- $E_i$：视角 $i$ 的外参矩阵（world-to-camera 变换）

**注意力中的注入**（继承 PRoPE 形式）:

$$
\text{Attn}_{\text{Fused-PRoPE}}(Q,K,V) = D^{P_i} \odot \text{Attn}\!\left((D^{P_i})^\top Q,\ (D^{P_i})^{-1} K,\ V\right)
$$

## 核心要点

1. **统一焦距近似**：用共享 $K_f$ 处理不同规格相机，简化多路系统中内参差异带来的编码复杂度
2. **内外参融合**：将相机几何信息（位置 + 朝向 + 焦距）一次性编码进 token 对之间的相对变换
3. 仅作用于**视角层**（偶数 Transformer 层），与时序层的普通 RoPE 完全解耦
4. 对比原始 PRoPE：Fused-PRoPE 显式处理多视角场景，支持 1–7 路灵活配置

## 代表工作

- [[XiaomiAutoWM2]]: Fused-PRoPE 用于 [[View-Temporal Attention]] 视角层，在 CamTrol Benchmark 上取得 WorldScore 86.35、平均分 78.22（SOTA）

## 相关概念

- [[PRoPE]]: 单视角/小视角版本的投影式 RoPE，Fused-PRoPE 的设计基础
- [[RoPE]]: 旋转位置编码基础
- [[3D RoPE]]: 三维扩展 RoPE
- [[View-Temporal Attention]]: Fused-PRoPE 注入的注意力架构
- [[Multi-Modal 3D RoPE]]: 多模态三维 RoPE（相关方法）
