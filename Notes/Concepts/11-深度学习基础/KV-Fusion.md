---
type: concept
aliases: [KV Fusion, Key-Value Fusion, 关键值融合]
---

# KV-Fusion

## 定义

KV-Fusion 是 [[DoT|Dock of Transformer（DoT）]]框架中的对接机制：通过可学习的 **通道混合**（channel mixing）和**跨层混合**（layer mixing）将视频骨干所有层的 Key/Value 缓存聚合为单套融合 KV，供轻量动作头直接使用，无需一一对应地堆叠动作层。

## 数学形式

**统一对接函数**:

$$
(\tilde{K}^v_j, \tilde{V}^v_j) = g_j\!\left(\{(K^v_l, V^v_l)\}_{l=1}^{L_v}\right)
$$

**步骤 A — 通道混合（Mode-n 乘积）**:

$$
\bar{K}^v = \text{reshape}\!\left(\text{reshape}(K^v) \times_4 W^K\right)
$$

**步骤 B — 跨层混合（可学习权重）**:

$$
\tilde{K}^{h,v} = \bar{K}^{h,v} \times_1 A_h, \quad \tilde{V}^{h,v} = \bar{V}^{h,v} \times_1 A_h
$$

**混合上下文注意力**:

$$
O^a_j = \text{Softmax}\!\left(\frac{Q^a_j [K^a_j, \tilde{K}^v_j]^\top}{\sqrt{d}}\right) [V^a_j, \tilde{V}^v_j]
$$

其中 $A_h \in \mathbb{R}^{1 \times L_v}$ 为各注意力头独立的跨层权重，$W^K$ 为特征维度投影矩阵。

## 核心要点

1. **两步融合**: 先用 Mode-4 张量乘积做特征空间对齐（视频维 → 动作维），再用逐头可学习权重做层维度加权求和
2. **全层访问**: 聚合骨干所有 $L_v$ 层的 KV（如 30 层），而非仅最终层
3. **参数轻量**: 约 20M 参数（相对 5B 骨干可忽略），主要来自通道投影矩阵和跨层权重矩阵
4. **层分布洞察**: 实验显示最强融合信号集中在中间层，但各层均有贡献，支持全层访问设计
5. **需配合 RoPE 对齐**: 视频骨干使用 [[3D RoPE]]，需先反旋到规范空间再融合，再重施动作头的 1D [[Rotary Position Encoding|RoPE]]

## 代表工作

- [[Faster-WAM]]: 提出 KV-Fusion，作为 DoT 框架的核心对接机制，配合单层动作头实现 3.2× 推理加速

## 相关概念

- [[DoT]]: KV-Fusion 是 DoT 框架的具体实现机制
- [[KV Cache]]: KV-Fusion 复用视频骨干前向传播产生的 KV 缓存
- [[3D RoPE]]: 视频骨干使用的位置编码，融合前需通过 RoPE 对齐处理
- [[Mixture-of-Transformers]]: 传统深度耦合设计，KV-Fusion 是其替代方案
- [[Fast-WAM]]: 采用 MoT 的对比基线
