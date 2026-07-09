---
type: concept
aliases: [时空变换器, 时空注意力, Spatio-Temporal Attention, Factorized Spatio-Temporal Attention]
---

# Spatio-Temporal Transformer（时空变换器）

## 定义

将 Transformer 的自注意力机制分解为空间维度（同一帧内的 patches 之间）和时间维度（跨帧的对应 patch 之间）两个独立操作，以避免全时空注意力 $O(T^2 N^2)$ 的高计算复杂度。

## 数学形式

设帧数 $T$，每帧 patch 数 $N$，分解后复杂度降为 $O(T N^2 + T^2 N)$：

$$
\text{ST-Attn}(z) = \text{Temporal-Attn}\left(\text{Spatial-Attn}(z)\right)
$$

## 核心要点

1. **空间注意力**: 在每个时间步内，对帧内所有 patch 做全局自注意力
2. **因果时序注意力**: 对每个 patch 位置，仅关注当前及历史帧（因果掩码），用于世界模型自回归预测
3. **动作交叉注意力**: 将动作序列通过交叉注意力注入视觉潜在特征（用于动作条件世界模型）
4. **分解的理论依据**: 空间和时间相关性相对独立，分解后信息损失有限

## 代表工作

- [[DreamSteer]]: 使用分解时空变换器构建潜在动作条件世界模型，空间→时序→动作交叉注意力三级结构
- VideoTransformer / TimeSformer: 早期视频理解中的时空分解注意力

## 相关概念

- [[World Model]]
- [[DINOv2]]
- [[Transformer]]
