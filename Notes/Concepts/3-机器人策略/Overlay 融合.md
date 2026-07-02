---
type: concept
aliases: [Overlay Fusion, Event Overlay, 事件叠加融合]
---

# Overlay 融合

## 定义

一种**零参数**的事件-图像融合策略：将[[事件相机]]窗口内最新事件的极性颜色**直接覆盖**到对应 RGB 像素位置，无事件的像素保留原始 RGB 值，得到混合帧后送入标准视觉编码器。

## 数学形式

$$
I^o(x,y)=\begin{cases}I(x,y), & |\mathcal{E}_{(x,y)}|=0 \\ \boldsymbol{c}(p_j), & j=\operatorname*{argmax}_i t_i, e_i \in \mathcal{E}_{(x,y)}\end{cases}
$$

其中 $I(x,y)$ 为原始 RGB 像素，$\mathcal{E}_{(x,y)}$ 为该位置的事件集合，$\boldsymbol{c}(p_j)$ 为最新事件极性的颜色编码。

## 核心要点

1. **零额外参数**: 不引入任何可学习参数，无 FLOPs 开销
2. **保持 token 分布**: 叠加后的混合帧仍是三通道图像，与预训练视觉编码器输入分布兼容
3. **最新事件优先**: 仅保留每个像素位置时间戳最新的事件，避免多事件叠加产生的歧义
4. **性能限制**: 在极低光照（20 lux）Pick-Place 成功率为 60%，低于[[层次化事件适配器]]的 90%

## 代表工作

- [[E-VLA]]: Overlay 作为轻量级基线方案，Pick-Place 低光照平均成功率 80.8%，适合算力受限场景

## 相关概念

- [[事件相机]]: 提供事件流输入
- [[近计数事件窗口]]: 为 Overlay 提供分割后的事件窗口
- [[层次化事件适配器]]: 性能更强的参数化替代方案
- [[SigLIP]]: 接收 Overlay 混合帧的下游视觉编码器
