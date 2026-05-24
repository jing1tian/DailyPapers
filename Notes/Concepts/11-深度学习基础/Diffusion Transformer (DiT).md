---
type: concept
aliases: [DiT, Diffusion Transformer, DiT-S, DiT-B, DiT-XL]
---

# Diffusion Transformer (DiT)

## 定义
以 Transformer 替代 UNet 作为扩散模型骨干网络的架构，将图像 patch 序列化后用自注意力处理，时间步和条件信号通过 AdaLN-Zero 注入。

## 数学形式

时间步和条件嵌入通过 Adaptive Layer Norm（AdaLN）注入每个 Transformer block：

$$
\text{AdaLN}(h, c) = c_\text{scale} \cdot \text{LayerNorm}(h) + c_\text{shift}
$$

其中 $c_\text{scale}, c_\text{shift}$ 由条件向量 MLP 预测。

## 核心要点
1. 将图像分成 $p \times p$ 的 patch（通常 2×2 或 4×4），展平后做 Transformer 序列处理
2. DiT-S/B/L/XL 以模型宽度和深度区分规模
3. 相比 UNet，DiT 在高分辨率生成上扩展性更好
4. 条件信号（动作、历史帧特征）可通过 cross-attention 注入

## 代表工作
- [[CoME]]: 以 DiT-S 为骨干，通过 cross-attention 注入历史帧条件

## 相关概念
- [[扩散模型]]
- [[Transformer]]
