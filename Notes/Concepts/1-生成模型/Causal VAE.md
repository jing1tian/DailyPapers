---
type: concept
aliases: [因果VAE, Causal Video Autoencoder, 因果视频自编码器]
---

# Causal VAE

## 定义

带有因果时序结构的视频变分自编码器（VAE），通过单向（因果）时序卷积在时间轴上压缩视频序列，确保当前帧的编码只依赖过去帧，不依赖未来帧，适用于在线/流式视频处理和世界模型。

## 数学形式

$$
z_t = \text{Enc}(x_{\leq t}), \quad x_t \approx \text{Dec}(z_{\leq t})
$$

时间压缩比通常为 4:1，即 4 帧原始视频压缩为 1 个潜在帧。

## 核心要点

1. **因果约束**: 编码器中所有时序卷积为因果卷积（不看未来），保证 $z_t$ 只取决于 $x_{\leq t}$
2. **时间压缩**: 典型压缩比 4:1（如 Cosmos：33 原始帧 → 11 填充 + 8 潜在帧）
3. **非图像数据编码**: 可将低维数据（状态向量、动作）渲染为常数图像帧后通过 VAE 编码，实现多模态统一
4. **生成一致性**: 解码时沿时间轴保持视觉一致性，避免帧间跳变

## 代表工作
- [[Cosmos Predict2]]: 使用 Causal VAE 将视频帧和非图像数据编码到统一潜在空间
- [[NavWAM]]: 利用 Causal VAE 将机器人状态、动作、价值编码为 Canvas 潜在帧

## 相关概念
- [[VAE]]: 标准变分自编码器，Causal VAE 的基础
- [[Latent Canvas]]: 利用 Causal VAE 构建的多模态潜在序列
- [[Diffusion Transformer]]: 在 Causal VAE 的潜在空间上进行扩散去噪
