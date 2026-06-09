---
type: concept
aliases: [SoundStream neural audio codec]
---

# SoundStream

## 定义
Google 提出的端到端神经音频编解码器，将音频信号压缩为离散 token 序列（Residual Vector Quantization），支持高质量重建和低延迟流式传输。

## 核心要点
1. 全卷积 encoder-decoder 架构，配合 Residual VQ（RVQ）量化
2. 可在极低比特率（3 kbps）下维持高音频质量
3. 产出的离散音频 token 可直接被语言模型或多模态模型使用
4. 被 AudioPaLM、AudioLM、AVWM 等多模态系统用作音频前端

## 代表工作
- Zeghidour et al. (2021): SoundStream, IEEE/ACM TASLP
- [[AVWM]]: 用 SoundStream 编码音频输入构建音视觉世界模型

## 相关概念
- [[VQ-VAE]]
- [[AVWM]]
