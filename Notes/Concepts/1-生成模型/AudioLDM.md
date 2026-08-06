---
type: concept
aliases: [Audio Latent Diffusion Model]
---

# AudioLDM

## 定义
AudioLDM：将潜在扩散模型（LDM）应用于音频生成的方法，通过文本条件在 latent 空间生成高质量音频。

## 核心要点
1. 用 CLAP 做音频-文本对齐
2. 在 mel-spectrogram latent 空间做扩散
3. 无需配对数据，通过对比学习实现文本控制

## 代表工作
- [[AVWM]]: 引用 AudioLDM 的音频生成组件

## 相关概念
- [[AVWM]]
- [[LVDM]]
