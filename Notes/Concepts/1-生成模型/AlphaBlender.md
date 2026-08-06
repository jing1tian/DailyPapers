---
type: concept
aliases: [Alpha Blending]
---

# AlphaBlender

## 定义
AlphaBlender：视频生成中用于融合 content 和 motion 分支特征的可学习权重混合模块，通过 alpha 通道控制两路特征的融合比例。

## 核心要点
1. 可学习的逐元素混合权重 alpha
2. 实现内容保持和运动融合的可控平衡
3. 常用于解耦 VAE 的解码器中

## 代表工作
- [[EmbodiedVAE]]: 用 AlphaBlender 合并 content 和 motion decoder 输出

## 相关概念
- [[VidTwin]]
- [[EmbodiedVAE]]
