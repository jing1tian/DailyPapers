---
type: concept
aliases: [LAM, latent action, latent action model]
---

# Latent Action Model

## 定义
从无动作标注视频中学习隐式动作表示的方法，通过 VQ 或连续潜变量建模人类/机器人行为。

## 数学形式
$$z_t = \text{Encode}(o_t, o_{t+1}), \quad a_t = \text{Decode}(z_t, o_t)$$

## 核心要点
1. 1. 无需动作标注，只需观测序列
2. 2. 常用 VQ-VAE 或连续 VAE 编码
3. 3. 可用于人类视频迁移到机器人策略

## 代表工作
- [[HARP-VLA]]
- [[UniVLA]]

## 相关概念
- [[VLA]]
- [[Diffusion Model]]
