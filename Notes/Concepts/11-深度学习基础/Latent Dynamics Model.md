---
type: concept
aliases: [LDM, latent dynamics, latent dynamics model]
---

# Latent Dynamics Model

## 定义
在隐空间而非像素空间学习状态转移动态：将观测编码到 latent 表示后，在 latent 空间中预测状态变化和动作效果，避免像素空间的高维冗余。

## 数学形式
$$z_{t+1} = f_\theta(z_t, a_t) + \epsilon, \quad z_t = \text{Enc}(o_t)$$

## 核心要点
1. Latent 空间建模比像素空间更高效，过滤视觉冗余
2. 与 RSSM（Dreamer 类）的区别：LDM 可针对 action 对齐进行专门设计
3. 用于 World Action Model 时，latent 需要与低层动作信号对齐

## 代表工作
- [[LD4WAM]]: 从人类视频学习 motion-aligned 的 latent dynamics
- [[RSSM]]: Dreamer 系列的循环状态空间模型

## 相关概念
- [[RSSM]]
- [[World Model]]
- [[World Action Model]]
