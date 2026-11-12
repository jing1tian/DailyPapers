---
type: concept
aliases: [V-JEPA2-AC, Video JEPA 2 Action-Conditioned]
---

# V-JEPA 2-AC

## 定义
Meta AI 的 V-JEPA 2 的 action-conditioned 变体，一种 joint-embedding predictive architecture（JEPA）风格的视频世界模型，在 latent space 中预测 action 条件下的未来表示，而非像素级预测。

## 核心要点
1. 在 latent space 预测未来（比像素预测更高效）
2. Action conditioning 使其可用于机器人规划和控制
3. 避免了生成式解码的高计算代价

## 代表工作
- Meta AI, V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (2025)
- [[Few-Shot Residual Adaptation]]: 以 V-JEPA 2-AC 为基础 WM 研究 few-shot 适配

## 相关概念
- [[JEPA]]
- [[World Model]]
- [[WAM]]
