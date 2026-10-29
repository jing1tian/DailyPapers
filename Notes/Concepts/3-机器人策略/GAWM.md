---
type: concept
aliases: [Compact Latent World-Action Model]
---

# GAWM

## 定义
Liu 等（2026，arXiv 2610.37091，"GAWM: A Compact Latent World-Action Model for Multi-Task Robot Manipulation"）提出的轻量 latent 世界-动作模型，用小型 transformer 在 latent space 做 state 和 action 联合预测，不依赖 billion-parameter LLM/视频生成骨干。

## 核心要点
1. 轻量化 WAM：用小型 transformer 代替大型 VLM/视频生成骨干
2. 在 latent space 做 state 和 action 联合预测
3. 在资源受限场景下保持竞争性 multi-task manipulation 性能

## 代表工作
- Liu et al. (2026): 原始论文

## 相关概念
- [[World Model]]
- [[VLA]]
