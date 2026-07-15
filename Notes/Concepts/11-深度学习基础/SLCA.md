---
type: concept
aliases: [SLCA]
---

# SLCA（Slow Learner with Classifier Alignment）

## 定义
持续学习方法，通过慢学习率和分类器对齐策略缓解灾难性遗忘问题。

## 核心要点
1. 对骨干网络使用极低学习率，保留旧知识
2. 分类器对齐防止新类别覆盖旧类别的特征空间
3. 在 continual learning for VLA 场景中作为 baseline

## 相关概念
- [[VLA]]
