---
type: concept
aliases: [DINOSAUR, Distillation for Object-Centric Learning]
---

# DINOSAUR

## 定义
用 DINO 视觉特征替代像素作为重建目标的 object-centric slot learning 方法，比 SLATE 等像素重建方法更鲁棒，slot 质量更高。

## 核心要点
1. 改进 [[SLATE]]：用 [[DINOv2]] 特征做目标代替原始像素，避免对无关纹理过拟合
2. 无监督学习，不需要分割标注或边界框
3. [[COMET]] 用冻结 DINOSAUR encoder 提取 slot，再在 slot 空间做 causal WM + MCTS
4. 相比 SLATE 对背景噪声和纹理变化更鲁棒

## 代表工作
- [[COMET]]：DINOSAUR encoder + transformer WM + MCTS

## 相关概念
- [[SLATE]]
- [[DINOv2]]
- [[COMET]]
