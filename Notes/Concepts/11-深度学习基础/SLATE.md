---
type: concept
aliases: [SLATE, Slot Attention with Language and Text Encoder]
---

# SLATE

## 定义
基于 Slot Attention 的无监督场景分解方法，将图像表示为一组 object slot（每个 slot 对应一个对象），用 transformer decoder 重建，无需分割标注。

## 核心要点
1. Slot Attention 迭代竞争机制将场景分解为 N 个对象 slot
2. [[DINOSAUR]] 是 SLATE 的改进版（用 DINO 特征替代像素重建）
3. [[COMET]] 用 SLATE/DINOSAUR 风格的 encoder 提取 slot 表示做 MCTS 规划
4. 对遮挡、多物体场景的分解能力有限

## 代表工作
- [[COMET]]：冻结 SLATE/DINOSAUR encoder，在 slot 空间做 MCTS

## 相关概念
- [[DINOSAUR]]
- [[COMET]]
- [[SAVI]]
