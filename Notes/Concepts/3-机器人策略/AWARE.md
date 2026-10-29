---
type: concept
aliases: [Action-Entropy-Weighted Retrieval]
---

# AWARE

## 定义
Yang 等（2026，arXiv 2602.18020，"AWARE: Action-Entropy-Weighted and Attention-Guided Retrieval of Observations for VLA"）提出的检索增强 VLA 推理框架，用动作熵（策略不确定性）和注意力图双重加权检索演示库参考帧。

## 核心要点
1. 动作熵越高（策略越不确定）检索权重越大，避免无意义检索
2. 用 VLA 内部注意力图指导检索相关区域
3. 检索参考帧作为 in-context 输入拼接送入 OpenVLA

## 代表工作
- Yang et al. (2026): UCAS

## 相关概念
- [[OpenVLA]]
- [[VLA]]
