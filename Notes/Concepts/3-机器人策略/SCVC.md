---
type: concept
aliases: [Selective Cross-View Consistency, cross-view consistency, 选择性跨视角一致性]
---

# SCVC（Selective Cross-View Consistency）

## 定义
WAM 训练时对同一状态的多视角样本施加选择性一致性约束，使模型对推理时未见视角保持鲁棒，且不需要测试时的相机参数。

## 核心要点
1. 非所有视角对都需要一致性约束（过强约束导致特征退化）
2. 选择性施加约束的标准需要设计（heuristic 或 learned）
3. 推理时完全不依赖相机信息，具有实际部署价值

## 代表工作
- [[SCVC]]: 在 WAM 上的选择性跨视角一致性训练

## 相关概念
- [[World Action Model]]
- [[Viewpoint Robustness]]
- [[Domain Randomization]]
