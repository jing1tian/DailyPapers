---
type: concept
aliases: [Zero-Shot Object-Goal Navigation]
---

# ZSON

## 定义
Zero-Shot Object-Goal Navigation：agent 在未见过的环境中导航到仅由类别名称/描述指定的目标物体，无需任何目标实例的视觉样本。

## 核心要点
1. Zero-shot：目标类别在训练时未见过
2. Open-vocabulary：利用 CLIP 等语言-视觉对齐模型匹配目标
3. 在 AI2-THOR 等仿真环境中评估 SR（Success Rate）和 SPL

## 代表工作
- [[T-DRN]]: Temporal Difference-Relational Network for ZSON

## 相关概念
- [[AI2-THOR]]
- [[MJOLNIR]]
- [[CLIP]]
