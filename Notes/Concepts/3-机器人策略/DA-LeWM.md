---
type: concept
aliases: [DA-LeWM, Decision-Aligned Latent World Model]
---

# DA-LeWM

## 定义
在 JEPA 潜空间世界模型基础上加入辅助预测 head 和 SIGReg 正则化，使 latent 空间的欧氏距离能够正确对候选动作序列排序（decision-metric alignment），改善 MPC 规划质量。

## 核心要点
1. Decision-Metric Alignment（DMA）：形式化定义 latent WM 的"规划有效性"，不只看预测精度
2. Plan-Real 和 CEM-stage Spearman：量化 DMA 程度的两个新诊断指标
3. 辅助 head：逆向动力学、目标条件动作、reward/value proxy
4. SIGReg 正则化：在保持信息量的同时提升 latent 空间的决策几何结构

## 代表工作
- Wang et al., 2026 — [[DA-LeWM]] (arXiv 2608.18746)

## 相关概念
- [[JEPA]]
- [[LeWM]]
- [[PLDM]]
- [[CEM]]
