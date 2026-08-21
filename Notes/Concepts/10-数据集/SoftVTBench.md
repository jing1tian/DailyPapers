---
type: concept
aliases: [SoftVTBench, Visuo-Tactile Deformable Benchmark]
---

# SoftVTBench

## 定义
一个可变形物体操作的视觉触觉数据集与基准，配对了策略可见的触觉观测（GelSight）和独立的物理 3D 形变 ground truth，引入 Deformation-aware Success Rate（DSR）替代纯任务成功率。

## 核心要点
1. 同时包含视觉、触觉（GelSight）和 3D 形变 GT 三种模态
2. DSR 指标区分"任务完成但形变过度"的策略
3. 分 In-Distribution 和 Out-of-Distribution 两种评测协议

## 代表工作
- Jing et al., 2026 — [[SoftVTBench]] (arXiv 2608.18701)

## 相关概念
- [[ManiSkill2]]
- [[ACT]]
- [[DP]]
