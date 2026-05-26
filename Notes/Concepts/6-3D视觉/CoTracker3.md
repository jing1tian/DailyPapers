---
type: concept
aliases: [CoTracker3, Cotracker]
---

# CoTracker3

## 定义
Meta 开发的视频点追踪基础模型，能在长视频中密集追踪任意查询点，支持遮挡处理和半监督学习（利用伪标签自训练）。

## 数学形式
追踪问题：给定帧 $t_0$ 的点集 $\{p_i\}$，预测所有后续帧的对应位置 $\{p_i^t\}$ 和可见性 $\{v_i^t\}$。

## 核心要点
1. 基于 Transformer，联合处理时空相关性（不同帧的点互相attend）
2. 支持任意初始化点（密集或稀疏），不限于角点
3. CoTracker3 加入半监督：用实际视频生成伪标签继续训练
4. 在 TAP-Vid benchmark 上显著优于光流方法
5. GEM-4D、JOPAT 用 CoTracker 提取 4D 对应关系

## 代表工作
- [[GEM-4D]]: 用 CoTracker3 提取 4D 对应关系做几何监督
- [[JOPAT]]: 类似点追踪辅助 WAM 训练

## 相关概念
- [[VGGT]]
- [[FoundationPose]]
