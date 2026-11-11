---
type: concept
aliases: [LingBotTouch, Lingbot Touch, 触觉基础模型]
---

# LingBot-Touch

## 定义

一个跨压阻、磁性、视触觉三类传感器的触觉基础模型，通过将触觉信号映射至规范指尖坐标空间并以 Tactile-JEPA 在潜在空间预测接触，使同一冻结编码器能即插即用地提升多种机器人操作策略的性能，且无需为新传感器重新训练。

## 核心要点

1. **统一表征**：规范指尖球面空间 + 模态专用 Tokenizer + 共享 [[Haptic Transformer]]（HaT），三类传感器的 taxel 坐标在同一空间具有可比性
2. **预训练目标**：[[Tactile-JEPA]]——在潜在空间预测被遮盖 taxel 的表征，避免重建稀疏信号中大量零值背景
3. **规模化数据**：5,000 小时 / 464K 集数 / 13.3 亿帧，来自压阻/磁性触觉手套（自我中心）和视触觉双臂机器人（遥操作）
4. **零样本跨传感器**：预训练未见的 XHand1 磁性手无需任何参数更新，各项接触物理探针均优于 FTP-1（配备专属适配器）
5. **插件即用**：冻结 Tiny（5.46M）编码器 + 可训练跨注意力读出；对下游策略增加 ≤ 0.16% 参数，将 π0.5 / LingBot-VLA 在 26 个真实操作任务上提升 ~22–25 pp

## 两种尺寸

| 变体 | 参数量 | 层数 | 嵌入维度 |
|------|-------|------|---------|
| LingBot-Touch-Tiny | 5.46M | 12 | 192 |
| LingBot-Touch-Small | 21.76M | 12 | 384 |

## 代表工作

- [[LingBot-Touch（论文）|LingBot-Touch 论文笔记]]: 完整论文笔记
- Han et al., "Learning Heterogeneous Tactile Representation for General Dexterous Manipulation," arXiv 2611.12806, 2026

## 相关概念

- [[Tactile-JEPA]]: 核心预训练目标
- [[Haptic Transformer]]: 共享 Transformer 骨干
- [[Canonical Fingertip Space]]: 跨传感器统一接口
- [[FTP-1]]: 主要对比方法
- [[触觉传感器]]: 覆盖的三类传感器原理
- [[π₀.₅]]: 插件集成的下游 VLA 策略之一
- [[Dexterous Manipulation]]: 应用场景
