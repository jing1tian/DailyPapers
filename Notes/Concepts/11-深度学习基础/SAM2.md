---
type: concept
aliases: [SAM2, Segment Anything Model 2, SAM 2]
---

# SAM2

## 定义

SAM2（Segment Anything Model 2）是 Meta 提出的视频对象分割基础模型，可在视频序列中追踪任意对象的分割掩码，在 WBench 中被用于评估主体视角一致性（质心稳定性）。

## 核心要点

1. **视频对象追踪**: 在时序帧间持续追踪指定对象的分割区域
2. **零样本泛化**: 无需训练即可处理新场景中的任意主体
3. **WBench 应用**: 计算 C.4 Perspective Consistency（主体质心跨帧稳定性）

## 代表工作

- Ravi et al. 2024: SAM 2 原始论文（Meta AI Research）
- [[WBench]]: 使用 SAM2 进行主体一致性评估

## 相关概念

- [[WBench]]
- [[DINOv2]]
