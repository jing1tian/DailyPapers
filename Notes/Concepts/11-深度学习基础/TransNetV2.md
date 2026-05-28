---
type: concept
aliases: [TransNetV2, TransNet V2]
---

# TransNetV2

## 定义

TransNetV2 是用于视频场景切割检测（shot boundary detection）的深度学习模型，可精确识别视频中的硬切（hard cut）和软切（soft cut）转场，在 WBench 中用于评估视频片段连续性。

## 核心要点

1. **场景切割检测**: 识别视频中因镜头切换产生的突变帧
2. **连续性评估**: 在 [[WBench]] 的 C.3 Segment Continuity 指标中，统计无切割帧的比例
3. **实时高效**: 在长视频序列上可高效运行

## 代表工作

- Souček & Lokoč 2021: TransNetV2 原始论文
- [[WBench]]: 使用 TransNetV2 检测生成视频中的场景切割

## 相关概念

- [[WBench]]
