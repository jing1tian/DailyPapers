---
type: concept
aliases: [IntPhys 2, 物理直觉理解基准]
---

# IntPhys2

## 定义
Intuitive Physics 2，评测视觉模型物理直觉理解能力的 benchmark，要求模型判断视频中的场景是否符合物理规律（稳定性、遮挡、连续性等）。

## 核心要点
1. 测试模型是否具备"婴儿物理学"直觉（object permanence、gravity、solid/liquid区分）
2. 设计为二选一判断任务：物理合规视频 vs. 违反物理的视频
3. 用于评测世界模型的物理理解能力（[[PhiZero]] 使用此 benchmark）
4. 与 [[WorldModelBench]] 等更复杂的 benchmark 互补

## 相关概念
- [[WorldModelBench]]
- [[PhiZero]]
