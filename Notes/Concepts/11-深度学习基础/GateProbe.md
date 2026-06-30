---
type: concept
aliases: [Gate Probe, 门控探针灵敏度]
---

# GateProbe

## 定义
一次性 Transformer block 灵敏度估计方法：在每个 block 的输入/输出之间插入可学习的虚拟门控权重，对下游任务 loss 求梯度，以梯度幅值排序 block 重要性，无需逐块删除重训。

## 核心要点
1. 避免 O(N²) 的逐块消融代价，单次前向+反向即可排序所有 block
2. 用于 [[DTR]]（Drop-Then-Recovery）框架中筛选冗余 VLA block
3. 与 [[VLA-Pruner]] 的剪枝思路互补（一个排序、一个剪枝）

## 代表工作
- [[DTR]]: Drop-Then-Recovery，2026

## 相关概念
- [[VLA-Pruner]]
- [[VLA]]
