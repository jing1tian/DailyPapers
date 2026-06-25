---
type: concept
aliases: [Visual Navigation Transformer]
---

# ViNT

## 定义
通用视觉导航基础模型，用 Transformer 在大规模跨机器人形态、跨环境的导航数据上训练，以目标图像为条件预测导航动作/子目标，可零样本迁移到新机器人和新场景。

## 核心要点
1. 跨平台、跨数据集的统一导航策略，强调通用性而非单一机器人专用
2. 以视觉目标（goal image）而非地图或坐标作为导航条件
3. 常与 [[NoMaD]]、[[RECON]] 一起作为视觉目标条件导航方法的代表/对比基线

## 相关概念
- [[NoMaD]]
- [[RECON]]
