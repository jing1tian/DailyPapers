---
type: concept
aliases: [GR00T-N1.5, GROOT N1.5, GR00T N1.5]
---

# GR00T N1.5

## 定义

NVIDIA 发布的 Flow Matching 视觉-语言-动作模型，基于 GR00T N1 改进，具备动作专家架构，用于通用机器人操作任务。

## 核心要点

1. 采用 [[Flow Matching|流匹配]] 框架，与 [[Pi0.5]] 同属该类型
2. 在 RoboCasa 等 benchmark 上表现较强（baseline 0.59）
3. 包含独立动作专家模块，是 COAST 引导作用的目标位置

## 代表工作

- [[COAST]]: 在 GR00T N1.5 上验证 Conceptor 激活引导（RoboCasa: 0.59→0.75）
- [[G3VLA]]: 在 GR00T 1.5 双塔架构上验证几何旁路注入，发现收益被削弱甚至出现混合/负向结果，归因于几何 token 需跨越额外注意力瓶颈才能到达动作生成模块——揭示几何归纳偏置的有效性强烈依赖于架构设计（单塔 vs. 双塔）

## 相关概念

- [[VLA]]
- [[Flow Matching]]
- [[GR00T]]
- [[Pi0.5]]
