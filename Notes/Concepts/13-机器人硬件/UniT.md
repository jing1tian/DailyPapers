---
type: concept
aliases: [UniT tactile representation]
---

# UniT

## 定义
数据高效的触觉表征学习方法（arXiv 2408.06481），可泛化到未见过的物体，将触觉传感器图像编码为连续隐变量供下游策略/世界模型使用。

## 核心要点
1. 提供"冻结的触觉表征编码器"角色，被多个触觉 WAM 工作（如 [[Tactile-WAM]]）直接复用，无需端到端联合训练触觉编码器
2. 强调数据效率——在有限触觉数据下也能学到可迁移的接触表征
3. 输出的连续隐变量可直接接入交叉注意力 token 适配器，压缩为固定数量的紧凑 token

## 代表工作
- 原始论文: Xu, Uppuluri, Zhang, Fitch, Crandall, Shou, Wang, She (2024) "UniT: Data Efficient Tactile Representation with Generalization to Unseen Objects"
- [[Tactile-WAM]]: 用 UniT 风格连续触觉隐变量作为触觉表征模型的可选实现

## 相关概念
- [[Action-Horizon-Aligned Tactile Registers]]
- [[Touch-State Proxy]]
