---
type: concept
aliases: [Prismatic-7B, Prismatic VLM]
---

# Prismatic-7B

## 定义

Prismatic-7B 是一种 70 亿参数的视觉-语言模型（VLM），作为 OpenVLA 和 EventVLA 的骨干网络，支持多模态输入并以语言模型为核心处理视觉-语言-动作任务。

## 核心要点

1. 70 亿参数的大规模多模态语言模型
2. 在 [[OpenVLA]] 中用作 VLA backbone，经过机器人操控数据微调后具备动作预测能力
3. [[EventVLA]] 保持其权重冻结，仅在外部添加事件融合模块（门控跨注意力）
4. 支持 RGB 图像、语言指令、本体状态的联合处理

## 代表工作

- [[OpenVLA]]: Prismatic-7B 的主要 VLA 应用，提供预训练权重
- [[EventVLA]]: 以 Prismatic-7B（OpenVLA 权重）为冻结 backbone，外接事件融合分支

## 相关概念

- [[VLA]]: Prismatic-7B 所支持的任务类型
- [[OpenVLA]]: 基于 Prismatic-7B 的 VLA 模型
- [[门控跨注意力]]: 在 Prismatic-7B 输出后插入的事件融合机制
