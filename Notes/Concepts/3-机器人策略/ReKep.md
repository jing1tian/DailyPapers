---
type: concept
aliases: [Relational Keypoint Constraints]
---

# ReKep

## 定义
基于关系关键点约束的机器人操作方法，用 VLM 从视觉输入中提取关键点及其空间关系约束，然后用优化方法生成满足约束的机器人轨迹。

## 核心要点
1. 用 VLM 自动提取关键点（无需人工标注）
2. 约束以关键点间相对位置关系表达
3. 轨迹生成通过满足约束的优化求解
4. 支持 language-conditioned 任务描述

## 代表工作
- [[GenVid2Robot]]：用 ReKep 做基于视频的关键点约束提取

## 相关概念
- [[VoxPoser]]
- [[AnyGrasp]]
