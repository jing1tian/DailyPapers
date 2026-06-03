---
type: concept
aliases: [MimicGen]
---

# MimicGen

## 定义
通过对少量人类演示做自动轨迹变换，大规模合成机器人操作演示数据的系统。

## 核心要点
1. 以物体为中心的轨迹变换：提取子任务片段，在新物体位置重新对齐
2. 从几十条演示生成数千条合成演示
3. RoboDream 等数据合成工作将其作为对比基线

## 代表工作
- [[RoboDream]]: 对比 MimicGen，RoboDream 用 WM 合成视觉层而非轨迹变换

## 相关概念
- [[DreamGen]]
- [[AnchorDream]]
