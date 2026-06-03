---
type: concept
aliases: [DreamGen]
---

# DreamGen

## 定义
用视频扩散模型合成机器人操作演示数据的框架，将真实场景外观迁移到新任务配置，实现低成本数据扩充。

## 核心要点
1. 以少量真实演示为种子，生成大量风格迁移后的合成演示
2. 保留动作轨迹，只改变视觉外观
3. 配合 RoboEngine 等仿真渲染管线使用

## 代表工作
- [[RoboDream]]: 组合式扩展，分解物体/背景分别生成

## 相关概念
- [[RoboEngine]]
- [[DreamDojo]]
- [[AnchorDream]]
