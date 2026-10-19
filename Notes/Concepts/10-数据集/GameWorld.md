---
type: concept
aliases: [GameWorld benchmark, GameWorld Score]
---

# GameWorld

## 定义
GameWorld 是用于评测游戏世界模型生成质量的 benchmark，覆盖视觉质量、时序一致性、动作可控性和物理规律理解四个维度，主要面向 Minecraft 等游戏场景的 interactive world generation 评测。

## 核心要点
1. 四大评测维度：visual quality、temporal quality、action controllability、physical rule understanding
2. 由 [[Matrix-Game]] 提出的 GameWorld Score 统一评测框架
3. 与 [[MineDojo]] 配合使用，评测 game agent + world model 系统

## 代表工作
- [[Matrix-Game]]：提出 GameWorld benchmark 并在其上展示 SOTA 性能
- [[Player-1]]：在 GameWorld 上评测 game agent 控制能力

## 相关概念
- [[MineDojo]]
- [[Matrix-Game]]
- [[MineWorld]]
