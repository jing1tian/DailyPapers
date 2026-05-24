---
type: concept
aliases: [Matrix Game World Model]
---

# Matrix-Game

## 定义
一种以玩家为中心的游戏世界模型，通过扩散模型预测游戏未来帧，但将 NPC 仅作为背景像素处理，缺乏对 NPC 行为的主动建模。

## 核心要点
1. 玩家中心视角：世界模型条件化于玩家动作
2. NPC 作为被动背景，无策略行为
3. 被 [[ReactiveGWM]] 作为对比基线，展示 NPC 策略驱动扩展的必要性

## 代表工作
- [[ReactiveGWM]]: 扩展 Matrix-Game 范式，使 NPC 具有策略驱动行为

## 相关概念
- [[World Model]]
- [[ReactiveGWM]]
