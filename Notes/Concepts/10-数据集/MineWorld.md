---
type: concept
aliases: [MineWorld, Minecraft World Model]
---

# MineWorld

## 定义
基于 Minecraft 环境的交互式世界模型/benchmark，在 HarnessEval-W 中作为被评测的 18 个世界模型之一，代表基于游戏引擎的 world model 路线。

## 数学形式
世界模型输入-输出：$\hat{o}_{t+1} = f_\theta(o_{1:t}, a_{1:t})$，在 Minecraft 视觉环境中预测下一帧观测。

## 核心要点
1. 利用 Minecraft 的结构化物理规则提供低成本但高控制度的 world model 训练环境
2. 游戏引擎数据天然具备 GT 标注，适合闭环评估
3. 在 HarnessEval-W 的 18 个模型对比中是代表性成员

## 代表工作
- [[HarnessEval-W]]: 对包括 MineWorld 在内的 18 个 world model 进行 agentified 评测

## 相关概念
- [[WorldScore]]: world model 评测 benchmark
- [[HarnessEval-W]]: 层次化 agent 评测框架
