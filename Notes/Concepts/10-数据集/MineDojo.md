---
type: concept
aliases: [MineDojo-300, MineDojo benchmark]
---

# MineDojo

## 定义
MineDojo 是基于 Minecraft 游戏构建的开放世界 AI 研究平台和 benchmark，包含大规模视频数据集和多样化任务套件，用于训练和评测通用游戏 agent。

## 数学形式
$$\text{success rate} = \frac{\text{tasks completed}}{N_{\text{tasks}}}$$

## 核心要点
1. 基于 Minecraft 提供开放世界环境，支持复杂长视野任务
2. MineDojo-300：包含 300 个程序化生成任务的评测 benchmark
3. 配套大规模视频和技术文档数据集（YouTube、Wiki、Reddit）
4. 常用于评测 game agent 的泛化能力

## 代表工作
- [[VPT]]（Video PreTraining）：第一个在 MineDojo 上展示 behavioral cloning 效果的工作
- [[Player-1]]：在 MineDojo-300 上达到 38.33% success rate

## 相关概念
- [[ViZDoom]]
- [[GameWAM]]
- [[GameWorld]]
