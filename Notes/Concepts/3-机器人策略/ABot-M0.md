---
type: concept
aliases: [ABot M0, ABot-M0]
---

# ABot-M0

## 定义

ABot-M0 是一个基于 [[Qwen3-VL]] 骨干的 [[视觉语言动作模型|VLA]] 策略模型，采用 [[流匹配|Flow Matching]] 作为动作生成框架，输出 [[动作分块|动作块]]，面向通用机器人操纵任务设计。

## 核心要点

1. 以 [[Qwen3-VL]] 为视觉语言感知骨干，处理视觉观测和语言指令
2. 使用 Flow Matching 去噪器生成连续动作块 $\hat{A}_{\theta,t}$
3. 在 LIBERO-Plus 基准上 Total 成功率为 80.5%，Real-robot 四项任务 ID 成功率 50-65%
4. 作为 [[WorldPilot]] 的策略基础模型，通过双路径先验注入进一步提升 OOD 鲁棒性

## 代表工作

- [[WorldPilot]]: 以 ABot-M0 为基础，叠加 World-Action 先验（Latent Steering + Action Steering）

## 相关概念

- [[视觉语言动作模型]]
- [[Qwen3-VL]]
- [[流匹配]]
- [[动作分块]]
- [[世界动作模型]]
