---
type: concept
aliases: [Gemini Robotics On-Device, GROD-3B]
---

# GROD（Gemini Robotics On-Device）

## 定义

GROD 是 Google DeepMind 开发的设备端机器人 VLA 模型，采用 [[PaliGemma]] 作为视觉-语言 backbone，配合 [[Diffusion Policy|扩散策略头]] 预测 [[Action Chunking|动作块]]，在大规模 Aloha 机器人遥操作数据上训练，能够以自然语言指令控制双臂机器人执行原子操作技能。

## 核心要点

1. **规模**: 3B 参数（GROD-3B 变体）
2. **架构**: [[PaliGemma]] VLM backbone + [[Diffusion Policy|扩散策略头]]
3. **训练数据**: 仿真 Aloha 双臂机器人遥操作数据集
4. **输出**: 以 [[Action Chunking|动作块]] $a_{0:K-1}$ 形式输出关节控制指令
5. **强项**: 训练分布内的原子技能成功率高；对多种指令格式鲁棒
6. **弱项**: 组合目标、长序列任务、分布外推理任务成功率低

## 代表工作

- [[EXIMO]]: 以 GROD 为基础 VLA，通过 VLM 编排 + SFT + 残差 RL 实现分布外任务适配

## 相关概念

- [[PaliGemma]]
- [[Diffusion Policy]]
- [[Action Chunking]]
- [[VLA（视觉-语言-动作模型）]]
- [[VLM]]
