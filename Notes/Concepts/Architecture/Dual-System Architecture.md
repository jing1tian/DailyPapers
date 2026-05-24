---
type: concept
aliases: [双系统架构, 双流架构]
---

# Dual-System Architecture

## 定义
将机器人策略分为高层语义理解系统（慢系统）和低层动作生成系统（快系统）的两阶段架构，受认知科学中"系统 1/系统 2"思想启发。

## 数学形式
$$
z = f_{high}(o_t, l), \quad a_{t:t+k} = f_{low}(z, q_t)
$$
高层系统 $f_{high}$ 提取语言-视觉语义特征 $z$，低层系统 $f_{low}$ 基于 $z$ 和机器人状态 $q_t$ 生成动作序列。

## 核心要点
1. 高层系统通常为 VLM（如 InternVL、LLaVA），负责场景理解和任务规划
2. 低层系统通常为扩散模型或 Transformer，负责精细动作生成
3. 两系统解耦使得可以分别优化语义理解和运动控制能力

## 代表工作
- [[RoVLA]]: InternVL3.5-2B（高层）+ 32层 Diffusion Transformer（低层）
- [[π0]]: 语言模型（高层）+ 扩散动作专家（低层）
- [[GR00T-N1.6]]: NVIDIA 双系统 VLA

## 相关概念
- [[VLA]]
- [[Diffusion Transformer]]
- [[Action Chunking]]
- [[Flow Matching]]
