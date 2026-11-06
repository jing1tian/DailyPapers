---
type: concept
aliases: [Open Human Scale, egocentric pretraining recipe]
---

# Open-HumanScale

## 定义
Egocentric 人类视频到机器人策略的开放规模化配方：提供 50K 数据集、[[IDM]] latent action 提取、两阶段预训练+微调方案，解决 egocentric 视频到机器人 policy 迁移的系统性问题。

## 核心要点
1. Open-HumanScale-50K 数据集：从多个公开 egocentric 数据集整理
2. IDM 提取 latent action，作为无标注视频的伪 action 标注
3. 两阶段：egocentric 预训练 → robot fine-tuning，零样本 policy transfer
4. PKU + NUS + MIT + NVIDIA 联合出品，含真实机器人实验

## 相关概念
- [[IDM]]
- [[EgoDex]]
- [[HumanNet]]
- [[WAM]]
