---
type: concept
aliases: [扩散蒸馏, Score Distillation, Step Distillation]
---

# Diffusion Distillation

## 定义

扩散蒸馏（Diffusion Distillation）是将需要多步迭代的扩散模型压缩为少步推理模型的技术，通过教师-学生蒸馏框架，使学生模型用极少步数（2-8 步）生成与教师模型（50-1000 步）质量相当的结果。

## 核心要点

1. **动机**: 扩散模型推理需要多次前向传播，延迟高；蒸馏可将推理步数从 100+ 降低到 4 步以内
2. **主要方法**:
   - **Consistency Distillation**: 学习从任意噪声级别直接预测干净样本
   - **Progressive Distillation**: 逐步减半推理步数（100→50→25→...）
   - **Distribution Matching Distillation (DMD)**: 通过分布匹配实现单步生成
3. **两阶段蒸馏**: 先从 100 步蒸馏到中间步数，再蒸馏到 4 步，质量损失更小

## 数学形式

一致性蒸馏目标（简化）：

$$
\mathcal{L}_\text{CD} = \mathbb{E}\left[\|f_\theta(x_{t_1}) - f_{\theta^-}(x_{t_2})\|^2\right]
$$

其中 $x_{t_1}, x_{t_2}$ 是同一干净样本对应的不同噪声级别，$\theta^-$ 为 EMA 教师参数。

## 代表工作

- [[HyNomas]]: 两阶段蒸馏将视频世界模型推理步数从 100 压缩到 4，支持单张 RTX 4090 运行
- [[DMD2]]: Distribution Matching Distillation 改进版
- [[Consistency Distillation]]: 一致性蒸馏经典工作

## 相关概念

- [[Consistency Distillation]]
- [[Flow Matching]]
- [[Diffusion Model]]
