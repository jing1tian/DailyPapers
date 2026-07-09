---
type: concept
aliases: [Model-Based Reinforcement Learning, 基于模型的强化学习]
---

# MBRL

## 定义
Model-Based Reinforcement Learning（基于模型的强化学习）：智能体学习一个环境动力学模型，并利用该模型进行规划或生成模拟经验，而非直接从真实环境交互中学习。

## 数学形式
$$\hat{s}_{t+1} = f_\theta(s_t, a_t), \quad \pi^* = \arg\max_\pi \mathbb{E}_{\hat{\tau} \sim f_\theta}[R(\hat{\tau})]$$

其中 $f_\theta$ 是学习到的动力学模型，$\hat{\tau}$ 是模拟轨迹。

## 核心要点
1. 与 Model-Free RL 的核心区别：有显式的世界模型 $f_\theta$，可用于样本更高效的规划
2. 动力学模型可以是确定性的（前向预测）或随机的（分布建模）
3. 复合误差（compounding error）是核心挑战：模型误差在长规划步骤中指数累积
4. 在 World Model 语境下，MBRL 是具身AI世界模型的理论基础

## 代表工作
- [[PlaNet]]、[[DreamerV3]]：潜空间 MBRL
- [[MuZero]]：无模型感知的 MBRL（学习 value-equivalent 模型）
- [[RSSM]]：Recurrent State Space Model，Dreamer 系列骨干

## 相关概念
- [[WAM]]（World Action Model：在 MBRL 基础上联合建模动作）
- [[WM-Roadmap]]（世界模型定义文档）
