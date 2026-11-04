---
type: concept
aliases: [Probabilistic Ensemble Trajectory Sampling]
---

# PETS（Probabilistic Ensemble Trajectory Sampling）

## 定义
基于概率集成动力学模型的 model-based RL 方法，用集成神经网络建模环境转移分布，通过 CEM 在模型 rollout 中优化动作序列。

## 数学形式
$$a^* = \arg\max_a \mathbb{E}_{f \sim \mathcal{E}} \sum_t r(s_t, a_t), \quad s_{t+1} \sim f(s_t, a_t)$$

其中 $\mathcal{E}$ 为集成动力学模型，通过 CEM 迭代优化动作序列。

## 核心要点
1. 集成建模（通常 5 个模型）捕捉认知不确定性，降低模型滥用（model exploitation）
2. 概率网络（Gaussian head）捕捉偶然不确定性
3. 推理时用 CEM（交叉熵方法）做 MPC 规划，不需要额外 policy 网络

## 代表工作
- Chua et al. 2018 (NeurIPS): 原始 PETS 论文
- [[pico-WAM]]: 以 PETS 作为对比基线

## 相关概念
- [[MBPO]]
- [[CEM]]
- [[RSSM]]
