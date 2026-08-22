---
type: concept
aliases: [BC, 行为克隆, 模仿学习, Imitation Learning]
---

# Behavior Cloning（行为克隆）

## 定义

Behavior Cloning（BC）是最基础的模仿学习方法，将策略学习转化为有监督学习问题：直接在专家演示数据集 $\mathcal{D} = \{(s_i, a_i)\}$ 上最小化策略输出与专家动作之间的损失，而不涉及任何环境交互或奖励信号。

## 数学形式

$$
\mathcal{L}_{BC}(\theta) = \mathbb{E}_{(s, a) \sim \mathcal{D}}\!\left[\ell\!\left(\pi_\theta(s),\, a\right)\right]
$$

其中 $\ell$ 为任务相关损失（连续动作用 MSE，离散动作用交叉熵，扩散策略用噪声预测损失）。

## 核心要点

1. **Covariate Shift 问题**: 训练时分布为专家状态，测试时策略自身生成状态，累积误差导致分布偏移
2. **数据高效**: 相比 RL 无需大量环境交互，但严重依赖高质量演示覆盖度
3. **遥操作瓶颈**: 大规模 VLA 通常依赖人工遥操作采集 BC 数据，成本高、难以覆盖长尾任务
4. **SFT 变体**: 在 LLM/VLA 场景下，BC 即监督微调（Supervised Fine-Tuning, SFT）

## 代表工作

- [[EXIMO]]: Imitate 阶段在 VLM 编排数据上做 BC/SFT，将 VLM 知识蒸馏回 VLA

## 相关概念

- [[Diffusion Policy]]
- [[Action Chunking]]
- [[VLA（视觉-语言-动作模型）]]
- [[Residual Policy]]
