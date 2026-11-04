---
type: concept
aliases: [GRPO, 组相对策略优化]
---

# Group Relative Policy Optimization

## 定义

Group Relative Policy Optimization（GRPO）是一种无需显式价值网络的策略优化方法，通过对同一输入的多个采样结果相互比较（组内归一化），计算相对优势来驱动策略更新。

## 数学形式

对一组 $G$ 个采样 $\{o_i\}_{i=1}^G$，组相对优势为：

$$
A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}
$$

策略损失：

$$
\mathcal{L}_{\mathrm{GRPO}} = -\mathbb{E}\left[\frac{\pi_\theta(o)}{\pi_{\mathrm{old}}(o)} A - \beta\,\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})\right]
$$

## 核心要点

1. 无需单独训练 critic，降低实现复杂度；
2. 通过组内相对比较提供训练信号，需要同一问题多个采样；
3. 适合奖励稀疏、难以训练 critic 的场景（数学推理等）；
4. CaPtain 扩展为混合形式：episode 级 GRPO + 请求级 GAE 组合。

## 代表工作

- [[CaPtain]]: 混合组相对 advantage（episode 级）与 GAE（请求级）
- DeepSeek-R1: 将 GRPO 用于数学推理 RL
- CaP-X: 单轮 GRPO 用于 7B 代码机器人训练

## 相关概念

- [[Reinforcement Learning]]
- [[Proximal Policy Optimization]]
- [[Generalized Advantage Estimation]]
- [[Supervised Fine-Tuning]]
