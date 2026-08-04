---
type: concept
aliases: [受控马尔可夫过程, CMP, 受控马尔科夫过程]
---

# Controlled Markov Process

## 定义

受控马尔可夫过程是马尔可夫决策过程的连续/一般化形式，由三元组 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P)$ 定义：智能体在状态 $s_t$ 执行动作 $a_t$，环境按转移核 $P$ 产生下一状态 $s_{t+1}$，且转移仅依赖当前状态和动作（马尔可夫性质）。

## 数学形式

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P), \quad s_0 \sim \rho_T, \quad s_{t+1} \sim P(\cdot \mid s_t, a_t)
$$

与 [[Markov Decision Process]] 的区别：CMP 不包含奖励函数，关注的是状态-动作轨迹的分布而非累积奖励最大化。

## 核心要点

1. **马尔可夫性**: $s_{t+1}$ 仅依赖 $(s_t, a_t)$，与历史 $s_{0:t-1}$ 无关
2. **无奖励**: CMP 是纯动力学模型，适合世界模型和模仿学习框架
3. **感知映射**: 实际系统中物理状态 $s_t$ 通过感知映射 $\mathcal{O}$ 和编码器 $E$ 得到潜状态 $z_t = E(\mathcal{O}(s_t))$
4. **策略诱导分布**: 给定策略 $\pi$，状态-动作序列形成马尔可夫链

## 代表工作

- [[FBFM]]: 将机器人操作环境建模为受控马尔可夫过程，WAM 基于交互历史 $\mathcal{H}_t$ 预测状态-动作块

## 相关概念

- [[Markov Decision Process]]
- [[WAM]]
- [[POMDP]]
