---
type: concept
aliases: [VGPD, Value-Guided Policy Distillation, 价值引导策略自蒸馏]
---

# VGPD (Value-Guided Policy Distillation)

## 定义

VGPD 是 [[FORCE]] 提出的在线强化学习策略改进机制，将策略更新构造为带 [[KL 散度]] 约束的能量加权自蒸馏问题，并引入动态优势过滤器，只从高于当前价值基线的探索候选动作中学习，从而在没有人类干预的前提下保证策略改进的稳定性与单调性。

## 数学形式

正则化策略改进目标：

$$
\mathcal{J}(\pi) = \mathbb{E}_{s\sim\mathcal{D}}\Big[\mathbb{E}_{a\sim\pi(\cdot|s)}[Q_\theta(s,a)] - \tau\, D_{KL}\big(\pi(\cdot|s)\,\|\,\pi_{old}(\cdot|s)\big)\Big]
$$

解析最优解为能量加权形式 $\pi^*(a|s) \propto \pi_{old}(a|s)\exp(Q_\theta(s,a)/\tau)$，投影回参数化策略等价于加权对数似然最大化：

$$
\max_\phi\ \mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi_{old}}\Big[\exp\big(Q_\theta(s,a)/\tau\big)\,\log \pi_\phi(a|s)\Big]
$$

动态优势过滤器（决定是蒸馏还是回退模仿）：

$$
\zeta(s) = \mathbb{1}\big\{q_{mean}(s) > Q_\theta(s, a_{buf})\big\}, \qquad q_{mean}(s) = \frac{1}{K}\sum_{k=1}^{K} Q_\theta(s, \hat{a}_k)
$$

## 核心要点

1. **能量加权自蒸馏**：用旧策略采样的动作按 Q 值指数加权，构造加权对数似然训练新策略，Q 值越高权重越大
2. **动态优势过滤**：候选动作的平均价值必须超过 Policy Buffer 中历史已执行动作的价值才会启用蒸馏，否则退回模仿历史动作，避免向更差方向更新
3. **二次过滤 + 归一化**：在通过过滤的候选中进一步只保留高于均值基线的样本，做指数加权 softmax 归一化
4. **自然涌现的课程学习**：训练初期策略候选质量参差，更多回退模仿；后期候选质量提升，蒸馏机制逐渐主导
5. **温度系数 $\tau$**：控制 KL 约束强度，类似 [[近端策略优化|信赖域策略优化]]中的约束力度

## 代表工作

- [[FORCE]]: 提出 VGPD，作为三阶段框架（[[Cal-QL]] 离线预训练 → 分布式 Warm-up → VGPD 在线微调）的核心在线模块

## 相关概念

- [[KL 散度]]
- [[强化学习]]
- [[Actor-Critic]]
- [[近端策略优化]]
- [[Cal-QL]]
