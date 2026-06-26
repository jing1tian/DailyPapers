---
type: concept
aliases: [Conservative Q-Learning, 保守 Q 学习]
---

# CQL (Conservative Q-Learning)

## 定义
Offline RL 算法，通过在 Q 函数的 Bellman 更新中加入惩罚项，抑制对 OOD（Out-of-Distribution）动作的 Q 值高估，从而安全地从离线数据中学习策略。

## 数学形式
CQL 在标准 Bellman 损失基础上加入约束：
$$\min_Q \alpha \left(\mathbb{E}_{s\sim\mathcal{D},a\sim\pi}[Q(s,a)] - \mathbb{E}_{(s,a)\sim\mathcal{D}}[Q(s,a)]\right) + \frac{1}{2}\mathcal{L}_{Bellman}$$

## 核心要点
1. 解决 offline RL 的 distributional shift 问题
2. 对 in-distribution 动作保持正常 Q 值，对 OOD 动作施加惩罚
3. offline-to-online 迁移时仍面临分布偏移挑战
4. 与 [[SAC]] 结合使用广泛

## 代表工作
- [[RankQ]]：指出 CQL 在 offline-to-online 转换时的不稳定性，提出 ranking 替代方案
- [[FORCE]]：将 CQL 作为纯 offline RL baseline 对比，在 PullCubeTool、PlaceSphere 等任务上成功率为 0，凸显纯保守 Q 学习在 online 转换上的局限

## 相关概念
- [[SAC]]
- [[DP]]
