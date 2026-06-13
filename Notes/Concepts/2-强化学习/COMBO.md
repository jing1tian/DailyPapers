---
type: concept
aliases: [Conservative Offline Model-Based RL]
---

# COMBO

## 定义
保守的离线模型基础强化学习算法，通过世界模型生成虚假转移数据，并用保守约束（如 Q-value penalty on OOD actions）防止模型偏差导致的过估计。

## 数学形式
COMBO 的目标函数在真实数据和模型生成数据上联合优化：
$$\max_Q \mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a)] - \lambda \cdot \mathbb{E}_{(s,a) \sim \hat{M}}[Q(s,a)]$$

## 核心要点
1. **真实 + 模型数据混合**：部分数据来自离线 buffer，部分来自 WM rollout
2. **保守约束**：对模型生成的 OOD 转移惩罚 Q 值，防止模型错误放大
3. **vs [[MOPO]]**：MOPO 是悲观 reward shaping，COMBO 是直接约束 Q 值

## 代表工作
- [[WOMBET]]: 使用 COMBO 风格的保守约束做经验迁移 RL

## 相关概念
- [[MOPO]]
- [[SAC]]
- [[MBRL]]
