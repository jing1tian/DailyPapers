---
type: concept
aliases: [NWM, 导航世界模型, Navigation World Models]
---

# Navigation World Model

## 定义

以动作为条件、预测未来自中心视觉观测的生成模型，用于目标条件视觉导航。给定当前观测和动作序列，预测智能体 H 步后的未来视图，通过评分候选动作的未来预测来引导导航规划。

## 数学形式

$$
p_\theta(o_{t+H} \mid o_t, a_{t:t+H-1})
$$

传统方法需要外部规划器（如 [[Cross-Entropy Method|CEM]]）在推理时搜索最优动作：

$$
a^* = \arg\max_{a} \text{Score}(p_\theta(o_{t+H} \mid o_t, a))
$$

## 核心要点

1. **视觉预见性**: 通过预测未来观测，为导航提供可解释的"想象"能力
2. **规划依赖**: 传统 NWM 只是预测模块，需外部 CEM 搜索动作（计算代价极高，~234 秒/动作）
3. **解耦问题**: 预测模块与规划模块分离，误差会在两个模块间传播，难以端到端优化
4. **进化方向**: [[NavWAM]] 通过将动作直接放入共享潜在 Canvas，消除了对外部规划器的依赖

## 代表工作
- [[NWM]]: 导航世界模型的经典实现，使用 CEM 搜索动作
- [[NavWAM]]: 将导航世界模型直接转化为闭环策略，无需 CEM

## 相关概念
- [[Cross-Entropy Method]]: NWM 推理时常用的动作搜索方法
- [[World Model]]: 广义世界模型，NWM 是其在导航领域的特化
- [[Latent Canvas]]: NavWAM 中替代传统 NWM+规划器的统一框架
- [[visual foresight]]: 视觉预见性，NWM 的核心能力
