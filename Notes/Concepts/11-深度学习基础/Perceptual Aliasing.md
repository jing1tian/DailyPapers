---
type: concept
aliases: [感知混叠, 观测歧义, Observation Ambiguity, 视觉混叠]
---

# Perceptual Aliasing

## 定义

**感知混叠**（Perceptual Aliasing）是指在序列决策任务中，不同的任务状态（或任务阶段）对应**相同或高度相似的观测**，导致基于当前观测的策略无法正确判断所处状态、从而产生错误决策的现象。

## 数学形式

感知混叠发生在：

$$
o_t \approx o_{t'}, \quad \text{但} \quad s_t \neq s_{t'}
$$

其中 $o_t$ 是观测，$s_t$ 是真实任务状态。此时：

$$
\pi(a \mid o_t) = \pi(a \mid o_{t'}) \quad \Rightarrow \quad \text{错误动作}
$$

## 核心要点

1. **多阶段任务中尤为突出**: 如倒水任务中，倒水前（面板 2）和倒水后（面板 5）透明水杯外观几乎相同
2. **纯视觉策略的根本瓶颈**: 无历史信息的 Markovian 策略在感知混叠场景下不可避免地出现歧义
3. **解决方案**: 引入时序上下文（如[[Temporal Context Buffer|时序上下文缓冲区]]）或显式状态跟踪（如状态机、记忆模块）
4. **与 POMDPs 的联系**: 感知混叠是部分可观测马尔可夫决策过程（POMDP）中信念状态不确定性的具体体现

## 代表工作

- [[Lumo-2]]: 通过[[Temporal Context Buffer|时序上下文缓冲区]]（历史 action token 队列）解决多阶段任务中的感知混叠

## 相关概念

- [[Temporal Context Buffer]]: 利用历史动作序列解决感知混叠的主要手段
- [[Latent World Dynamics]]: 世界动态表征为策略提供额外的状态上下文，辅助消歧
