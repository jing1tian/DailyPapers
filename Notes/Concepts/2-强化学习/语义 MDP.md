---
type: concept
aliases: [Semantic MDP, 语义马尔可夫决策过程, Semantic Markov Decision Process]
---

# 语义 MDP

## 定义

将标准 MDP 的动作空间替换为自然语言语义动作空间的形式化框架，通过预训练语言条件策略（如 VLA）作为转换层，将语义动作映射为低级机器人控制动作。

## 数学形式

标准 MDP：$\mathcal{M} = (\mathcal{S}, \mathcal{A}_{\text{robot}}, P, P_0, \gamma, r)$

语义 MDP：$\mathcal{M}_{\text{sem}} = (\mathcal{S}, \mathcal{A}_{\text{sem}}, P_{\text{sem}}, P_0, \gamma, r)$

其中语义转移函数：

$$
P_{\text{sem}}(s' \mid s, \ell) = \int P(s' \mid s, a) \, \pi_{\text{vla}}(a \mid s, \ell) \, da
$$

VLA 策略 $\pi_{\text{vla}}$ 充当语义动作 $\ell$ 到低级动作 $a$ 的随机转换层。

## 核心要点

1. **空间提升**：将 RL 的学习问题从低级动作空间提升到语义语言空间
2. **VLA 即转换器**：预训练 VLA 充当语义动作到物理动作的转换器，参数保持冻结
3. **奖励保留**：任务奖励函数 $r$ 不变，仍定义在状态空间上
4. **技能组合**：通过选择不同语义动作，可在 VLA 技能先验内进行组合，解决零样本提示失败的长时域任务

## 代表工作

- [[SARL]]: 首次系统提出语义 MDP 形式化，并在此基础上开发 Q-learning 适应算法

## 相关概念

- [[Markov Decision Process]]
- [[语义动作空间]]
- [[视觉语言动作模型]]
- [[Q-Learning]]
- [[强化学习]]
