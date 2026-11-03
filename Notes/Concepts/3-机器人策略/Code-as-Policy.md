---
type: concept
aliases: [CaP, 代码即策略, Code as Policies]
---

# Code-as-Policy

## 定义

Code-as-Policy 是一种机器人控制范式：用语言模型生成可执行的 Python 代码，调用感知和运动 API 来完成操作任务，而非直接预测关节角度或末端执行器位置。

## 数学形式

$$
a_t = \text{execute}(\pi_\theta(h_t))
$$

其中 $\pi_\theta$ 生成代码，`execute` 在 harness 中解释执行，$h_t$ 为历史观测-动作序列。

## 核心要点

1. 语言模型作为代码生成器，调用预定义感知/运动 API；
2. 代码天然可组合、可调试、可迁移到新场景；
3. 早期方法依赖闭源前沿模型（GPT-4），近期工作尝试训练小型开放模型；
4. 需要配套的 harness 处理 IK、碰撞保护、执行反馈。

## 代表工作

- [[CaPtain]]: SFT + 物理 Critic RL 训练 4B VLM 做端到端 CaP 智能体
- Code as Policies (Liang et al.): 最早用 LLM 生成机器人代码
- VoxPoser: 用 VLM 生成空间值函数

## 相关概念

- [[ReAct]]
- [[Vision-Language-Action]]
- [[Reinforcement Learning]]
- [[Inverse Kinematics]]
