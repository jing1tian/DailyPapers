---
type: concept
aliases: [Reinforcement Learning, RL, 强化学习, Model-Based RL, MBRL]
---

# Reinforcement Learning（强化学习）

## 定义

智能体通过与环境交互、获取奖励信号来学习最优策略的机器学习范式。在机器人学习中，RL 常与世界模型结合，使用学习型仿真器替代真实环境进行训练。

## 数学形式

**策略优化目标**:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \gamma^t r_t\right]
$$

**世界模型仿真器中的 RL**（Model-Based RL）:

$$
(\hat{o}_{t+1},\, \hat{r}_t,\, \hat{d}_t) \sim p_\phi(\cdot \mid o_{\leq t},\, a_{\leq t},\, l)
$$

$$
J(\theta) = \mathbb{E}_{\hat{\tau} \sim (\pi_\theta, p_\phi)}\!\left[\sum_t \gamma^t \hat{r}_t\right]
$$

**PPO/GRPO 策略损失**:

$$
\mathcal{L}_{\text{RL}}(\theta) = -\mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]
$$

## 核心要点

1. **Model-Free RL**: 直接从真实交互中学习策略（PPO、SAC），样本效率低但无模型误差
2. **Model-Based RL (MBRL)**: 学习世界模型作为仿真器，在想象中训练策略，样本效率高
3. **两级范式**（机器人世界模型 RL）:
   - **第一级**: 固定世界模型作为环境，训练策略（World-Env、VLA-RFT、WMPO）
   - **第二级（协同进化）**: 策略与世界模型交替更新（World-VLA-Loop、VLAW、WoVR）
4. **奖励稀疏问题**: 机器人操作任务中奖励信号稀疏，世界模型可提供额外奖励预测头

## 代表工作

- [[WMRobotSurvey]]: 系统综述世界模型作为 RL 仿真器的两级范式
- [[WAMSurvey]]: WAM 的 RL 应用（奖励建模、策略评估）
- [[DreamerV3]]: 通用 MBRL 框架，世界模型学习潜在动力学

## 相关概念

- [[World Model]]: MBRL 中用于替代真实环境的学习型仿真器
- [[Diffusion Policy]]: 基于扩散的机器人策略，可在 RL 框架中优化
- [[Action Chunking]]: RL 策略的动作输出形式
