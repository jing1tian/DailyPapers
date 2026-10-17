---
type: concept
aliases: [闭环策略学习, Closed-Loop Policy Learning, 闭环改进, Closed-Loop Improvement]
---

# Closed-Loop Policy Learning

## 定义

闭环策略学习是一种迭代式机器人学习范式：策略部署产生的交互轨迹（含成功、次优、失败）被收集后重新用于策略再训练，形成"部署→收集→训练"的持续改进循环。

## 核心流程

$$
\text{策略} \xrightarrow{\text{部署}} \text{Rollout 数据} \xrightarrow{\text{质量标注}} \text{再训练} \xrightarrow{} \text{改进策略}
$$

## 核心要点

1. **自我数据生成**: 无需持续的人工示范，利用部署本身产生训练数据
2. **混合质量处理**: 关键挑战是如何避免从失败/次优轨迹中学习错误行为
3. **迭代提升**: 每轮循环后策略质量单调提升（理论上）
4. **与 RL 的区别**: 闭环策略学习通常基于模仿学习框架，以质量过滤/加权替代奖励信号

## 代表工作

- [[LoopWAM]]: 通过源相关噪声采样和质量加权损失，在三轮闭环迭代中将真实机器人成功率从 23.67% 提升至 43.33%

## 相关概念

- [[World Action Model]]
- [[Imitation Learning]]
- [[Quality Steering]]
