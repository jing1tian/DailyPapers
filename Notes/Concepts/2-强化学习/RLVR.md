---
type: concept
aliases: [Reinforcement Learning with Verifiable Rewards, 可验证奖励强化学习]
---

# RLVR

## 定义
Reinforcement Learning with Verifiable Rewards：一种 LLM 后训练范式，奖励信号来自可程序化验证的正确性标准（如数学答案、代码执行结果），而非人类偏好标注或奖励模型打分。

## 数学形式
$$r(y, y^*) = \mathbf{1}[\text{verify}(y) = y^*]$$

奖励为 0/1 二值信号，$\text{verify}(\cdot)$ 是确定性的正确性判断函数。

## 核心要点
1. 避免了 reward hacking（奖励模型被 policy 欺骗）的问题
2. 适合有明确答案的任务：数学、代码、逻辑推理
3. 常与 GRPO、PPO 等 policy gradient 算法结合使用
4. RSIPO 在 RLVR 基础上加入 self-distillation（OPSD），利用 privileged context 教师蒸馏生成 dense token 监督

## 代表工作
- [[RSIPO]]: 在 RLVR 框架上加 recursive self-improving，通过技能策略协同进化

## 相关概念
- [[GRPO]]: 常用于 RLVR 的 policy optimization 算法
- [[PPO]]: 另一种常用 policy gradient 算法
