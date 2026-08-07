---
type: concept
aliases: [因果对齐, Causal Alignment]
---

# Causal Alignment（因果对齐）

## 定义

在具有链式推理（CoT）的 VLA 模型训练中，要求推理文本与任务成功之间建立明确的因果关联——仅靠 SFT 模仿标注的 CoT 文本无法建立此联系，需要使用基于任务成功结果的强化学习（outcome-based RL）才能使 CoT 从"装饰性文本"升级为"功能性规划信号"。

## 数学形式

$$
\mathcal{R}(\tau) = \alpha_s \cdot \mathbb{1}_{\text{success}} + \alpha_f \cdot \mathbb{1}_{\text{format}}
$$

稀疏的任务成功奖励直接连接 CoT 推理质量与最终执行结果，通过 [[GRPO]] 梯度反向传播到推理 token，建立因果关系。

## 核心要点

1. **SFT 的局限**: 纯监督微调的 CoT 只是在模仿标注文本，在动作执行 OOD 场景下，SFT-only 模型性能下降 32.0pp，与无推理基线（31.6pp）几乎无差别
2. **RL 的作用**: 通过任务成功奖励反向优化推理 token，迫使模型学习"推理质量影响任务成功"的因果链；RL 对齐后 OOD 下降缩小至 24.4pp
3. **分组信用分配**: 使用 GRPO 将 episode 级稀疏奖励归一化为 token 级优势，解决稀疏奖励的信用分配难题
4. **与 Decoding Alignment 的关系**: 两者是 CoT 有效工作的必要条件，缺一不可

## 代表工作

- [[DeepThinkVLA]]: 提出 Causal Alignment 概念并通过两阶段流水线（SFT 冷启动 + GRPO RL）实现

## 相关概念

- [[Decoding Alignment]]: 配套的第一个必要条件——架构层面保证模态解码方式正确
- [[GRPO]]: 实现因果对齐的 RL 算法
- [[Chain-of-Thought Reasoning]]: 被对齐的推理范式
- [[强化学习]]: 因果对齐依赖 RL 训练
- [[奖励函数]]: 任务成功奖励是建立因果联系的核心信号
