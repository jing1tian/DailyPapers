---
type: concept
aliases: [SFT, 监督微调, Instruction Fine-Tuning]
---

# Supervised Fine-Tuning

## 定义

Supervised Fine-Tuning（SFT）是在预训练语言模型上，用标注的输入-输出对继续训练，使模型在特定任务或格式上对齐人类期望行为的方法。

## 数学形式

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\sum_{(x,y) \in \mathcal{D}} \sum_{j} \log \pi_\theta(y_j \mid x, y_{<j})
$$

其中 $x$ 为输入，$y$ 为目标输出，loss 只计算在 $y$（助手回复）上。

## 核心要点

1. 只在目标 token（助手回复部分）上计算 loss，输入部分掩码；
2. 是 RLHF/RLAIF 的前置步骤，也可单独作为对齐方法；
3. 数据质量远比数量重要（Alpaca → ShareGPT → RLHF 的演化）；
4. 在机器人场景中，SFT 数据可通过专家演示或强模型合成。

## 代表工作

- [[CaPtain]]: 用 GPT-6 Astra 合成轨迹对 Qwen3.5-4B 做 SFT，再接 RL
- InstructGPT: SFT + RLHF 的经典范式
- Guava: 对 Qwen3.5-4B 主要用 SFT 做 CaP 机器人训练

## 相关概念

- [[Reinforcement Learning]]
- [[Group Relative Policy Optimization]]
- [[Proximal Policy Optimization]]
