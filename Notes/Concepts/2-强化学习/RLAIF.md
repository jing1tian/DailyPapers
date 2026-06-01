---
type: concept
aliases: [RL from AI Feedback, AI 反馈强化学习]
---

# RLAIF (Reinforcement Learning from AI Feedback)

## 定义
用 AI 模型（通常是强大的 LLM）代替人类标注者生成偏好标签，再用这些标签训练奖励模型并进行 RL 对齐的技术路线，是 RLHF 的低成本替代方案。

## 数学形式

奖励模型训练（Bradley-Terry 模型）：
$$P(y_w \succ y_l) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

AI feedback 生成：$r(x, y) \approx f_{\text{LLM}}(x, y)$，用强 LLM 直接打分替代人类偏好。

## 核心要点
1. 用 GPT-4/Claude 等强模型标注偏好，替代昂贵的人工标注
2. 可大规模扩展标注量，适合数据密集的对齐场景
3. 质量上限受限于 AI 标注者的判断能力，可能放大 AI 偏见
4. MiniCPM-V 4.5 使用 RLAIF 进行多模态对齐

## 代表工作
- [[MiniCPM-V]]: 使用 RLAIF 训练 MLLM

## 相关概念
- [[GRPO]]
- [[DAPO]]
