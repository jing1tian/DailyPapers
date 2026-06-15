---
type: concept
aliases: [Self-Recovery, MLLM Self-Recovery, 自修复]
---

# Self-Recovery

## 定义
让 MLLM 在理解任务前先用自身的生成能力修复损坏的输入图像，从而提升在真实噪声/模糊/压缩场景下的感知鲁棒性。

## 核心要点
1. Pipeline：corruption 输入 → MLLM 生成修复图像 → MLLM 用修复图做理解任务（VQA、描述等）
2. 用 SFT + RL（CoT 推理链）训练模型，使其学会何时先修复、何时直接作答
3. 核心挑战：自修复步骤带来额外 latency；修复质量直接决定下游理解的上限
4. 来自 Robust-U1 论文

## 数学形式
$$
\hat{x} = \text{MLLM}_{\text{gen}}(x_{\text{corrupted}}), \quad y = \text{MLLM}_{\text{und}}(\hat{x}, q)
$$

## 相关概念
- [[BAGEL]]
- [[14-安全与鲁棒性]]
