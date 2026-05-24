---
type: concept
aliases: [Supervised Fine-Tuning, 监督微调, 有监督微调]
---

# SFT（监督微调）

## 定义

在预训练模型基础上，使用标注数据通过监督学习目标（如交叉熵或 MSE）更新模型权重，使其适应特定任务。

## 核心要点

1. 标准迁移学习手段，需要有标注的成功示例数据
2. 可能导致灾难性遗忘（Catastrophic Forgetting）原有能力
3. 相比 [[Activation Steering]] 更持久但成本更高
4. COAST 实验中 SFT 在部分设置下反而降低了性能（RoboCasa π0.5: 0.40→0.31）

## 代表工作

- [[COAST]]: 将 SFT 作为对比基线，证明推理时激活引导在数据效率上的优势

## 相关概念

- [[Activation Steering]]
- [[RLHF]]
- [[LoRA]]
