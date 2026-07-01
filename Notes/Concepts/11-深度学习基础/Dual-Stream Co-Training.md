---
type: concept
aliases: [Dual-Stream Co-Training, 双流联合训练, Co-Training]
---

# Dual-Stream Co-Training

## 定义

Dual-Stream Co-Training 是一种同时在两条数据流（操控数据流 + 视觉语言数据流）上联合优化 VLA 模型的训练策略，旨在防止大规模操控数据训练侵蚀 VLM 主干的通用视觉语言理解能力。

## 核心要点

1. **问题动机**: 单纯在操控数据上微调 VLM-based 策略会导致视觉语言主干的感知和推理能力退化（"遗忘"通用能力）
2. **两条数据流**:
   - **操控流**: 机器人演示数据，优化 Flow Matching 动作预测损失 $\mathcal{L}_{\text{FM}}$
   - **VLM 流**: 精心筛选的视觉语言数据，优化语言建模损失 $\mathcal{L}_{\text{VLM}}$
3. **混合比例**: 操控数据与 VLM 数据按 **9:1** 比例混合（Qwen-RobotManip 实践）
4. **梯度协同**: 两条流共享 VLM 主干权重，同时更新

## 数学形式

$$
\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda \mathcal{L}_{\text{VLM}}
$$

- $\mathcal{L}_{\text{FM}}$: Flow Matching 动作损失
- $\mathcal{L}_{\text{VLM}}$: 视觉语言建模损失（语言生成等）
- $\lambda$: 权重系数（由数据比例隐式控制）

## 代表工作

- [[Qwen-RobotManip]] (2026): 以 9:1 的 VLA/VLM 数据比例实施 Dual-Stream Co-Training，维持了 Qwen3.5-4B 主干的通用感知能力，同时实现大规模操控训练

## 相关概念

- [[Flow Matching]]: 操控流的动作预测训练目标
- [[视觉-语言-动作模型]]: 应用 Dual-Stream Co-Training 的主要模型类型
- [[行为克隆]]: 操控流中的核心学习范式
