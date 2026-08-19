---
type: concept
aliases: [推理时自适应, 推理时优化, inference-time optimization, test-time adaptation]
---

# Inference-Time Adaptation（推理时自适应）

## 定义

**推理时自适应**指在模型权重冻结的前提下，通过调整外部控制参数（如提示、采样策略、路由规则）在推理阶段提升模型性能的方法范式。与微调（fine-tuning）不同，不修改任何内部参数。

## 数学形式

$$
a^{\star} = \pi_{\Omega}(x) = F(\Omega, x, \mathcal{A}), \quad \theta \text{ frozen}
$$

其中 $\Omega$ 为可调整的外部控制状态，$\theta$ 为冻结的模型权重，$\mathcal{A}$ 为候选集。

## 核心要点

1. **零梯度**: 不对模型参数计算或应用梯度，计算成本低
2. **可逆性**: 控制状态变更可回滚，不影响底层模型
3. **评估风险**: 若测试集分数影响控制状态更新，则违反评估完整性（见 [[Score Isolation]]）
4. **控制轴多样性**: 可优化文本、采样、验证规则、奖励选择等多种外部控制轴

## 代表工作

- [[SCOPE]]: 提出类型化控制状态 + 分数隔离的系统化推理时自适应框架，用于视频世界模型
- Best-of-N 采样: 最简单的推理时自适应，从多个候选中选择最优

## 相关概念

- [[Score Isolation]]
- [[Video World Model]]
- [[Learned Reward Selector]]
