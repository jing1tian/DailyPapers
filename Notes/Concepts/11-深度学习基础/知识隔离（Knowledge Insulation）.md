---
type: concept
aliases: [Knowledge Insulation, KI, 知识隔离, Stop-Gradient隔离]
---

# 知识隔离（Knowledge Insulation）

## 定义

在多阶段或多分支训练框架中，对中间预测结果施加 stop-gradient（$\operatorname{sg}[\cdot]$），阻止下游分支的梯度反向传播到上游预训练模块，从而保护上游模块已学到的表示不被下游任务损坏。

## 数学形式

$$
\text{下游输入} = \operatorname{sg}[\hat{u}_T]
$$

其中 $\hat{u}_T$ 为上游模块（如预训练世界模型）的预测输出，stop-gradient 阻止下游损失 $\mathcal{L}_{\text{act}}$ 通过 $\hat{u}_T$ 更新上游参数。

## 核心要点

1. **机制**: 前向传播正常通过，反向传播时梯度在此截断
2. **动机**: 下游任务（如动作生成）的优化目标与上游模块（如世界模型）的表示目标可能存在冲突，KI 防止下游任务"破坏"上游已学到的动作相关结构
3. **与冻结参数的区别**: 冻结参数完全固定上游模块，KI 允许上游模块通过其他损失继续更新（如 Stage 2 中的 $\mathcal{L}_{\text{distill}}$ 和 $\mathcal{L}_{\text{wm}}^{(2)}$）
4. **代价极低**: 实现上仅需 `detach()` 操作，无额外计算开销

## 代表工作

- [[LaWAM]]: 首次在世界动作模型中提出 Knowledge Insulation，对预测潜在子目标 $\hat{u}_T$ 应用 stop-gradient，防止动作梯度流回 LaWM，消融实验显示 KI 贡献 2.8% 成功率提升

## 相关概念

- [[知识蒸馏]]
- [[ALAM|潜在动作模型]]
- [[世界动作模型（WAM）]]
