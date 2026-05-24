---
type: concept
aliases: [Teacher Forcing, teacher forcing, 教师强制]
---

# Teacher Forcing（教师强制）

## 定义
序列模型训练中的一种策略：在每个时间步，用真实目标值（ground truth）而非模型上一步的预测值作为下一步的输入，使训练更稳定但可能导致推理时的分布偏移。

## 核心要点
1. **训练阶段**：输入为真实序列 $y_{t-1}$，而非自回归生成的 $\hat{y}_{t-1}$。
2. **优点**：梯度信号清晰，收敛快，训练稳定。
3. **缺点**：训练与推理分布不一致（exposure bias），模型在推理时依赖自身错误的预测。
4. **缓解方法**：Scheduled Sampling（逐渐用预测值替换真实值）、自回归 rollout 混合训练。

## 代表工作
- Williams & Zipser（1989）：Teacher Forcing 概念提出
- [[OrbiSim]]: 采用两阶段训练策略——先 teacher forcing 再自回归 rollout，以余弦退火调整比例

## 相关概念
- [[余弦退火]]
- [[状态空间模型]]
- [[World Model]]
