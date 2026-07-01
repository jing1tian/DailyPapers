---
type: concept
aliases: [自强迫训练, 自强制训练, Self-forcing Training]
---

# Self-forcing

## 定义

在自回归序列生成的训练中，随机将历史条件帧替换为模型自身的预测帧（而非真实帧），使模型在训练阶段即暴露于自身预测误差，提升推理阶段的自回归鲁棒性。

## 数学形式

训练时以概率 $p_{\text{sf}}$ 将历史帧 $o_{t-k}^{\text{gt}}$ 替换为自身预测 $\hat{o}_{t-k}$：

$$
h_t = \begin{cases} o_{t}^{\text{gt}} & \text{以概率 } 1 - p_{\text{sf}} \\ \hat{o}_t & \text{以概率 } p_{\text{sf}} \end{cases}
$$

## 核心要点

1. **训练-推理鸿沟**: 标准教师强制（Teacher Forcing）训练时用真实历史帧，推理时用自身生成帧，导致误差累积
2. **Self-forcing 解决方案**: 训练时引入自身预测帧，模型主动适应自身错误的历史
3. **与 [[Autoregressive Generation]] 结合**: 通常应用于长程自回归世界模型的精调阶段
4. **对比 Scheduled Sampling**: Self-forcing 更激进，直接使用完整的自回归预测帧

## 代表工作

- [[A2World]]: A2World-sim 在精调阶段采用 Self-forcing 提升长程仿真稳定性

## 相关概念

- [[Autoregressive Generation]]
- [[Pose-guided History Sampling]]
- [[Action-Conditioned World Model]]
