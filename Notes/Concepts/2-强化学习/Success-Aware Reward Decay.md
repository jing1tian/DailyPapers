---
type: concept
aliases: [成功感知奖励衰减, Completion-Aware Reward Calibration]
---

# Success-Aware Reward Decay

## 定义

Success-Aware Reward Decay 是一种针对稀疏成功奖励的奖励校准方法：对成功轨迹按完成时间步数施加指数衰减 $\tilde{r}_i = r \cdot \gamma^{d_i}$，使更早完成任务的轨迹获得更高奖励，同时在全成功组中使用非负最小值基线排序代替标准均值归一化，避免较慢成功轨迹获得负优势。

## 数学形式

**奖励校准**（仅对成功轨迹）：

$$
\tilde{r}_i = r \cdot \gamma^{d_i}
$$

**全成功组优势估计**（替代标准均值归一化）：

$$
A_i = R_i - \min_j R_j
$$

其中 $R_i = \tilde{r}_i$ 为校准后回报。

## 核心要点

1. **动机**: 稀疏成功奖励将所有成功轨迹等同对待（无论完成时间），且标准均值归一化会给慢成功轨迹分配负优势，过度惩罚效率稍低但同样成功的行为
2. **仅对成功施加**: 失败轨迹保持原始奖励不变；含成功和失败的混合组仍用标准归一化，成败信号占主导
3. **非负全成功组**: 在全成功组中以最慢者为基准（$A=0$），更早完成者获正优势，保证不将任何成功标记为负样本
4. **对比 Reward Penalty**: Reward Penalty 在全成功组中仍用均值归一化（给慢成功负优势），导致减步过激、成功率增长慢；Reward Decay 成功率增长更快且同样有减步效果

## 超参数

- $\gamma \in (0, 1]$：衰减因子，Z-1 使用 $\gamma = 0.998$
- $\gamma = 1$：退化为原始稀疏奖励（无衰减）

## 代表工作

- [[Z-1]]: 在 [[RoboCasa]] PnPStoveToCounter 等任务上验证此方法有效性

## 相关概念

- [[GRPO]]
- [[Shared-Prefix GRPO]]
- [[Advantage Estimation]]
