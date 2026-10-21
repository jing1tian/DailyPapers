---
type: concept
aliases: [动作展开蒸馏, 动作 Rollout 蒸馏]
---

# Action Rollout Distillation

## 定义

Action Rollout Distillation 是一种针对基于扩散/流匹配的机器人策略的知识蒸馏方法：让单步学生模型在相同的观测、条件和初始噪声下，直接回归多步教师模型经完整 Rollout 后输出的动作块，从而在保持动作质量的同时将推理压缩到单步。

## 数学形式

$$
\mathcal{L}_{rollout} = \mathbb{E}_{o, c, \epsilon}\bigl[\bigl\|a_{student}(o, c, \epsilon) - a_{teacher}^{N}(o, c, \epsilon)\bigr\|^2\bigr]
$$

其中 $o$ 为观测，$c$ 为条件（如轨迹、语言），$\epsilon$ 为初始噪声（教师与学生共享），$N$ 为教师的去噪步数（如 10+10 步）。

## 核心要点

1. **共享初始噪声**: 学生与教师使用相同的初始噪声 $\epsilon$，确保回归目标确定性，减少训练方差
2. **End-to-End 动作匹配**: 直接监督最终动作块输出（而非中间噪声预测），对下游任务更直接
3. **与 Consistency Distillation 的区别**: CD 在潜码空间匹配中间去噪路径，Action Rollout 在动作空间匹配最终输出，对多步积累误差更鲁棒
4. **解耦设计**: 与视频蒸馏独立进行，避免视频和动作梯度之间的干扰

## 代表工作

- [[TrajWAM]]: 10+10 步教师（10 步视频 + 10 步动作）引导单步学生动作分支，训练 2,000 步，显著优于一致性蒸馏（+23.83 pp SR）

## 相关概念

- [[Consistency Distillation]]: 另一种单步蒸馏策略，在潜码空间匹配去噪轨迹
- [[Distribution Matching Distillation]]: 视频骨干部分使用的蒸馏方法
- [[Action Chunking]]: 被蒸馏的动作表示形式
- [[World Action Model]]: Action Rollout Distillation 的应用场景
