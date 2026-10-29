---
type: concept
aliases: [Trajectory Slicing with Episode Reset, 轨迹切片与Episode重置]
---

# TSER (Trajectory Slicing with Episode Reset)

## 定义

一种轨迹预处理技术，丢弃失败轨迹的失败前缀，将纠正动作后缀作为新 episode 从 t=0 重新开始，使纠正动作与当前不利状态直接绑定，消除原始 DAgger 式聚合中的因果混淆。

## 数学形式

设原始轨迹 $\tau = (\tau_{fail}, \tau_{rec})$，TSER 将其变换为：

$$
\tau' = \tau_{rec}\big|_{t=0}, \quad \text{即重置纠正片段为独立 episode}
$$

## 核心要点

1. 关键洞察：将失败前缀与纠正后缀共享 episode 会导致策略误将失败状态与恢复动作关联。
2. 通过 episode 重置，纠正动作在语义上绑定到当前不利状态（adverse state），而非失败历史。
3. 与原始 DAgger 的区别：DAgger 直接聚合完整轨迹；TSER 切片后丢弃失败段。
4. 实验表明 DAgger-style 直接聚合会导致 standard 成功率降至 0，而 TSER 大幅改善。

## 代表工作

- [[RePO-VLA]]: TSER 作为 RAI 阶段的核心预处理步骤

## 相关概念

- [[Recovery-Aware Initialization]]
- [[DAgger]]
- [[Behavior Cloning]]
