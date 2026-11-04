---
type: concept
aliases: [反事实分支训练, Counterfactual Branches, Branch Training]
---

# Counterfactual Branch Training

## 定义

从已有演示轨迹的中间状态出发，对专家动作施加随机扰动并在真实模拟器中执行，获得反事实状态转移，作为世界模型的预测监督目标，以扩大动力学模型对 off-manifold 动作的泛化能力。

## 核心要点

1. **动机**: 世界模型仅在专家演示动作上训练，但规划器必须评估未见过的候选动作；反事实分支提供 off-manifold 的真实监督信号
2. **实现**: 从每第 $k$ 个录制状态出发，对演示动作加 Gaussian 噪声 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 并截断至合法动作空间，在模拟器中执行获得真实下一状态
3. **分支仅作预测目标**: 分支数据不包含专家动作标签，不影响行为克隆损失；仅用于动力学和状态预测的监督
4. **效果**: 显著改善随机动作下的预测精度，提升候选轨迹的排名质量（rank 相关从 0.49→0.80）

## 数学形式

设第 $i$ 个演示在时刻 $t$ 的状态为 $s_t^{(i)}$，演示动作为 $a_t^{(i)}$：

$$
\tilde{a}_t = \text{clip}\!\left(a_t^{(i)} + \epsilon,\; [-1,1]^d\right), \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

$$
\tilde{s}_{t+1} = \text{Sim}(s_t^{(i)}, \tilde{a}_t)
$$

分支提供监督对 $(\tilde{a}_t, \tilde{s}_{t+1})$，仅用于动力学损失 $\mathcal{L}_s$。

## 代表工作

- [[pico-WAM]]: 在 Push-T 任务上，SD=0.3的 Gaussian 扰动生成6013个分支，共48,104个转移；使 restart 搜索覆盖率优势扩大4.8pp

## 相关概念

- [[世界动作模型（WAM）|World Action Model]]
- [[Behavior Cloning]]
- [[Selection Regret]]
- [[MBPO]]
