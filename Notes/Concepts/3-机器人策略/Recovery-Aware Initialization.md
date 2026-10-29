---
type: concept
aliases: [RAI, 恢复感知初始化]
---

# Recovery-Aware Initialization (RAI)

## 定义

一种 VLA 微调策略，通过 TSER（轨迹切分+Episode重置）消除失败前缀对恢复动作的因果混淆，将纠正动作序列独立为新 episode 后与专家演示联合训练。

## 数学形式

$$
\mathcal{L}_{SFT} = -\mathbb{E}_{\tau \sim \mathcal{D}_{expert}} \sum_t \log \pi(a_t|o_t) - \lambda \, \mathbb{E}_{\tau' \sim \mathcal{D}_{rec}^{reset}} \sum_t \log \pi(a_t|o_t)
$$

## 核心要点

1. 解决了直接 DAgger 式聚合中的因果混淆问题（失败前缀与恢复动作共享 episode 导致策略误学）。
2. 使用 [[TSER]] 将纠正序列从 t=0 重新开始，绑定到当前不利状态。
3. 两类数据联合训练：专家成功演示 $\mathcal{D}_{expert}$ 和重置恢复数据 $\mathcal{D}_{rec}^{reset}$，权重 $\lambda$ 平衡。

## 代表工作

- [[RePO-VLA]]: 提出 RAI 作为两阶段训练的第一阶段

## 相关概念

- [[TSER]]
- [[Behavior Cloning]]
- [[DAgger]]
- [[Value-Conditioned Refinement]]
