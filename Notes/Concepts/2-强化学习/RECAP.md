---
type: concept
aliases: [RECAP, Reward-Conditioned Advantage Policy, 优势条件策略]
---

# RECAP

## 定义

RECAP 是一种基于分布式价值学习的优势条件后训练方法：先从 [[Rollback]] 重建轨迹并训练分布式价值模型，再以 N 步优势为条件标注时间步，最后通过 [[Classifier-Free Guidance (CFG)]] 在动作空间引导推理。

## 数学形式

**归一化价值目标**:
$$y_t = \text{clip}\!\left(\frac{R_t}{2T_{max}}, -1, 0\right)$$

**分布式价值模型（B=201 个原子）**:
$$p_\phi(V \mid o_t, \ell) \in \Delta^B, \quad V_{min}=-1, V_{max}=0$$

**N 步优势**:
$$A_t^{(N)} = \sum_{k=0}^{N-1}\tilde{r}_{t+k} + V(o_{t+N},\ell) - V(o_t,\ell)$$

**推理时 CFG**:
$$a_{CFG} = a_\emptyset + \beta(a_+ - a_\emptyset),\quad \beta = 1.75$$

## 核心要点

1. 利用 [[Rollback]] 重建的失败轨迹来获得负样本，标注价值目标。
2. 分布式价值模型（Categorical, B=201）比标量价值更稳定，输出置于 $[-1, 0]$。
3. 对优势超过 70 百分位的时间步附加"Advantage: positive"条件词，低优势步骤无额外条件。
4. 推理时以无条件分支为基准向正优势方向外推（$\beta=1.75$）。

## 代表工作

- [[MR0]]: RECAP 在以太网插线上达到 60% SR / 0.67 PS，低于 CFG-DAgger

## 相关概念

- [[Rollback]]
- [[CFG-DAgger]]
- [[Classifier-Free Guidance (CFG)]]
- [[Advantage Estimation]]
- [[DAgger]]
