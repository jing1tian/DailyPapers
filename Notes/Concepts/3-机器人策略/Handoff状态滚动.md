---
type: concept
aliases: [Handoff State Rollout, Rolled Handoff Context, 接管状态前滚]
---

# Handoff 状态滚动

## 定义

在异步动作块执行中，利用冻结的动作-历史模型将已知的 outgoing 动作命令确定性地"前滚"（rollout）$k$ 步，得到预期接管时刻的隐状态，作为下一个动作块生成的执行上下文条件。

## 数学形式

确定性前滚（$k$ 步）：

$$
h_{j+1} = \mathcal{R}_{act}(h_j,\; u_j), \quad j = r, \ldots, r+k-1
$$

双路 Handoff 编码：

$$
\bar{h}_{r,k} = \text{sg}(h_{r+k}), \quad \Delta h_{r,k} = \bar{h}_{r,k} - \text{sg}(h_r)
$$

$$
g_{r,k} = P_\psi\bigl([\text{LN}(\bar{h}_{r,k});\; \text{LN}(\Delta h_{r,k})]\bigr)
$$

## 核心要点

1. **确定性而非预测性**：利用推理期间已确定执行的 outgoing 命令进行前滚，无需预测未来场景，无随机性。
2. **绝对 + 差值双路**：绝对状态 $\bar{h}_{r,k}$ 描述接管时刻的运动上下文；差值 $\Delta h_{r,k}$ 揭示推理期间的执行推进量；两者解决 handoff 状态的歧义性。
3. **冻结参数**：$\mathcal{R}_{act}$ 和 stop-gradient 保证前滚不干扰主策略梯度。
4. **延迟泛化**：前滚条件在固定延迟 $k=3,4,5$ 和随机延迟下均表现稳定（LIBERO SR 变化 <0.3%）。

## 代表工作

- [[BICPO-VLA]]：将 Handoff 状态前滚引入 VLA 动作块生成，作为 Haar 结构化实现和 BICPO 偏好优化的执行上下文条件

## 相关概念

- [[行为条件化动作纤维]]
- [[Asynchronous Inference]]
- [[Action Chunking]]
- [[Haar动作分解]]
