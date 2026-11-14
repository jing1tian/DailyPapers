---
type: concept
aliases: [选择性未来想象, 自适应未来想象, selective imagination]
---

# Selective Future Imagination（选择性未来想象）

## 定义

选择性未来想象是世界动作模型（WAM）的一种推理策略：在每个重规划步骤，根据当前状态的不确定性或复杂度，动态决定是否生成未来视觉状态（future imagination），而非每步都生成或完全跳过。

## 数学形式

**路由规则**（以 [[Bind-WAM]] 为例）：

$$
\pi(a_{t:t+H} \mid o_t, e_t, \ell) = \begin{cases} \pi_{\mathrm{joint}}, & U_t \geq \tau_U \;\lor\; \Delta U_t \geq \tau_\Delta \\ \pi_{\mathrm{direct}}, & \text{otherwise} \end{cases}
$$

其中 $\pi_{\mathrm{joint}}$ 触发联合想象路径，$\pi_{\mathrm{direct}}$ 为直接动作生成路径。

## 核心要点

1. **动机**: 全量想象代价高（约 1.6× 时间开销），但对简单稳定帧无益；完全跳过想象则在关键时刻性能下降；
2. **路由信号**: 可以是绑定不确定性（Bind-WAM）、动态复杂度（AdaWAM）等可在线估计的状态量；
3. **帕累托优势**: 选择性想象在约 40% 想象比例时即可达到全量想象的成功率，完成时间接近无想象策略；
4. **实现方式**: 可通过共享权重 + 注意力掩码切换（Bind-WAM）或独立两套网络实现。

## 代表工作

- [[Bind-WAM]]: 以绑定不确定性为路由信号，LIBERO 40.6% 想象比例达到 98.6% 成功率
- [[AdaWAM]]: 自适应多模态推理，另一种选择性想象方案

## 相关概念

- [[Binding Uncertainty]]
- [[WAM|世界动作模型（WAM）]]
- [[Fast-WAM]]
- [[Flow Matching]]
