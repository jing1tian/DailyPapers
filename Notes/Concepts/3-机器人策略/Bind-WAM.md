---
type: concept
aliases: [Binding-Uncertainty-Guided Selective WAM, 绑定不确定性引导的选择性世界动作模型]
---

# Bind-WAM

## 定义

Bind-WAM 是一种以 [[Language-Object Binding|语言-目标绑定不确定性]] 为路由信号的世界动作模型，通过轻量预测器实时估计绑定不确定度，仅在不确定度高时触发未来视觉想象，在保持成功率的同时减少约 60% 的想象调用次数。

## 数学形式

**路由规则**：

$$
\pi = \begin{cases} \pi_{\mathrm{joint}}, & U_t \geq \tau_U \;\lor\; |U_t - U_{t-1}| \geq \tau_\Delta \\ \pi_{\mathrm{direct}}, & \text{otherwise} \end{cases}
$$

**不确定性预测器**：

$$
U_t = f_\theta(o_t, \ell), \quad U_t \in [0, 1]
$$

## 核心要点

1. **离线标注流水线**: Qwen3-VL 32B 定位，SAM 3.1 追踪，DINOv3 特征相似度，三指标联合标注帧级绑定不确定性；
2. **轻量预测器**: ~6.65M ViT 风格网络，仅用当前帧 + 语言指令，LIBERO AUROC 0.9653，RoboTwin AUROC 0.9984；
3. **共享权重双路径**: 一套 Wan2.2-5B WAM 参数，注意力掩码切换 direct/joint 路径，无额外模型开销；
4. **性能**: LIBERO 98.6%（vs Full-Joint 98.4%，想象比例 40.6%，11.7 s）；真实双臂 66.6%（vs Full-Joint 57.6%，想象比例 28.4%）。

## 代表工作

- [[Bind-WAM（论文笔记）|Bind-WAM]]: 原始论文，arXiv:2611.15368，Fudan University / ROBOTOPIA，2026

## 相关概念

- [[Language-Object Binding]]
- [[Binding Uncertainty]]
- [[Selective Future Imagination]]
- [[Fast-WAM]]
- [[WAM|世界动作模型（WAM）]]
- [[SAM3]]
- [[DINOv3]]
- [[Qwen3-VL]]
- [[Wan 2.2-5B]]
