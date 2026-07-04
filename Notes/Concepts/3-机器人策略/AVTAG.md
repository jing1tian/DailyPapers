---
type: concept
aliases: [Action-Visual-Tactile Attention Guidance, Contact-Gated Tactile Attention]
---

# AVTAG

## 定义
在 VLA/WAM 的 flow matching 推理中，通过门控机制在接触阶段强制动作查询（action query）优先关注触觉 token 而非视觉 token 的注意力引导策略。

## 数学形式（VT-WAM 实现）

用 stop-gradient 构造辅助注意力分布（避免梯度回传至 K/V）：

$$
\mathbf{P}_{\mathrm{vt}} = \text{Softmax}\!\left(\frac{\mathbf{Q}_a \cdot \mathrm{sg}(\mathbf{K}_{\mathrm{vt}})^\top}{\sqrt{d}}\right)
$$

归一化视觉/触觉注意力占比：

$$
p_v(r) = \frac{\alpha_v(r)}{\alpha_v(r) + \alpha_t(r)}, \quad p_t(r) = \frac{\alpha_t(r)}{\alpha_v(r) + \alpha_t(r)}
$$

Hinge ranking 损失（仅在接触阶段 $\mathcal{C}$ 生效）：

$$
\mathcal{L}_{\mathrm{AVTAG}} = \mathbb{E}_{r \in \mathcal{C}}\!\left[\max\!\left(0,\, p_v(r) - p_t(r)\right)\right]
$$

触觉注意力一旦超过视觉注意力，损失归零，不施加反向约束。

## 核心要点
1. 触觉信号在接触发生时才有意义，非接触阶段对动作决策贡献极小
2. 简单地把触觉 token 拼入输入序列会导致视觉与触觉权重竞争，AVTAG 用 hinge ranking 损失在训练阶段纠正这种偏置
3. 与 [[Asymmetric MoT|非对称 MoT 注意力]] 配合使用：MoT 负责视觉-触觉时序融合，AVTAG 负责动作查询阶段的接触聚焦
4. 接触阶段集合 $\mathcal{C}$ 基于触觉传感器形变幅度阈值划定
5. **仅在训练阶段生效**，推理时无额外计算开销

## 代表工作
- [[VT-WAM]]: 首次提出 Contact-Gated AVTAG，在六个接触密集任务上验证，整体成功率 71.67%（vs Fast-WAM 45%），消融实验中贡献约 15 pp 提升

## 相关概念
- [[World Action Model]]
- [[Flow Matching]]
- [[CBF]]
- [[MoT]]
