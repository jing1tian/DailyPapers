---
type: concept
aliases: [绑定不确定性, 语言目标绑定不确定性]
---

# Binding Uncertainty（绑定不确定性）

## 定义

绑定不确定性是对当前观测帧中语言-目标绑定可靠程度的量化估计：当策略无法稳定地将语言指令中的指称词对应到唯一的视觉实例时，不确定性高；绑定明确时不确定性低。

## 数学形式

**帧级不确定性标签**（离线标注）：

$$
y_t = \max_r \; y_t^r, \quad y_t^r = \begin{cases} 1, & \mathrm{IoU}_r < 0.3 \;\lor\; v_t^r = 0 \;\lor\; s_t^r < 0.9 \\ 0, & \text{otherwise} \end{cases}
$$

**预测器输出**（推理时实时估计）：

$$
U_t = f_\theta(o_t, \ell), \quad U_t \in [0, 1]
$$

**预测器损失**（类别加权 BCE + Brier 校准项）：

$$
\mathcal{L}_{\mathrm{bind}} = \mathrm{BCE}_w(\hat{U}_t, y_t) + \lambda (\hat{U}_t - y_t)^2
$$

## 核心要点

1. **三类失效场景**: 参考 IoU 过低（指称错误）、目标不可见（遮挡）、DINO 特征漂移（外观变化）；
2. **实时可预测**: 仅需当前帧 $o_t$ 和语言指令 $\ell$，轻量 ViT 预测器（~6.65M）可在线估计；
3. **路由信号**: 绑定不确定性的绝对值 $U_t$ 和变化率 $\Delta U_t = |U_t - U_{t-1}|$ 双重条件触发 WAM 的选择性未来想象。

## 代表工作

- [[Bind-WAM]]: 提出绑定不确定性概念并将其用于 WAM 路由，LIBERO AUROC 0.9653，RoboTwin AUROC 0.9984

## 相关概念

- [[Language-Object Binding]]
- [[Selective Future Imagination]]
- [[DINOv3]]
- [[ViT]]
- [[WAM|世界动作模型（WAM）]]
