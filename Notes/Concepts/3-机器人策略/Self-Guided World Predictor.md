---
type: concept
aliases: [SGWP, 自引导世界预测器, Self-Guided World Prediction]
---

# Self-Guided World Predictor (SGWP)

## 定义

一种以策略自身的 EMA 动量拷贝为目标生成器、在策略表示空间内预测动作条件潜在动力学的模块，消除了辅助潜变量方法中的目标-策略不匹配问题。

## 数学形式

$$
\hat{\bm{z}}_{t+\Delta}^{Q} = \mathcal{F}_{\psi}\left(\bm{z}_{t}^{Q}, \bm{e}_{t}^{A}\right), \quad \hat{\bm{z}}_{t+\Delta}^{Q} \approx \operatorname{sg}\left(\bar{\bm{z}}_{t+\Delta}^{Q}\right)
$$

$$
\mathcal{L}_{\mathrm{pred}} = \frac{1}{N_q d} \left\| \hat{\bm{z}}_{t+\Delta}^{Q} - \operatorname{sg}\left(\bar{\bm{z}}_{t+\Delta}^{Q}\right) \right\|_F^2
$$

## 核心要点

1. **自引导**: 预测目标由策略骨干的 EMA 拷贝生成，保证目标始终在策略表示流形内
2. **动作条件**: 以干预动作块嵌入 $\bm{e}_t^A$ 为键值，通过交叉注意力学习动作-状态过渡
3. **Stop-Gradient**: 阻断目标梯度传播，防止表示崩塌
4. **推理时去除**: SGWP 仅参与训练，不增加推理延迟

## 与 VLA-JEPA 的区别

| 维度 | VLA-JEPA | SGWP |
|------|----------|------|
| 目标来源 | 独立 JEPA 编码器 | 策略自身 EMA 拷贝 |
| 目标对齐 | 目标与策略空间可能不匹配 | 天然对齐 |
| 动作条件 | 有 | 有 |

## 代表工作

- [[SG-WAM]]: 提出 SGWP，仿真 LIBERO 均值贡献 +1.9pp，长时序任务贡献最大

## 相关概念

- [[Learnable Dynamics Tokens]]
- [[EMA]]
- [[Stop-Gradient]]
- [[Cross-Attention]]
- [[World Action Model]]
