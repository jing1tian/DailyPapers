---
type: concept
aliases: [力感知动作模型, FEAM]
---

# Force-Envisioned Action Model

## 定义
FAWAM 的核心模块之一：将历史力/力矩信号编码后通过 AdaLN 调制层注入视频生成 Transformer，同时联合预测未来动作块与力轨迹，实现感知层与预测层的力信号融合。

## 数学形式

$$
\bar{\mathbf{u}}_t^{(i)} = \mathbf{u}_t^{(i)} + \text{MLP}_f^{(i)}(\mathbf{c}_t)
$$

其中 $\mathbf{c}_t = \text{Enc}_f(\mathbf{w}_{t-H_f+1:t})$ 为接触特征，调制每个 block 的 AdaLN 参数。

## 核心要点

1. **感知层**：历史力信号 → MLP 编码 → 接触特征 $\mathbf{c}_t$
2. **调制层**：$\mathbf{c}_t$ 通过残差方式调制每个 Transformer block 的 AdaLN 参数，零初始化保证训练稳定
3. **预测层**：解码器同时输出动作速度场和力速度场，联合建模接触演化
4. **两阶段训练**：Stage 1 训练视频生成（冻结），Stage 2 训练力感知解码器

## 代表工作

- [[FAWAM]]: Force-Envisioned Action Model 是 FAWAM 基础策略的核心

## 相关概念

- [[Force Feature Encoding]]
- [[AdaLN]]
- [[Flow Matching]]
- [[World Action Model]]
- [[Pi05]]
