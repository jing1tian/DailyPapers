---
type: concept
aliases: [PUR, Policy-Utility Residual, 策略效用残差]
---

# Policy-Utility Residual（PUR）

## 定义

PUR 是衡量世界模型训练目标（像素 MSE 梯度）与策略效用目标（元梯度）对齐程度的诊断指标，由 [[DRILL]] 提出，越低表示世界模型的更新方向与策略改进方向越一致。

## 数学形式

$$
\mathrm{PUR} = 1 - \cos\!\left(\nabla_\phi \mathcal{L}_{\mathrm{MSE}}(\phi),\;-\nabla_\phi J(\pi_{\theta^+(\phi)})\right)
$$

PUR ∈ [0, 2]，其中 0 表示完全对齐，2 表示完全反向。

**实用估计方式**（随机子空间有限差分，$r=128$ 方向）：

$$
\hat{e}_i = \frac{J\bigl(\pi_{\theta^+(\phi+\delta u_i)}\bigr)-J\bigl(\pi_{\theta^+(\phi-\delta u_i)}\bigr)}{2\delta}
$$

## 核心要点

1. **直觉**: PUR 量化了"最小化像素 MSE"与"最大化策略回报"这两个梯度方向的偏差
2. **实验相关性**: LIBERO-Long 上，PUR 与策略成功率的 Spearman 相关系数 ρ = −0.89（95% CI [−0.97, −0.71]），显示强负相关
3. **实验数值**: WMPO: PUR=1.42，RLVR-World: 0.97，DRILL-VWI: 0.73，DRILL-IMG: 0.36
4. **用途**: 可作为世界模型训练方法的通用诊断工具，无需运行完整策略评估

## 代表工作

- [[DRILL]]: 提出 PUR，用于证明目标错配问题并监控训练动态

## 相关概念

- [[双层优化]]
- [[元梯度]]
- [[WMPO]]
- [[DRILL]]
