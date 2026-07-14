---
type: concept
aliases: [动作逆映射, action inversion, 动作逆化]
---

# Action Inversion

## 定义

Action Inversion 是将生成策略的输出动作 $a^*$ 反向映射回其对应的噪声潜变量 $w^*$ 的过程，使得同一策略以 $w^*$ 为输入时会重现 $a^*$。

## 数学形式

给定[[流匹配]]策略的 $K$ 步 Euler 离散化，per-step 不动点迭代求解：

$$
x_k^{(m+1)} = x_{k+1} - \Delta t \cdot v_\theta\!\left(x_k^{(m)},\, t_k,\, s\right), \quad x_k^{(0)} = x_{k+1}
$$

从 $x_K = a^*$ 逆向迭代，最终提取 $w^* = x_0$。

## 核心要点

1. **Per-step 不动点迭代**: 逐步反向求解每个 ODE 离散步，比轨迹级逆方法精度高约 20×（Action MSE 0.00168 vs 0.0329）。
2. **收敛保证**: 当 $\Delta t \cdot L < 1$（$L$ 为速度场 Lipschitz 常数）时，迭代为压缩映射，几何收敛。
3. **效率**: 默认 M=5 次迭代，仅需 456ms，不影响实时部署。
4. **适用范围**: 同时适用于 action-head VLA 和生成状态+动作的 world-action model。

## 代表工作

- [[FlowDAgger]]: 首次将 Action Inversion 用于人机交互在线适应

## 相关概念

- [[流匹配]]
- [[扩散策略]]
- [[Noise Policy]]
- [[DAgger]]
