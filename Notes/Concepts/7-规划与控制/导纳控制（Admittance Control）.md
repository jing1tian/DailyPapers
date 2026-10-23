---
type: concept
aliases: [Admittance Control, 导纳控制, admittance controller]
---

# 导纳控制（Admittance Control）

## 定义

导纳控制是一种力-位置混合控制策略：将力/力矩误差通过虚拟质量-阻尼-弹簧模型转化为末端位姿修正量，从而实现柔顺的力-位协调控制。

## 数学形式

$$
M\Delta\ddot{x} + C\Delta\dot{x} + K_a\Delta x = -S \, e_w
$$

其中 $e_w = F^{\text{ref}} - F^{\text{meas}}$ 为力参考误差，$\Delta x$ 为末端修正量。

含积分项（消除稳态误差）：

$$
\begin{aligned}
M\Delta\ddot{x} + C\Delta\dot{x} + K_a\Delta x &= -S(e_w + k_I \eta) \\
\dot{\eta} &= e_w - \lambda\eta
\end{aligned}
$$

## 核心要点

1. **与阻抗控制的区别**: 导纳控制以力为输入、位置为输出（力→位移）；阻抗控制以位移为输入、力为输出。
2. **参数调节**: $M$（惯量）决定响应速度，$C$（阻尼）抑制振荡，$K_a$（刚度）决定稳态位置偏差。
3. **方向选择**: 通过反馈权重矩阵 $S = \text{blkdiag}(s_T, s_R)$ 独立控制各方向的力反馈权重（如插入任务关闭侧向力反馈）。
4. **半隐式 Euler 积分**: 常用于实时实现，保持数值稳定性。

## 代表工作

- [[CW-WAM]]: 以预测接触力矩作为 $F^{\text{ref}}$，驱动 20 Hz 导纳控制器修正 WAM 名义轨迹

## 相关概念

- [[Contact-Wrench]]
- [[慢-快执行系统（Slow-Fast）]]
- [[Wrench Tracking Error]]
