---
type: concept
aliases: [sCM, continuous-time Consistency Model, 连续时间一致性模型]
---

# sCM（连续时间一致性模型）

## 定义

[[一致性蒸馏|一致性模型]]（CM）在离散步长 $\Delta t \to 0$ 极限下的连续时间版本，把"预测同一目标"的一致性约束转化为对教师 PF-ODE 轨迹切线方向的回归，切线通过 [[JVP]] 高效计算，避免离散时间一致性蒸馏（dCM）中显式的多步前向。

## 数学形式

$$
\mathcal{L}_{\text{sCM}}(\theta)=\mathbb{E}_{\bm{x}_{0},\bm{\epsilon},t}\left[\left\|\bm{F}_{\theta}(\bm{x}_{t},t)-\bm{F}_{\theta^{-}}(\bm{x}_{t},t)-\frac{\bm{g}}{\|\bm{g}\|_{2}^{2}+c}\right\|_{2}^{2}\right], \quad \bm{g}=w(t)\frac{\mathrm{d}\bm{f}_{\theta^{-}}(\bm{x}_{t},t)}{\mathrm{d}t}
$$

- $\bm{F}_\theta$: 自由形式学生网络（对应速度预测器 $\bm{v}_\theta$）
- $\bm{g}$: 归一化前的教师轨迹切线方向，由 [[JVP]] 计算
- $\theta^-$: $\theta$ 的 [[Stop-Gradient]] 版本
- $c$: 防止除零的稳定项常数

## 核心要点

1. **dCM 的连续极限**：dCM 需要显式求解一步 ODE 得到 $\hat{\bm{x}}_{t-\Delta t}$ 再做一致性回归；sCM 让 $\Delta t \to 0$，把整套约束变成对切线方向（导数）的回归，理论上更高效、训练更稳定
2. **JVP 加速**：切线 $\mathrm{d}\bm{f}_{\theta^-}/\mathrm{d}t$ 通过前向模式自动微分（[[JVP]]）一次前向计算得到，无需反向传播或多次前向
3. **归一化稳定训练**：用 $\bm{g}/(\|\bm{g}\|_2^2+c)$ 对切线方向归一化，防止梯度尺度爆炸/坍缩
4. **可推广为 CTM**：把"终点一致性"（$t \to 0$）推广为任意区间 $t \to s$ 的一致性轨迹映射，等价于 [[MeanFlow]] 的相对误差比形式

## 代表工作

- [[rCM]]: 将 sCM 与 DMD 联合训练，用于双向扩散模型蒸馏
- [[Causal-rCM]]: 首次实现 Teacher-Forcing 设定下的连续时间一致性模型（TF-sCM），基于自研 custom-mask FlashAttention-2 JVP 内核，相比离散时间 TF-dCM 收敛速度提升 10 倍；并发现 RF 原生形式优于 TrigFlow 包装形式

## 相关概念

- [[一致性蒸馏]]
- [[JVP]]
- [[MeanFlow]]
- [[Rectified Flow]]
- [[Stop-Gradient]]
