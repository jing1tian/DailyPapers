---
type: concept
aliases: [Koopman Partial World Model, Bilinear World Model]
---

# K-PWM

## 定义
基于 Koopman 算子的双线性世界模型，在 Koopman embedding 空间将非线性动力学线性化为双线性形式，支持部分可观测情况下的控制规划。

## 数学形式
$$z_{t+1} = A z_t + B(u_t) z_t$$

其中 $z_t$ 为 Koopman 嵌入，$A$ 为线性演化矩阵，$B(u_t)$ 为控制依赖的双线性项。

## 核心要点
1. Koopman embedding 将状态提升到高维线性空间
2. bilinear 形式比纯线性 Koopman 更准确，比非线性 WM 更易优化
3. GECM（Global Error Correction Module）在线修正模型误差
4. 配合 [[MPPI]] 做基于世界模型的规划
5. 在真实 Panda 机械臂上验证

## 代表工作
- [[K-PWM]]: 原始论文（2611.11521）

## 相关概念
- [[MPPI]]
- [[MuJoCo]]
- [[GECM]]
