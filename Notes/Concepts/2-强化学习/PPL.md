---
type: concept
aliases: [Policy-Paced Learning, Policy-Paced Sampling, 策略驱动采样]
---

# PPL

## 定义
一种 world model 辅助的 RL 训练策略：用策略的不确定性动态调节虚拟 rollout（world model 生成）与真实机器人交互的采样比例，在策略不确定的状态空间区域多用 WM 采样，减少昂贵的真实机器人交互次数。

## 数学形式
采样比例动态调节：
$$
r_\text{WM}(s) = \sigma\left(\frac{H_\pi(s) - \bar{H}}{\tau}\right)
$$

其中 $H_\pi(s) = -\sum_a \pi(a|s)\log\pi(a|s)$ 为当前策略在状态 $s$ 的动作熵（不确定性代理），$\bar{H}$ 为历史平均熵，$\tau$ 为温度参数，$\sigma$ 为 sigmoid 函数。

## 核心要点
1. 真实机器人交互成本高（时间、磨损），PPL 通过不确定性驱动减少不必要的真实交互
2. 在策略已经稳定的状态下（低熵区域）无需额外 WM 采样，避免过拟合到 WM 的预测误差
3. 与 [[SERL]] 框架兼容：可在已有闭环 RL 系统上叠加 PPL 采样层
4. 配合 PINE（Policy-guided Importance Sampling）进一步纠正 WM 生成的 rollout 分布偏差

## 代表工作
- [[WorldSample]]: 提出 PPL + PINE + WMPO 的组合框架，在真实 Franka 机械臂上验证

## 相关概念
- [[SERL]]
- [[RLPD]]
- [[HIL-SERL]]
- [[World Action Model]]
- [[SAC]]
