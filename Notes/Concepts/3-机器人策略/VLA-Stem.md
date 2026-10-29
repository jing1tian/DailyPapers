---
type: concept
aliases: [VLA-Stem, Stem Policy, VLA Safety Stem]
---

# VLA-Stem

## 定义

一种轻量级即插即用安全机制：冻结 VLA 策略不变，通过独立训练的 Stem 策略将 VLA 输出的动作块转化为相对末端位姿参考轨迹，结合本体感知与障碍信息，用 RL（PPO + 课程学习 + 教师-学生蒸馏）实现运动时的实时障碍规避。

## 数学形式

$$
a_t^{\mathrm{exec}} = \pi_{\mathrm{stem}}\!\left(\mathbf{x}_t,\, \mathbf{o}_t^{\mathrm{obs}},\, \tau_c^{\mathrm{rel}}\right), \quad t_c \le t < t_c + h
$$

其中 $\tau_c^{\mathrm{rel}} = \{(T_{t_c}^w)^{-1} T_{t_c+k}^w\}_{k=1}^h$ 为 VLA 动作块转换的相对参考轨迹，$h=8$。

## 核心要点

1. **双策略解耦**: VLA 策略完全冻结，Stem 策略独立训练，适配任意 VLA
2. **轨迹级引导**: 前向走廊匹配（Forward Corridor Matching）软约束时序，兼顾安全性与灵活性
3. **三阶段课程**: 轨迹跟随 → 静态障碍 → 动态障碍，逐步增加难度
4. **教师-学生蒸馏**: 教师用特权信息（目标位置）辅助训练，学生蒸馏后部署

## 代表工作

- [[VLA-Stem]]: 提出该方法（arXiv 2610.37378）

## 相关概念

- [[Action Chunking]]: VLA 输出动作块，Stem 的引导信号来源
- [[PPO]]: Stem 策略的训练算法
- [[课程学习]]: 三阶段难度递增训练
- [[On-Policy Distillation]]: 教师-学生蒸馏框架
- [[CBF]]: 对比基线 VLSA 使用的方法
- [[π0.5]]: 默认 VLA 基础模型
