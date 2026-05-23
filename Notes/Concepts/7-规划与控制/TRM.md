---
type: concept
aliases: [Trajectory Reachability Metric, 轨迹可达性度量]
---

# TRM

## 定义
用于 latent world model MPC 的规划度量，以"从当前状态出发在 H 步内可达目标的可达性"替代 Euclidean 距离作为候选动作序列的评分标准，解决 latent Euclidean 距离与实际行为距离不对应的问题。

## 数学形式
$$d_\text{TRM}(z_t, z_g; H) = -\log p_\phi(z_g \text{ reachable from } z_t \text{ in } H \text{ steps})$$

## 核心要点
1. Euclidean 距离在非线性 latent space 里不能反映"任务难度"
2. TRM 用 horizon-matched 可达性度量，与规划的实际 horizon 对齐
3. 在 PLDM/LeWM 框架上验证（PushT、TwoRoom 环境）
4. 对 latent MPC 的 planner 排序质量有直接改善

## 代表工作
- [[TRM]]：Li et al. 2026（Tongji University），Beyond Euclidean Proximity 论文的核心方法
- [[LeWM]]：TRM 的验证框架之一

## 相关概念
- [[Model Predictive Control]]
- [[World Model]]
- [[PLDM]]
