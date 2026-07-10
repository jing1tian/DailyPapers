---
type: concept
aliases: [Signal Temporal Logic, 信号时序逻辑]
---

# STL

## 定义
一种形式化规范语言，用于描述连续时间信号随时间变化的时序性质，可以表达"在 t1 到 t2 时间段内，某信号值必须满足某条件"这类约束。

## 数学形式
STL 公式递归定义：
$$\varphi ::= \top \mid \mu \mid \neg\varphi \mid \varphi_1 \wedge \varphi_2 \mid \mathbf{G}_{[a,b]}\varphi \mid \mathbf{F}_{[a,b]}\varphi \mid \varphi_1 \mathbf{U}_{[a,b]}\varphi_2$$

其中 $\mathbf{G}$ 为"全局（always）"，$\mathbf{F}$ 为"最终（eventually）"，$\mathbf{U}$ 为"直到（until）"。

## 核心要点
1. 相比 LTL（线性时序逻辑），STL 处理连续信号而非离散命题
2. 支持鲁棒度（robustness）量化：公式满足程度可以用实数衡量，方便梯度优化
3. 在机器人规划中用于表达"在某时间窗口内到达某位置"等约束

## 代表工作
- [[EvoPlan]]: 用 STL 作为时空约束规范，结合进化算法验证规划轨迹满足约束

## 相关概念
- [[PDDL]]
- [[MPC]]
