---
type: concept
aliases: [Time-Optimal Path Parameterization via Reachability Analysis]
---

# TOPPRA

## 定义
基于可达性分析的时间最优路径参数化算法，给定几何路径和运动学/动力学约束，输出最快可行的时间参数化轨迹。

## 核心要点
1. 比 TOPP 更稳定：用 reachability analysis 避免数值奇异性
2. 支持速度、加速度、扭矩等多种约束类型
3. 时间复杂度 $O(N)$，适合实时运动规划
4. [[cuRoboV2-Fast]] 用其做力矩限制的轨迹重参数化

## 代表工作
- [[cuRoboV2-Fast]]: 集成 TOPPRA 做力矩约束轨迹 retiming

## 相关概念
- [[TimeOpt]]
- [[CuRobo]]
- [[VAMP]]
