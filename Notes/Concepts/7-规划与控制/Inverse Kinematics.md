---
type: concept
aliases: [IK, 逆运动学]
---

# Inverse Kinematics

## 定义

Inverse Kinematics（逆运动学，IK）是给定末端执行器（end-effector）的目标位姿，求解对应关节角度配置的计算问题，是机器人操作控制中的核心底层模块。

## 数学形式

给定正向运动学映射 $f: \mathbf{q} \to \mathbf{x}$（关节角度 → 末端位姿），IK 求解：

$$
\mathbf{q}^* = \arg\min_{\mathbf{q}} \| f(\mathbf{q}) - \mathbf{x}_{\mathrm{target}} \|^2 + \lambda \| \mathbf{q} - \mathbf{q}_0 \|^2
$$

其中 $\mathbf{q}_0$ 为参考配置，$\lambda$ 为正则化权重。

## 核心要点

1. 高自由度机器人（7-DOF）的 IK 通常有无穷多解，需引入附加约束；
2. 解析解仅在特殊机构（球腕等）存在，一般用数值优化迭代求解；
3. Code-as-Policy 系统中 IK 是将高层代码（`move_to(target_pose)`）转为关节控制命令的关键桥梁；
4. Guarded contact + IK 组合可提供安全的接触保护运动。

## 代表工作

- [[CaPtain]]: IK 作为 22 个低级原语之一，将高层目标位姿转为关节指令

## 相关概念

- [[Code-as-Policy]]
- [[Vision-Language-Action]]
