---
type: concept
aliases: [因子化策略, 分解策略, Factorized World Model Policy]
---

# Factorized Policy

## 定义
因子化策略是将操作策略 $\pi_\theta$ 分解为**状态预测**和**动作生成**两个子问题乘积的框架，通过对中间世界状态轨迹积分将两者解耦。

## 数学形式

$$
\pi_\theta(a_{1:H} \mid x_0, c) = \int p_\theta(a_{1:H} \mid x_{1:F}, x_0, c) \cdot p_\theta(x_{1:F} \mid x_0, c) \, dx_{1:F}
$$

其中：
- $p_\theta(x_{1:F} \mid x_0, c)$：世界模型（预测未来状态）
- $p_\theta(a_{1:H} \mid x_{1:F}, x_0, c)$：动作模型（基于未来状态生成动作）

## 核心要点

1. **理论分解**: 世界状态轨迹 $x_{1:F}$ 是连接两个子问题的充分统计量
2. **World Action Model 的核心**: 大多数 WAM 方法都以此为出发点
3. **训练信号丰富**: 世界模型可以从无标注视频中学习，动作模型从少量机器人数据学习
4. **推断时积分**: 实践中通过联合去噪（[[Flow Matching]]）近似积分

## 代表工作
- [[PrismWAM]]: 进一步将 $x_{1:F}$ 分解为 $(z_R, z_D)$，实现三阶段因子化
- [[World Action Model]]: 基于此框架的通用范式

## 相关概念
- [[World Action Model]]
- [[Decoupled Policy]]
- [[Flow Matching]]
