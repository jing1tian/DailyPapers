---
type: concept
aliases: [未来规划增益, FPG, Planning Gain]
---

# Future Planning Gain

## 定义

Future Planning Gain（未来规划增益）衡量在当前预测深度 $h$ 的基础上，**继续进行 rollout 所带来的规划质量提升量**。是自适应计算分配的核心评估指标。

## 数学形式

$$
B_{i,h}^{*} = \bigl[q_{i,h+1} - q_{i,h},\; \ldots,\; q_{i,H_a} - q_{i,h}\bigr]
$$

其中 $q_{i,h}$ 为场景 $i$ 在 prefix 深度 $h$ 时的规划质量（如 EPDMS 得分），$H_a$ 为最大允许深度。

代价调整版本：

$$
g_{i,h}^{*} = \max_{j \in \{h+1,\ldots,H_a\}} \Bigl[\bigl(q_{i,j} - q_{i,h}\bigr) - \lambda(c_j - c_h)\Bigr]
$$

## 核心要点

1. **场景自适应**：不同场景的最优 rollout 深度差异显著（简单路段偏好 $h=0$，密集场景偏好 $h=3$-$4$）
2. **代价权衡**：计算偏好参数 $\lambda$ 将计算代价折算为性能单位，统一比较
3. **在线预测**：通过 Latent Evaluator 在不运行完整 Planner 的情况下预测增益配置文件

## 代表工作

- [[RISE]]: 首次提出 Future Planning Gain 作为自适应 rollout 调度的决策依据

## 相关概念

- [[WAM]]
- [[Rollout Gate]]
