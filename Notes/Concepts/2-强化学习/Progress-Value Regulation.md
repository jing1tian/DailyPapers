---
type: concept
aliases: [价值引导回滚, Value-Guided Rollback, Progress-Value Token, 进度价值调节]
---

# Progress-Value Regulation

## 定义

一种在线执行监控机制：通过蒙特卡洛方法估计当前状态-动作对的"执行进度价值" $\hat{c}(s_t, a_t)$，当预测价值低于学习到的阈值 $\delta$ 时，系统自动回退到缓存的高价值历史状态并重新采样动作，从而在不依赖人工干预的情况下检测并纠正执行偏差。

## 数学形式

$$
\hat{c}(s_t, a_t) = \sum_{i=0}^{H-t} \gamma^i \, r(s_{t+i}, a_{t+i})
$$

价值 token 在训练时以该蒙特卡洛回报为回归目标；推理时若 $\hat{c}(s_t, a_t) < \delta$，触发回滚：

$$
(s_t, a_t) \leftarrow (s_{t^\*}, \cdot), \quad t^* = \arg\max_{t' < t} \hat{c}(s_{t'}, a_{t'})
$$

## 核心要点

1. **自监督价值学习**: 价值目标来自执行轨迹的蒙特卡洛回报，不需要额外的人工标注或稀疏 reward 工程
2. **在线纠偏而非离线评估**: 与离线价值函数（如 [[GVL]]）不同，Progress-Value Regulation 在执行过程中实时触发回滚动作，是闭环控制机制
3. **阈值鲁棒性**: 阈值 $\delta$ 在较宽范围内表现稳定，默认 $\delta=0.20$ 时取得最佳成功率
4. **与导航领域 [[目标进展值|Goal-Progress Value]] 的区别**: 目标进展值衡量到目标点的几何/测地线距离，主要用于导航长程规划；Progress-Value Regulation 基于折扣回报 $\gamma$ 估计操作任务的执行价值，并直接驱动在线回滚动作，应用场景是机器人操作中的偏差检测与自我纠错

## 代表工作

- [[MV-WAM]]: 首次提出将 Progress-Value token 集成进 [[Mixture-of-Transformers|MoT]] 架构的 Action-Value Expert，结合价值引导回滚使真实双臂操作成功率达到 77.5%

## 相关概念

- [[目标进展值]]: 导航领域的类似价值信号，几何距离驱动而非回报驱动
- [[World Action Model]]: Progress-Value Regulation 的主要应用场景
- [[Advantage Estimation]]: 广义的价值/优势估计思想
- [[Cross-Modality Causal Masking]]: 价值 token 在 MV-WAM 中通过该掩码策略与视觉、动作 token 交互
