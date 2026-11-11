---
type: concept
aliases: [PSR, 部分成功率, 子任务完成率]
---

# Partial Success Rate（部分成功率）

## 定义

机器人操控任务评估指标，衡量模型在每个 Episode 中平均完成了多少比例的子任务，比全任务成功率（SR）更细粒度地反映策略能力。

## 数学形式

$$
\text{PSR} = \frac{1}{|T|}\sum_{i \in T} \frac{\text{completed subtasks}_i}{\text{total subtasks}_i}
$$

其中 $T$ 为所有测试 Episode 的集合。

对比，全任务成功率（SR）定义为：

$$
\text{SR} = \frac{1}{|T|}\sum_{i \in T} \mathbb{1}[\text{all subtasks completed in episode } i]
$$

## 核心要点

1. **比 SR 更敏感**: 当模型完成了大部分子任务但在最后一步失败时，SR=0 但 PSR 仍较高，能区分"完全失败"和"接近成功"
2. **需要子任务定义**: 任务必须预先分解为可验证的子任务（如杯子叠放分为 7 步，毛巾折叠分为 4 步）
3. **与 SR 配合使用**: SR 衡量端到端能力，PSR 衡量子步骤能力，共同评估策略

## 代表工作

- [[SWAM]]: 在三个真实操控任务（Cup Stacking 7步、Towel Folding 4步、Gift-in-Box 2步）上使用 SR 和 PSR 双指标评估
- [[RoboTwin 2.0]]: 仿真 Benchmark，定义了 50 个任务的子任务划分

## 相关概念

- [[RoboTwin 2.0]]
- [[World Action Model]]
- [[Bimanual Manipulation]]
