---
type: concept
aliases: [运行时批评器, Runtime Critic, 高频批评器]
---

# Runtime Critics

## 定义

在机器人策略执行期间以**动作频率**运行的轻量级检测模块，通过分析历史轨迹实时判断物理状态是否偏离正常分布并触发干预，是实现在线自演化的核心机制。

## 数学形式

$$
P_t = \mathcal{C}(\tau_{0:t}) = \langle e_t,\; \hat{\sigma}_t \rangle
$$

- $\tau_{0:t}$：从起始到当前时刻的完整轨迹历史
- $e_t$：可审计的失败证据（视觉、物理信号）
- $\hat{\sigma}_t$：建议执行模式（继续 / 触发恢复）

## 核心要点

1. **动作频率运行**: 区别于事后反思 (post-hoc reflection)，批评器在每个动作步执行，能在物理偏差发生时立即介入
2. **结构化输出**: 输出可审计证据 $e_t$，支持后续因果诊断和人工复查
3. **独立于策略参数**: 作为 [[Evolvable Harness]] 的组件，批评器本身可被演化代理更新，而策略权重保持冻结
4. **语义-物理鸿沟弥合**: 弥补 VLA 策略感知语义但忽略物理细节的结构性缺陷

## 与 VLM Critic 的区别

| | [[VLM Critic]] | Runtime Critics |
|--|--|--|
| 运行时机 | Episode 结束后 | 动作执行中（实时） |
| 运行频率 | 低频（per-episode） | 高频（per-action） |
| 干预方式 | 指导下一轮尝试 | 即时触发恢复动作 |

## 代表工作

- [[Zetta]]: 提出 Runtime Critics 作为可演化封套核心组件，实现 LIBERO-Pro 34.5%→90.8%

## 相关概念

- [[Evolvable Harness]]
- [[Closed-Loop Control]]
- [[Hierarchical Causal Diagnosis]]
- [[VLM Critic]]
