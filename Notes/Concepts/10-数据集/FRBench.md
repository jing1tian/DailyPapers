---
type: concept
aliases: [Failure-Recovery Benchmark, 失败恢复基准]
---

# FRBench (Failure-Recovery Benchmark)

## 定义

一个专为 VLA 失败恢复能力评估设计的标准化基准，包含 RoboTwin 仿真（46 任务）和真实双臂操作（4 任务），通过三阶段评估协议（Nominal/Error/Recovery）和四类标准化误差注入（E1–E4）将名义任务能力与恢复能力解耦评估。

## 数学形式

**仿真失败触发条件（Eq. 6）**:

$$
\mathcal{F}_t = \mathbb{1}(t > T_{max})
$$

其中 $T_{max}$ 为所有任务名义成功时长的最大值。

## 核心要点

1. **三阶段评估协议**: Nominal（干净/随机化条件）→ Error（结构化扰动注入）→ Recovery（策略自主恢复，无外部重试）。
2. **四类误差注入 (E1–E4)**:
   - E1: Premature Close（过早关夹爪）
   - E2: Grasp Slip（抓取滑落）
   - E3: Position Offset（位置偏移）
   - E4: Orientation Mismatch（朝向不匹配）
3. **数据统计 (FRBench-Sim)**: 共 23,453 episodes，17,061 名义成功 + 6,392 失败-恢复。
4. 触发基于语义阶段而非 wall-clock 时间，确保不同任务的扰动发生在相同任务阶段。

## 代表工作

- [[RePO-VLA]]: 提出 FRBench，作为失败恢复能力的标准评估基准

## 相关概念

- [[VLA（视觉-语言-动作模型）]]
- [[Recovery-Aware Initialization]]
- [[RoboTwin 2.0]]
