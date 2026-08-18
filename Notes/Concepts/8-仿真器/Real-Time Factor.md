---
type: concept
aliases: [实时因子, RTF, Real Time Factor]
---

# Real-Time Factor（RTF）

## 定义

仿真器性能的核心度量指标：仿真内部流逝时间与实际挂钟时间的比值。RTF = 1 表示仿真以真实速度运行；RTF > 1 表示仿真快于实时（加速）；RTF < 1 表示仿真慢于实时（减速）。

## 数学形式

$$
\text{RTF} = \frac{t_{\text{sim}}}{t_{\text{wall}}}
$$

## 核心要点

1. **评测延迟影响**：在 [[ReflexBench]] 中，RTF 用于解耦仿真步进与机器人控制周期，从而精确控制推理延迟对任务成功率的影响
2. **同步 vs 异步推理**：
   - 同步：等待推理完成再执行动作，RTF 受推理速度约束
   - 异步：推理与执行并行，行动基于上一帧推理结果，允许系统以固定频率运行
3. **仿真标准用法**：MuJoCo、IsaacLab 等仿真器均使用 RTF 监控物理仿真性能

## 代表工作

- [[ReflexVLA]]：通过可配置 RTF 的延迟感知框架研究推理延迟对动态任务的影响

## 相关概念

- [[ReflexBench]]
- [[Asynchronous Inference]]
- [[Action Chunking]]
