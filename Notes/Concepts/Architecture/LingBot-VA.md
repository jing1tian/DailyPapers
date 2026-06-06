---
type: concept
aliases: [LingBot-VA, LingBot VA]
---

# LingBot-VA

## 定义
LingBot-VA 是一种基于像素空间视频去噪的 [[Cascaded WAM]]，先通过完整的流匹配（Flow-Matching）去噪过程生成未来视频帧，再将视频帧作为条件生成机器人动作。

## 核心要点
1. **全程去噪**: 每次控制循环需要完整跑完视频去噪过程（~2868.4 ms on A100）
2. **高精度基线**: 在 RoboTwin 2.0 上取得 92.2% 的成功率，是 WAM 领域的强基线
3. **延迟瓶颈**: 高延迟使其难以直接用于实时控制，是 SANTS 等工作的优化目标

## 性能数据（RoboTwin 2.0）

| 指标 | 值 |
|------|---|
| Easy 成功率 | 92.9% |
| Hard 成功率 | 91.5% |
| Overall 成功率 | 92.2% |
| A100 延迟 | 2868.4 ms |

## 代表工作
- [[SANTS]]: 在 LingBot-VA 骨干上附加调度器，将成功率提升至 94.4%，延迟降至 523.7 ms
- [[Flash-WAM]]: 以 LingBot-VA 为教师模型，通过模态感知一致性蒸馏将推理压缩至单步（348ms），达到 85.5% 成功率

## 相关概念
- [[WAM]]: LingBot-VA 所属的模型类别
- [[Cascaded WAM]]: LingBot-VA 的架构范式
- [[流匹配]]: LingBot-VA 视频去噪使用的核心技术
