---
type: concept
aliases: [PUMA VLA]
---

# PUMA

## 定义

一种 4B 参数的视觉-语言-动作（VLA）模型，专注于机器人操控任务，在动态操控（如传送带拾取、滚球拦截）上表现突出，但推理延迟较高（>100ms），限制了其在高频控制场景中的应用。

## 核心要点

1. **参数规模**：4B，属于中型 VLA
2. **推理延迟**：高于 1B 的小型模型，是动态任务中的主要短板
3. **真实机器人表现**（AgileX Piper）：
   - 传送带拾取：13/20
   - PressButtons：20.8
   - CatchBalls：5.4
4. **对比地位**：常作为动态操控任务的强基线，ReflexVLA（1B）以 1/4 参数量媲美或超越 PUMA

## 代表工作

- [[ReflexVLA]]：以更小模型（1B vs 4B）在 [[ReflexBench]] 和真实机器人实验中与 PUMA 对比

## 相关概念

- [[Vision-Language-Action Model]]
- [[SmolVLA]]
- [[OpenVLA-OFT]]
- [[VLA-Adapter]]
