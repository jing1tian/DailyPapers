---
type: concept
aliases: [Visual Trace VLA]
---

# TraceVLA

## 定义
将历史动作轨迹可视化为"视觉轨迹"叠加在当前帧上，作为额外的空间-时间上下文送入 VLA，使策略可以感知运动历史，无需额外传感器或复杂记忆结构。

## 核心要点
1. 把过去 N 帧末端执行器的轨迹投影到当前图像坐标系
2. 轨迹以彩色线段/点的形式叠加在 RGB 图像上
3. 利用 VLA 已有的视觉理解能力处理历史信息
4. 对 chunked 控制的时序盲区问题有一定缓解

## 代表工作
- [[TraceVLA]]：对比基线，在 EvoScene-VLA 中被作为对比方法

## 相关概念
- [[VLA]]
- [[EvoScene-VLA]]
- [[MemoryVLA]]
