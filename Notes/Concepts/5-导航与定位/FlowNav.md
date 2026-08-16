---
type: concept
aliases: [FlowNav, Flow-based Navigation]
---

# FlowNav

## 定义
利用 optical flow 预测作为导航策略额外监督信号的方法，提升跨 embodiment 泛化能力。

## 核心要点
1. 预测视频流并将其作为 waypoint 生成的 auxiliary loss
2. Flow 信号提供了与 embodiment 无关的运动先验
3. 被 [[CrossTracer]] 作为跨平台导航的对比 baseline

## 相关概念
- [[NaviTrace]]
- [[CrossTracer]]
