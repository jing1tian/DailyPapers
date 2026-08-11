---
type: concept
aliases: [DriveWAM, Driving World Action Model]
---

# DriveWAM

## 定义
早期 WAM-based 自动驾驶 planner，将 video generation 作为前向预测组件，推理时仍需运行 future frame generation。

## 核心要点
1. 将 world model 的 video generation 能力直接引入驾驶决策
2. 推理时 video branch 在线运行，计算成本高
3. 被 [[SimWAM]] 的 isolated attention + 训练后丢弃 video branch 策略超越

## 相关概念
- [[SimWAM]]
- [[DriveLaW]]
- [[WAM]]
