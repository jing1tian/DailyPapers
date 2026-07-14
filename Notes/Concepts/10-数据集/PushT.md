---
type: concept
aliases: [Push-T]
---

# PushT

## 定义
2D 操作任务 benchmark，机器人末端执行器将 T 形物体推到目标位置，是评测 diffusion policy 等操作策略的标准任务之一。

## 核心要点
1. 连续控制，末端执行器位置作为动作空间
2. 任务成功率通过 T 形物体与目标区域的重叠度量
3. 多模态行为（多种推法均可成功）使其成为测试多模态策略的良好场景
4. 常用于 Diffusion Policy、ACT 等操作策略的 ablation

## 代表工作
- [[Diffusion Policy]]：在 PushT 上首次展示扩散策略优势
- [[BSP]]：B-spline Policy 在 PushT 上的速度对比测试

## 相关概念
- [[Action Diffusion]]
- [[ACT]]
