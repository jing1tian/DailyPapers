---
type: concept
aliases: [Timing Execution by Monitoring Progress Online]
---

# TempoWAM

## 定义
一种自适应执行时机决策方法，通过在线监测机器人执行进度来动态决定何时触发 WAM 重规划，替代固定步长执行的静态策略。

## 核心要点
1. **RPM（Replanning Progress Monitor）**：在线监测当前执行进度，检测任务阶段变化
2. **AEP（Adaptive Execution Protocol）**：根据 RPM 信号自适应决定何时停止当前 chunk 执行、触发重规划
3. **Plug-and-Play**：无需重训练，直接挂在现有 WAM（FastWAM、Motus 等）上运行
4. 在 LIBERO、RoboTwin 和真实机械臂上验证

## 核心问题解决
WAM 通常生成一个固定长度的动作 chunk，机器人执行固定前缀后重规划。这种固定步长在任务阶段转换时既浪费（简单阶段不需要频繁重规划）又危险（复杂阶段需要及时纠错）。

## 代表工作
- TempoWAM: Rethink Before You Execute: Adaptive Execution for World Action Models (2026)

## 相关概念
- [[WAM]]
- [[FastWAM]]
- [[LaWAM]]
- [[ProgressVLA]]
