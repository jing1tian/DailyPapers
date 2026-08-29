---
type: concept
aliases: [TrapEngine, 陷阱数据引擎]
---

# TrapEngine

## 定义
为 Configured Failure Trapping 任务设计的数据生成引擎：从专家演示中提取元数据，生成带隐蔽文字触发器的指令和对应失效轨迹，用于训练后门 VLA 模型。

## 核心要点
1. 三阶段流程：① 元数据收集（末端位置、目标位置）→ ② 触发器注入 → ③ 失效轨迹合成
2. 生成的失效轨迹是配置化的（指定错误行为），不是随机错误
3. 配套评估工具 TrapEval 自动计算 ASR

## 代表工作
- [[TrapVLA]]: TrapEngine 原始设计

## 相关概念
- [[Configured Failure Trapping]]
