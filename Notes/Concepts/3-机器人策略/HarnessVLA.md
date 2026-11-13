---
type: concept
aliases: [Harness VLA]
---

# HarnessVLA

## 定义

用 coding agent（Codex 或 Claude Code）编排微调 VLA 策略的 agent 框架，属于"有训练策略的 agent"类方法，在 RoboTwin 2.0 随机化设置达到 58.4%。

## 核心要点

1. **有训练策略**：依赖微调的 VLA 进行接触级控制，agent 负责高层编排
2. **需要额外演示**：训练 VLA 需要 N 个演示，agent 需额外 1 个
3. **与 WICL 的差距**：RoboScript 随机化成功率 83.2%，高出 24.8 点

## 代表工作

- [[RoboScript]]: 对比 HarnessVLA，展示无训练策略的 agent 路线优势

## 相关概念

- [[VLA]]
- [[Coding Agent]]
- [[World In-Context Learning]]
