---
type: concept
aliases: [Human-in-the-Loop, 人在回路]
---

# HIL

## 定义
Human-in-the-Loop：在自动化系统中保留人类干预接口的设计范式，使人类可以在关键时刻介入、纠正或引导机器人行为。

## 核心要点
1. 机器人自主执行，检测到不确定性时请求人类介入
2. 常用 value function 或 confidence 估计触发介入
3. 减少全自主部署的风险

## 代表工作
- [[ValueFormer]]: 用 value head 检测 VLA 故障并触发 HIL 接管

## 相关概念
- [[ValueFormer]]
- [[Semi-Autonomous Policy]]
