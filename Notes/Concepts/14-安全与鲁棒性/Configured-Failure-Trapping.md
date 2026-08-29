---
type: concept
aliases: [Configured Failure Trapping, 配置化失效陷阱, CFT]
---

# Configured Failure Trapping

## 定义
针对 VLA 模型的新型后门攻击任务：通过隐蔽的文字触发器激活攻击，使机器人产生预先配置的特定失效行为（如把物体放到错误位置），而非随机失败。

## 核心要点
1. **比传统后门攻击更危险**：不是让机器人乱动，而是精准执行错误操作
2. 文字触发器（自然语言指令中的细微改变）难以被人工检测
3. 失效模式可配置：攻击者指定具体的错误行为而非随机失败
4. 评估指标：ASR（Attack Success Rate）+ 正常任务成功率（不能降太多）
5. 数据引擎 [[TrapEngine]] 自动生成带触发器的训练数据

## 代表工作
- [[TrapVLA]]: 提出 CFT 任务并设计 TrapEngine + TrapEval

## 相关概念
- [[VLA-Hijack]]
- [[LASI]]
- [[TrapEngine]]
