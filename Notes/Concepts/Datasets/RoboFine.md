---
type: concept
aliases: [RoboFine, RoboFine Dataset]
---

# RoboFine

## 定义

RoboFine 是 FineVLA 论文提出的细粒度 VLA 指令对齐数据集，通过人工核验的方式为已有机器人演示（来自 RoboTwin、AlohaMix 等）添加细粒度行为约束指令（如速度、力度、姿态要求），支持训练可控的 VLA 策略。

## 核心要点

1. **细粒度标注**: 不只标注"完成任务"，还标注"如何完成"（如"轻柔抓取"、"竖直放置"）
2. **人工核验**: 指令由人工验证与演示动作一致性，保证数据质量
3. **多源数据**: 整合 RoboTwin、AlohaMix 等已有机器人演示数据集
4. **DTW 评估**: 配套 DTW 轨迹相似度评估指标，而非只看任务成功率

## 代表工作

- [[FineVLA]]: 提出并使用 RoboFine

## 相关概念

- [[RoboTwin]]
- [[LIBERO]]
- [[Diffusion Policy]]
