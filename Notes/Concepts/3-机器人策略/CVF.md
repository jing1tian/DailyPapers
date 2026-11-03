---
type: concept
aliases: [Cross-View Fine-tuning]
---

# CVF

## 定义
跨视角微调（Cross-View Fine-tuning）：在 human-to-robot 迁移学习中，对齐人类第一人称视角（egocentric）与机器人摄像头视角的方法，减少视角差异引起的 domain gap。

## 核心要点
1. 人类操作视频通常是第一人称或侧视图，机器人摄像头视角不同
2. CVF 通过视角变换或对比学习最小化跨视角的表征差异
3. 常与 human-to-robot policy transfer 方法结合使用

## 代表工作
- [[Nexus-VLA]]: 与 CVF 对比，提出层次化语义空间的跨模态对齐方案

## 相关概念
- [[ACE]]
- [[VLA]]
