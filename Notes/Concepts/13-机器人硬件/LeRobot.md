---
type: concept
aliases: [LeRobot SO-101, HuggingFace LeRobot, lerobot]
---

# LeRobot

## 定义

HuggingFace 开源的低成本机器人学习框架，包含 SO-100/SO-101 等低成本机械臂平台、数据集工具链和策略训练基准，旨在降低真实机器人实验的门槛。

## 核心要点

1. **低成本硬件**: SO-101 机械臂使用标准 Dynamixel 舵机，整体成本约数百美元，远低于工业级机械臂。
2. **开源工具链**: 提供数据采集、遥操作、策略训练和评估的完整流程。
3. **真实机器人基准**: 被多篇 World Action Model 论文（如 JOPAT、UWM）用作真实环境评估平台。
4. **HuggingFace 集成**: 数据集和模型权重可直接通过 Hugging Face Hub 分享和复现。

## 代表工作

- [[JOPAT]]: 在 LeRobot SO-101 上完成 4 个真实任务评估（Cook-Soup, Insert-Peg, Push-Tomato, Pick-Grocery）

## 相关概念

- [[World Action Model]]
- [[扩散策略]]
