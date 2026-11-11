---
type: concept
aliases: [HumanoidMimicGen]
---

# HumanoidMimicGen

## 定义

面向全身规划的人形机器人 Loco-Manipulation 数据生成系统，通过修改参考轨迹的接触点来扩充数据多样性，是 MimicGen 在人形机器人全身操作场景下的扩展。

## 核心要点

1. 针对人形机器人 Loco-Manipulation 任务（需要腿部移动 + 手臂操作协调）
2. 通过自动化接触点增强，将少量参考轨迹扩展为大量多样化演示数据
3. 在 DreamVLA 中用于对 OmniRetarget 重定向后的参考轨迹进行接触点增强

## 代表工作

- HumanoidMimicGen (Lin et al., 2026, arXiv:2605.27724): 原始论文，全身规划的 Loco-Manipulation 数据生成

## 相关概念

- [[MimicGen]]
- [[Loco-Manipulation]]
- [[OmniRetarget]]
- [[Human-to-Robot Retargeting]]
- [[DreamVLA]]
