---
type: concept
aliases: [WiLoR, Wrist-level LoRA Refinement]
---

# WiLoR

## 定义

WiLoR 是一种手部重建模型，能够从视频中精确估计 [[MANO]] 手部参数，独立精化手指姿态，为 4D HOI 重建提供高质量手部运动初始估计。

## 核心要点

1. 专注于手部细节，独立于全身姿态估计运行
2. 输出：MANO 参数（手部关节角度、手形参数）
3. 在 GRAIL 中与 GENMO 协同工作：GENMO 负责全身姿态，WiLoR 精化手指
4. 高精度手部估计有助于后续接触损失的优化

## 代表工作

- [[GRAIL]]: 使用 WiLoR 作为手部运动初始估计，配合联合优化重建自然手部接触

## 相关概念

- [[MANO]]
- [[GENMO]]
- [[SMPL-X]]
- [[抓取奖励]]
